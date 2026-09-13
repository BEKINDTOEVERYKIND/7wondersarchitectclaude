"""Policy / value / auxiliary-score network.

Two trunks are available:

* ``resmlp`` (default): pre-LayerNorm residual MLP.  ~1.5M parameters at width 512 ×
  6 blocks; ~2 ms per batch of 8 on one CPU thread — the sweet spot for CPU self-play.
* ``attention``: the flat feature vector is split into *entity* groups (see
  ``NetConfig.entity_slices``) that are embedded and processed by a small Transformer
  encoder before pooling.  Better inductive bias for the per-deck histograms, but ~3×
  slower; recommended when a GPU is available.

Heads:
* ``policy``: logits over the global action space (masked outside the model).
* ``value``: tanh scalar, expected game result for the player to move.
* ``score``: linear scalar, predicted final score margin (mover − opponent) / scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NetConfig:
    input_dim: int
    num_actions: int
    trunk: str = "resmlp"  # "resmlp" | "attention"
    width: int = 512
    depth: int = 6
    dropout: float = 0.0
    # attention trunk
    entity_slices: List[Tuple[int, int]] = field(default_factory=list)
    attn_dim: int = 128
    attn_heads: int = 4
    attn_layers: int = 2
    score_scale: float = 20.0  # score-margin target is divided by this


class ResBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 2)
        self.ln2 = nn.LayerNorm(width * 2)
        self.fc2 = nn.Linear(width * 2, width)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(F.gelu(self.ln1(x)))
        h = self.fc2(F.gelu(self.ln2(h)))
        return x + self.drop(h)


class ResMLPTrunk(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.inp = nn.Linear(cfg.input_dim, cfg.width)
        self.blocks = nn.ModuleList([ResBlock(cfg.width, cfg.dropout) for _ in range(cfg.depth)])
        self.ln = nn.LayerNorm(cfg.width)
        self.out_dim = cfg.width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return F.gelu(self.ln(h))


class AttentionTrunk(nn.Module):
    """Entity-wise embedding + Transformer encoder + mean/max pooling."""

    def __init__(self, cfg: NetConfig):
        super().__init__()
        if not cfg.entity_slices:
            raise ValueError("attention trunk requires NetConfig.entity_slices")
        self.slices = cfg.entity_slices
        self.embed = nn.ModuleList([nn.Sequential(nn.Linear(e - s, cfg.attn_dim), nn.GELU(),
                                                  nn.Linear(cfg.attn_dim, cfg.attn_dim))
                                    for s, e in self.slices])
        self.type_emb = nn.Parameter(torch.randn(len(self.slices), cfg.attn_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(cfg.attn_dim, cfg.attn_heads, cfg.attn_dim * 4,
                                           dropout=cfg.dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, cfg.attn_layers, enable_nested_tensor=False)
        self.global_fc = nn.Linear(cfg.input_dim, cfg.attn_dim)
        self.out = nn.Sequential(nn.Linear(cfg.attn_dim * 3, cfg.width), nn.GELU(), nn.LayerNorm(cfg.width))
        self.out_dim = cfg.width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        toks = [emb(x[:, s:e]) + self.type_emb[i] for i, ((s, e), emb) in enumerate(zip(self.slices, self.embed))]
        h = torch.stack(toks, dim=1)  # (B, E, D)
        h = self.encoder(h)
        pooled = torch.cat([h.mean(1), h.amax(1), F.gelu(self.global_fc(x))], dim=-1)
        return self.out(pooled)


class PolicyValueNet(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.cfg = cfg
        self.trunk = ResMLPTrunk(cfg) if cfg.trunk == "resmlp" else AttentionTrunk(cfg)
        d = self.trunk.out_dim
        self.policy_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, cfg.num_actions))
        self.value_head = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 1))
        self.score_head = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy_head(h), torch.tanh(self.value_head(h)).squeeze(-1), self.score_head(h).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ----- persistence -------------------------------------------------
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        torch.save({"config": self.cfg.__dict__, "state_dict": self.state_dict(), "extra": extra or {}}, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "PolicyValueNet":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = NetConfig(**ckpt["config"])
        net = cls(cfg)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return net


def compute_loss(net: PolicyValueNet, feats: torch.Tensor, mask: torch.Tensor, pi: torch.Tensor,
                 z: torch.Tensor, margin: torch.Tensor, score_weight: float = 0.25):
    """AlphaZero-style loss: CE(policy) + MSE(value) + w * MSE(score margin)."""
    logits, v, s = net(feats)
    logits = logits.masked_fill(~mask, -1e9)
    logp = F.log_softmax(logits, dim=-1)
    policy_loss = -(pi * logp).sum(-1).mean()
    value_loss = F.mse_loss(v, z)
    score_loss = F.mse_loss(s, margin / net.cfg.score_scale)
    total = policy_loss + value_loss + score_weight * score_loss
    with torch.no_grad():
        acc = (logits.argmax(-1) == pi.argmax(-1)).float().mean()
    return total, {"policy": policy_loss.item(), "value": value_loss.item(), "score": score_loss.item(),
                   "acc": acc.item()}
