"""Supervised update of the network on replay samples (AlphaZero loss)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

from ..nn.model import PolicyValueNet, compute_loss
from .replay import ReplayBuffer


@dataclass
class TrainConfig:
    batch_size: int = 256
    epochs: float = 1.0  # passes over the buffer per generation (fractional allowed)
    lr: float = 1e-3
    min_lr: float = 1e-4
    weight_decay: float = 1e-4
    score_weight: float = 0.25
    grad_clip: float = 1.0
    max_steps: Optional[int] = None
    num_threads: int = 4


class Trainer:
    def __init__(self, net: PolicyValueNet, cfg: TrainConfig, device: str = "cpu"):
        self.net = net.to(device)
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.total_steps = 0

    def train(self, buffer: ReplayBuffer, rng: np.random.Generator, log=print) -> Dict[str, float]:
        cfg = self.cfg
        torch.set_num_threads(cfg.num_threads)
        n = len(buffer)
        if n == 0:
            return {"steps": 0}
        steps = int(math.ceil(cfg.epochs * n / cfg.batch_size))
        if cfg.max_steps is not None:
            steps = min(steps, cfg.max_steps)
        self.net.train()
        agg: Dict[str, float] = {}
        for step in range(steps):
            lr = cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * step / max(1, steps)))
            for g in self.opt.param_groups:
                g["lr"] = lr
            b = buffer.sample(cfg.batch_size, rng)
            feats = torch.from_numpy(b.feats).to(self.device)
            mask = torch.from_numpy(b.mask).to(self.device)
            pi = torch.from_numpy(b.pi).to(self.device)
            z = torch.from_numpy(b.z).to(self.device)
            margin = torch.from_numpy(b.margin).to(self.device)
            loss, parts = compute_loss(self.net, feats, mask, pi, z, margin, cfg.score_weight)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.grad_clip)
            self.opt.step()
            self.total_steps += 1
            parts["total"] = float(loss)
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v
            if (step + 1) % max(1, steps // 5) == 0 or step == steps - 1:
                log(f"  step {step + 1}/{steps} loss {float(loss):.4f} policy {parts['policy']:.4f} "
                    f"value {parts['value']:.4f} score {parts['score']:.4f} acc {parts['acc']:.3f} lr {lr:.2e}")
        self.net.eval()
        return {k: v / steps for k, v in agg.items()} | {"steps": steps, "samples": n}
