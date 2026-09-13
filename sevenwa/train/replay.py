"""Replay buffer of self-play samples stored as compact numpy shards.

Each sample: features (float32[F]), legal mask (bool[A]), policy target (float32[A]),
value target z in {-1,0,1} from the mover's perspective, score margin (float32) from
the mover's perspective, and the generation index that produced it (for windowing).
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Batch:
    feats: np.ndarray
    mask: np.ndarray
    pi: np.ndarray
    z: np.ndarray
    margin: np.ndarray


class ReplayBuffer:
    def __init__(self, feature_dim: int, num_actions: int):
        self.F = feature_dim
        self.A = num_actions
        self.feats: List[np.ndarray] = []
        self.mask: List[np.ndarray] = []
        self.pi: List[np.ndarray] = []
        self.z: List[np.ndarray] = []
        self.margin: List[np.ndarray] = []
        self.gen: List[np.ndarray] = []

    def __len__(self) -> int:
        return int(sum(len(z) for z in self.z))

    def add_game(self, feats, mask, pi, z, margin, generation: int) -> None:
        n = len(z)
        if n == 0:
            return
        self.feats.append(np.asarray(feats, dtype=np.float32).reshape(n, self.F))
        self.mask.append(np.asarray(mask, dtype=bool).reshape(n, self.A))
        self.pi.append(np.asarray(pi, dtype=np.float32).reshape(n, self.A))
        self.z.append(np.asarray(z, dtype=np.float32).reshape(n))
        self.margin.append(np.asarray(margin, dtype=np.float32).reshape(n))
        self.gen.append(np.full(n, generation, dtype=np.int32))

    def consolidate(self) -> None:
        if len(self.z) > 1:
            self.feats = [np.concatenate(self.feats)]
            self.mask = [np.concatenate(self.mask)]
            self.pi = [np.concatenate(self.pi)]
            self.z = [np.concatenate(self.z)]
            self.margin = [np.concatenate(self.margin)]
            self.gen = [np.concatenate(self.gen)]

    def keep_generations(self, min_generation: int) -> None:
        """Drop samples older than ``min_generation`` (sliding window)."""
        self.consolidate()
        if not self.z:
            return
        keep = self.gen[0] >= min_generation
        self.feats = [self.feats[0][keep]]
        self.mask = [self.mask[0][keep]]
        self.pi = [self.pi[0][keep]]
        self.z = [self.z[0][keep]]
        self.margin = [self.margin[0][keep]]
        self.gen = [self.gen[0][keep]]

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        self.consolidate()
        n = len(self)
        idx = rng.integers(0, n, size=min(batch_size, n))
        return Batch(self.feats[0][idx], self.mask[0][idx], self.pi[0][idx], self.z[0][idx], self.margin[0][idx])

    def iterate(self, batch_size: int, rng: np.random.Generator):
        self.consolidate()
        n = len(self)
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            yield Batch(self.feats[0][idx], self.mask[0][idx], self.pi[0][idx], self.z[0][idx], self.margin[0][idx])

    # ---- persistence ------------------------------------------------------
    def save_shard(self, path: str) -> None:
        self.consolidate()
        if not self.z:
            return
        np.savez_compressed(path, feats=self.feats[0], mask=self.mask[0], pi=self.pi[0], z=self.z[0],
                            margin=self.margin[0], gen=self.gen[0])

    @staticmethod
    def save_game_shard(path: str, feats, mask, pi, z, margin, generation: int) -> None:
        n = len(z)
        np.savez_compressed(path, feats=np.asarray(feats, np.float32), mask=np.asarray(mask, bool),
                            pi=np.asarray(pi, np.float32), z=np.asarray(z, np.float32),
                            margin=np.asarray(margin, np.float32), gen=np.full(n, generation, np.int32))

    def load_shards(self, pattern: str, min_generation: int = 0) -> int:
        added = 0
        for p in sorted(glob.glob(pattern)):
            d = np.load(p)
            g = d["gen"]
            keep = g >= min_generation
            if not keep.any():
                continue
            self.feats.append(d["feats"][keep]); self.mask.append(d["mask"][keep]); self.pi.append(d["pi"][keep])
            self.z.append(d["z"][keep]); self.margin.append(d["margin"][keep]); self.gen.append(g[keep])
            added += int(keep.sum())
        return added

    def stats(self) -> dict:
        self.consolidate()
        if not self.z:
            return {"samples": 0}
        z = self.z[0]
        return {"samples": int(len(z)), "win_rate_mover": float((z > 0).mean()), "draw_rate": float((z == 0).mean()),
                "mean_margin": float(self.margin[0].mean()), "generations": [int(self.gen[0].min()), int(self.gen[0].max())]}
