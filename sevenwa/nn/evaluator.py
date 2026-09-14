"""Batched torch evaluator implementing :class:`sevenwa.search.mcts.Evaluator`.

* Encodes states with a user-supplied ``encode(state) -> (features, legal_mask)`` function.
* Runs the network in ``torch.no_grad`` on CPU (1 thread per worker by default).
* Caches results by ``state.key()`` (LRU) — chance sampling reaches identical states often.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch

from ..game import GameState
from .model import PolicyValueNet

EncodeFn = Callable[[GameState], Tuple[np.ndarray, np.ndarray]]


class TorchEvaluator:
    def __init__(self, net: PolicyValueNet, encode: EncodeFn, cache_size: int = 200_000,
                 device: str = "cpu", num_threads: Optional[int] = None, value_temperature: float = 1.0):
        self.net = net.to(device).eval()
        self.encode = encode
        self.device = device
        # Calibration: v' = tanh(atanh(v) / T).  T > 1 shrinks over-confident values towards 0
        # (fitted on held-out data: T ≈ 1.4-1.5 for the imitation networks).
        self.value_temperature = float(value_temperature)
        self.cache: "OrderedDict[bytes, Tuple[np.ndarray, float]]" = OrderedDict()
        self.cache_size = cache_size
        self.calls = 0
        self.hits = 0
        if num_threads is not None:
            torch.set_num_threads(num_threads)

    def clear_cache(self) -> None:
        self.cache.clear()

    @torch.no_grad()
    def evaluate(self, states: Sequence[GameState]) -> Tuple[np.ndarray, np.ndarray]:
        B = len(states)
        pols = [None] * B
        vals = np.zeros(B, dtype=np.float32)
        todo = []
        keys = []
        for i, s in enumerate(states):
            k = s.key()
            keys.append(k)
            hit = self.cache.get(k)
            if hit is not None:
                self.hits += 1
                self.cache.move_to_end(k)
                pols[i], vals[i] = hit
            else:
                todo.append(i)
        self.calls += B
        if todo:
            enc = [self.encode(states[i]) for i in todo]
            feats = torch.from_numpy(np.stack([e[0] for e in enc]).astype(np.float32)).to(self.device)
            masks = torch.from_numpy(np.stack([e[1] for e in enc]).astype(bool)).to(self.device)
            logits, v, _ = self.net(feats)
            logits = logits.masked_fill(~masks, -1e9)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            if self.value_temperature != 1.0:
                v = torch.tanh(torch.atanh(v.clamp(-0.999, 0.999)) / self.value_temperature)
            v = v.cpu().numpy()
            for j, i in enumerate(todo):
                pols[i] = probs[j]
                vals[i] = v[j]
                self.cache[keys[i]] = (probs[j], float(v[j]))
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        return np.stack(pols), vals
