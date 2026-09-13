"""Rollout-based evaluator: uniform priors + Monte-Carlo playout values (no network).

Useful as (a) a strong classical baseline (MCTS with playouts), (b) a bootstrap data
source before the network has learned anything, and (c) a way to unit-test the tree
search independently of PyTorch.  A ``policy_fn(state, rng) -> action`` can be
supplied to bias playouts (e.g. the heuristic agent), which greatly strengthens the
baseline compared with uniformly random playouts.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from ..game import CHANCE, GameState

PolicyFn = Callable[[GameState, np.random.Generator], int]


def random_policy(state: GameState, rng: np.random.Generator) -> int:
    la = state.legal_actions()
    return int(la[rng.integers(len(la))])


def playout(state: GameState, rng: np.random.Generator, policy_fn: PolicyFn = random_policy,
            max_plies: int = 2000) -> Tuple[float, float]:
    """Play the game to the end from ``state`` and return the terminal returns."""
    s = state
    for _ in range(max_plies):
        if s.is_terminal():
            return s.returns()
        if s.to_move() == CHANCE:
            outs = s.chance_outcomes()
            probs = np.fromiter((p for _, p in outs), dtype=np.float64, count=len(outs))
            probs /= probs.sum()
            idx = int(rng.choice(len(outs), p=probs))
            s = s.apply_chance(outs[idx][0])
        else:
            s = s.apply_action(policy_fn(s, rng))
    # Non-terminating playout: fall back to the partial score difference.
    d = s.score_diff()
    v = float(np.tanh(d / 10.0))
    return (v, -v)


class RolloutEvaluator:
    """Evaluator with uniform priors and averaged playout values."""

    def __init__(self, num_actions: int, playouts_per_leaf: int = 1,
                 policy_fn: PolicyFn = random_policy, rng: Optional[np.random.Generator] = None,
                 prior_fn: Optional[Callable[[GameState], np.ndarray]] = None):
        self.num_actions = num_actions
        self.playouts = playouts_per_leaf
        self.policy_fn = policy_fn
        self.rng = rng or np.random.default_rng()
        self.prior_fn = prior_fn

    def evaluate(self, states: Sequence[GameState]) -> Tuple[np.ndarray, np.ndarray]:
        B = len(states)
        pols = np.full((B, self.num_actions), 1.0 / self.num_actions, dtype=np.float32)
        vals = np.zeros(B, dtype=np.float32)
        for i, s in enumerate(states):
            if self.prior_fn is not None:
                pols[i] = self.prior_fn(s)
            mover = s.to_move()
            tot = 0.0
            for _ in range(self.playouts):
                r = playout(s, self.rng, self.policy_fn)
                tot += r[mover] if mover in (0, 1) else r[0]
            vals[i] = tot / self.playouts
        return pols, vals


class HybridEvaluator:
    """Network priors and value blended with heuristic playouts (AlphaGo-style mixing).

    ``value = (1 - lam) * v_net + lam * mean(playout results)``.  With a weak or over-fitted value
    head the playouts supply a grounded, unbiased (if noisy) estimate; with ``lam = 0`` this is the
    plain network evaluator.  Playouts cost ~4 ms each, so use modest simulation counts.
    """

    def __init__(self, net_evaluator, num_actions: int, lam: float = 0.5, playouts_per_leaf: int = 1,
                 policy_fn: Optional[PolicyFn] = None, rng: Optional[np.random.Generator] = None):
        self.net = net_evaluator
        self.num_actions = num_actions
        self.lam = lam
        self.playouts = playouts_per_leaf
        self.rng = rng or np.random.default_rng()
        if policy_fn is None:
            from ..agents.heuristic import heuristic_action
            policy_fn = heuristic_action
        self.policy_fn = policy_fn

    def evaluate(self, states: Sequence[GameState]) -> Tuple[np.ndarray, np.ndarray]:
        pols, vals = self.net.evaluate(states)
        vals = np.array(vals, dtype=np.float32)
        if self.lam > 0:
            for i, s in enumerate(states):
                mover = s.to_move()
                tot = 0.0
                for _ in range(self.playouts):
                    r = playout(s, self.rng, self.policy_fn)
                    tot += r[mover] if mover in (0, 1) else r[0]
                vals[i] = (1 - self.lam) * vals[i] + self.lam * tot / self.playouts
        return pols, vals
