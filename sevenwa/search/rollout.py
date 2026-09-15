"""Rollout-based evaluator: uniform priors + Monte-Carlo playout values (no network).

Useful as (a) a strong classical baseline (MCTS with playouts), (b) a bootstrap data
source before the network has learned anything, and (c) a way to unit-test the tree
search independently of PyTorch.  A ``policy_fn(state, rng) -> action`` can be
supplied to bias playouts (e.g. the heuristic agent), which greatly strengthens the
baseline compared with uniformly random playouts.
"""
from __future__ import annotations

from bisect import bisect_right
from itertools import accumulate
from math import tanh
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np

from ..game import CHANCE, GameState

PolicyFn = Callable[[GameState, np.random.Generator], int]
Returns = Tuple[float, float]


def random_policy(state: GameState, rng: np.random.Generator) -> int:
    la = state.legal_actions()
    return int(la[rng.integers(len(la))])


def sample_outcome(outcomes: Sequence[Tuple[int, float]], rng: np.random.Generator) -> int:
    """Sample an outcome id from ``(id, probability)`` pairs: one uniform draw + bisect on the
    cumulative probabilities (same distribution as ``rng.choice(p=...)``, ~10x cheaper)."""
    cum = list(accumulate(p for _, p in outcomes))
    idx = bisect_right(cum, rng.random() * cum[-1])
    if idx >= len(cum):  # only through floating-point slack at the top end
        idx = len(cum) - 1
    return outcomes[idx][0]


def playout(state: GameState, rng: np.random.Generator, policy_fn: PolicyFn = random_policy,
            max_plies: int = 2000, return_margin: bool = False) -> Union[Returns, Tuple[Returns, float]]:
    """Play the game to the end from ``state`` and return the terminal returns.

    With ``return_margin=True`` the result is ``(returns, margin)`` where ``margin`` is the final
    ``score_diff()`` (player 0 minus player 1) of the terminal position.
    """
    s = state
    for _ in range(max_plies):
        if s.is_terminal():
            r = s.returns()
            return (r, float(s.score_diff())) if return_margin else r
        if s.to_move() == CHANCE:
            s = s.apply_chance(sample_outcome(s.chance_outcomes(), rng))
        else:
            s = s.apply_action(policy_fn(s, rng))
    # Non-terminating playout: fall back to the partial score difference.
    d = float(s.score_diff())
    v = float(np.tanh(d / 10.0))
    return ((v, -v), d) if return_margin else (v, -v)


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

    ``value = (1 - lam) * v_net + lam * mean(playout values)``.  With a weak or over-fitted value
    head the playouts supply a grounded, unbiased (if noisy) estimate; with ``lam = 0`` this is the
    plain network evaluator.  Playouts cost ~4 ms each, so use modest simulation counts.

    ``margin_weight`` (``w``, default 0 = plain win/loss results) blends the playout's final score
    margin into each playout value: ``(1 - w) * result + w * tanh(margin / margin_scale)`` with the
    margin from the mover's perspective — a lower-variance signal than the ±1 result alone.
    """

    def __init__(self, net_evaluator, num_actions: int, lam: float = 0.5, playouts_per_leaf: int = 1,
                 policy_fn: Optional[PolicyFn] = None, rng: Optional[np.random.Generator] = None,
                 margin_weight: float = 0.0, margin_scale: float = 8.0):
        self.net = net_evaluator
        self.num_actions = num_actions
        self.lam = lam
        self.playouts = playouts_per_leaf
        self.rng = rng or np.random.default_rng()
        self.margin_weight = float(margin_weight)
        self.margin_scale = float(margin_scale)
        if policy_fn is None:
            from ..agents.heuristic import heuristic_action
            policy_fn = heuristic_action
        self.policy_fn = policy_fn

    def playout_value(self, state: GameState) -> float:
        """One playout from ``state``, valued from the mover's perspective."""
        mover = state.to_move()
        w = self.margin_weight
        if w <= 0.0:
            r = playout(state, self.rng, self.policy_fn)
            return r[mover] if mover in (0, 1) else r[0]
        r, m = playout(state, self.rng, self.policy_fn, return_margin=True)
        if mover == 1:
            res, margin = r[1], -m
        else:
            res, margin = r[0], m
        return (1.0 - w) * res + w * tanh(margin / self.margin_scale)

    def evaluate(self, states: Sequence[GameState]) -> Tuple[np.ndarray, np.ndarray]:
        pols, vals = self.net.evaluate(states)
        vals = np.array(vals, dtype=np.float32)
        if self.lam > 0:
            for i, s in enumerate(states):
                tot = 0.0
                for _ in range(self.playouts):
                    tot += self.playout_value(s)
                vals[i] = (1 - self.lam) * vals[i] + self.lam * tot / self.playouts
        return pols, vals
