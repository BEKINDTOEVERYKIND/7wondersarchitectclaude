"""A tiny stochastic two-player race game used to unit-test the search machinery.

Rules: players alternate.  Action 0 (SAFE) adds 1 point.  Action 1 (RISKY) leads to a
chance node: with probability ``p_hit`` you gain 3 points, otherwise 0.  The first
player to reach ``target`` wins.  Optimal play depends on the score gap, so a correct
chance-aware search must prefer RISKY when far behind and SAFE when it wins outright.
"""
from __future__ import annotations

from typing import Sequence, Tuple

from .game import CHANCE

SAFE, RISKY = 0, 1
HIT, MISS = 0, 1


class RaceState:
    __slots__ = ("scores", "mover", "pending", "target", "p_hit", "plies")

    def __init__(self, scores=(0, 0), mover=0, pending=False, target=10, p_hit=0.5, plies=0):
        self.scores = tuple(scores)
        self.mover = mover
        self.pending = pending  # True => chance node resolving the mover's RISKY action
        self.target = target
        self.p_hit = p_hit
        self.plies = plies

    def to_move(self) -> int:
        return CHANCE if self.pending else self.mover

    def is_chance(self) -> bool:
        return self.pending

    def is_terminal(self) -> bool:
        return (not self.pending) and (max(self.scores) >= self.target)

    def legal_actions(self) -> Sequence[int]:
        return (SAFE, RISKY)

    def apply_action(self, action: int) -> "RaceState":
        assert not self.pending and not self.is_terminal()
        if action == SAFE:
            s = list(self.scores)
            s[self.mover] += 1
            return RaceState(s, 1 - self.mover, False, self.target, self.p_hit, self.plies + 1)
        return RaceState(self.scores, self.mover, True, self.target, self.p_hit, self.plies + 1)

    def chance_outcomes(self) -> Sequence[Tuple[int, float]]:
        return ((HIT, self.p_hit), (MISS, 1.0 - self.p_hit))

    def apply_chance(self, outcome: int) -> "RaceState":
        assert self.pending
        s = list(self.scores)
        if outcome == HIT:
            s[self.mover] += 3
        return RaceState(s, 1 - self.mover, False, self.target, self.p_hit, self.plies)

    def returns(self) -> Tuple[float, float]:
        if not self.is_terminal():
            return (0.0, 0.0)
        return (1.0, -1.0) if self.scores[0] >= self.target else (-1.0, 1.0)

    def score_diff(self) -> float:
        return float(self.scores[0] - self.scores[1])

    def key(self) -> bytes:
        return bytes((self.scores[0], self.scores[1], self.mover, int(self.pending)))

    @staticmethod
    def num_actions() -> int:
        return 2
