"""The real game: owns the hidden information and resolves chance nodes with it.

* Wonder decks and the central deck are shuffled once; ``reveal``/``draw`` outcomes come from
  the true order.  Halicarnassus "shuffle the rest back" re-shuffles the physical deck.
* The Progress-token stack is shuffled once.
* ``observe(p)`` returns the public belief state with the central top card hidden unless
  ``p`` is the player who peeked at it.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..game import CHANCE
from .cards import NUM_KINDS, central_deck_counts, wonder_deck_counts
from .rules import RulesConfig
from .state import (C_DRAW_CENTRAL, C_HALI_REVEAL, C_PEEK, C_REVEAL, C_TOKEN_BLIND, C_TOKEN_REVEAL, CENTRAL,
                    GameState)
from .tokens import TOKENS
from .wonders import NUM_WONDERS, WONDERS


def _expand(counts: List[int]) -> List[int]:
    out = []
    for k, c in enumerate(counts):
        out.extend([k] * c)
    return out


class Environment:
    def __init__(self, wonders: Optional[Tuple[int, int]] = None, rules: Optional[RulesConfig] = None,
                 seed: Optional[int] = None, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng(seed)
        self.rules = rules or RulesConfig()
        if wonders is None:
            w = self.rng.choice(NUM_WONDERS, size=2, replace=False)
            wonders = (int(w[0]), int(w[1]))
        self.wonders = wonders
        # true deck orders: index 0 is the top
        self.decks: List[List[int]] = [
            self._shuffled(_expand(wonder_deck_counts(WONDERS[wonders[0]].name))),
            self._shuffled(_expand(wonder_deck_counts(WONDERS[wonders[1]].name))),
            self._shuffled(_expand(central_deck_counts())),
        ]
        self.token_stack: List[int] = self._shuffled([t.id for t in TOKENS for _ in range(t.copies)])
        self.state = GameState.initial(wonders, self.rules)
        self._resolve_chance()

    def _shuffled(self, items: List[int]) -> List[int]:
        arr = np.array(items, dtype=np.int64)
        self.rng.shuffle(arr)
        return [int(x) for x in arr]

    # ---- chance resolution with the truth ---------------------------------
    def _sync_decks(self) -> None:
        """Physical decks shrink exactly when the belief's deck sizes shrink (top card taken)."""
        for d in range(3):
            while len(self.decks[d]) > self.state.deck_size[d]:
                self.decks[d].pop(0)

    def _resolve_chance(self) -> None:
        s = self.state
        while s.is_chance():
            self._sync_decks()
            kind, ctx = s.ckind, s.cctx
            if kind == C_REVEAL:
                outcome = self.decks[ctx][0]
            elif kind == C_DRAW_CENTRAL:
                outcome = self.decks[CENTRAL][0]
            elif kind == C_PEEK:
                outcome = self.decks[CENTRAL][0]
            elif kind == C_TOKEN_REVEAL or kind == C_TOKEN_BLIND:
                outcome = self.token_stack.pop(0)
            elif kind == C_HALI_REVEAL:
                d, remaining, revealed = ctx
                outcome = self.decks[d][len(revealed)]
            else:  # pragma: no cover
                raise RuntimeError("unknown chance kind")
            probs = dict(s.chance_outcomes())
            if outcome not in probs:
                raise RuntimeError(f"true outcome {outcome} impossible under belief for chance {kind} {ctx}")
            self.state = s = s.apply_chance(outcome)
        self._sync_decks()

    # ---- Environment protocol ------------------------------------------------
    def to_move(self) -> int:
        return self.state.to_move()

    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    def observe(self, player: int) -> GameState:
        s = self.state
        if s.central_known_to in (-1, player) or s.deck_top[CENTRAL] < 0:
            return s
        o = s._copy()
        o.unseen[CENTRAL][o.deck_top[CENTRAL]] += 1
        o.deck_top[CENTRAL] = -1
        o.central_known_to = -1
        o._legal = s._legal
        o._outcomes = s._outcomes
        return o

    def step(self, action: int) -> None:
        s = self.state
        if s.is_chance() or s.is_terminal():
            raise ValueError("environment is not at a decision node")
        from .state import D_HALI_CHOOSE, A as _A  # local import to avoid cycles
        hali_choice = (s.dkind == D_HALI_CHOOSE)
        hali_ctx = s.dctx
        self.state = s.apply_action(action)
        if hali_choice:
            # Physical model: the kept card is removed from the top-5 window and the rest of the
            # deck is shuffled.  The belief already returned the other revealed cards to 'unseen'.
            d, revealed = hali_ctx
            kept = action - _A.HALI_BASE
            n = len(revealed)
            window = self.decks[d][:n]
            window.remove(kept)
            rest = window + self.decks[d][n:]
            self.decks[d] = self._shuffled(rest)
        self._resolve_chance()

    def returns(self) -> Tuple[float, float]:
        return self.state.returns()

    def scores(self) -> Tuple[int, int]:
        return self.state.scores()


def make_env(seed: int, wonders: Optional[Tuple[int, int]] = None, rules: Optional[RulesConfig] = None) -> Environment:
    return Environment(wonders=wonders, rules=rules, seed=seed)
