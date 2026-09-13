from __future__ import annotations

from typing import Optional

import numpy as np

from ..game import GameState


class RandomAgent:
    name = "random"

    def __init__(self, rng: Optional[np.random.Generator] = None, seed: Optional[int] = None):
        self.rng = rng or np.random.default_rng(seed)

    def new_game(self) -> None:
        pass

    def select_action(self, state: GameState) -> int:
        la = state.legal_actions()
        return int(la[self.rng.integers(len(la))])
