"""Search-based agent (works with any Evaluator: neural or rollout)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..game import GameState
from ..search.mcts import MCTS, MCTSConfig, Evaluator, SearchResult


class MCTSAgent:
    def __init__(self, evaluator: Evaluator, config: Optional[MCTSConfig] = None, name: str = "mcts",
                 temperature: float = 0.0, temperature_moves: int = 0, rng: Optional[np.random.Generator] = None,
                 add_noise: bool = False, num_actions: Optional[int] = None):
        self.name = name
        self.rng = rng or np.random.default_rng()
        self.mcts = MCTS(evaluator, config, rng=self.rng, num_actions=num_actions)
        self.temperature = temperature
        self.temperature_moves = temperature_moves
        self.add_noise = add_noise
        self.moves = 0
        self.last_result: Optional[SearchResult] = None

    def new_game(self) -> None:
        self.mcts.reset()
        self.moves = 0
        self.last_result = None

    def select_action(self, state: GameState) -> int:
        res = self.mcts.run(state, add_noise=self.add_noise)
        self.last_result = res
        temp = self.temperature if self.moves < self.temperature_moves else 0.0
        a = res.sample_action(self.rng, temp) if temp > 0 else res.best_action()
        self.moves += 1
        self.mcts.advance(a)
        return a
