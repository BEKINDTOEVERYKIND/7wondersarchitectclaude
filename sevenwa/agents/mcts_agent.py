"""Search-based agent (works with any Evaluator: neural or rollout)."""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..game import GameState
from ..search.gumbel import gumbel_search
from ..search.mcts import MCTS, MCTSConfig, Evaluator, SearchResult


class MCTSAgent:
    """``root_mode``: ``"puct"`` (visit-count policy) or ``"gumbel"`` (Sequential Halving with
    Gumbel-Top-k).  ``final_pick``: ``"visits"`` (today's argmax of visits / improved policy) or
    ``"q"`` (:meth:`SearchResult.best_action_q`: highest Q among the well-visited children)."""

    def __init__(self, evaluator: Evaluator, config: Optional[MCTSConfig] = None, name: str = "mcts",
                 temperature: float = 0.0, temperature_moves: int = 0, rng: Optional[np.random.Generator] = None,
                 add_noise: bool = False, num_actions: Optional[int] = None, root_mode: str = "puct",
                 gumbel_max_considered: int = 16, final_pick: str = "visits", q_min_frac: float = 0.25):
        if root_mode not in ("puct", "gumbel"):
            raise ValueError(f"unknown root_mode {root_mode!r} (puct|gumbel)")
        if final_pick not in ("visits", "q"):
            raise ValueError(f"unknown final_pick {final_pick!r} (visits|q)")
        self.name = name
        self.root_mode = root_mode
        self.gumbel_max_considered = gumbel_max_considered
        self.final_pick = final_pick
        self.q_min_frac = q_min_frac
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
        if self.root_mode == "gumbel":
            res = gumbel_search(self.mcts, state, self.mcts.cfg.num_simulations, max_considered=self.gumbel_max_considered)
            self.last_result = res
            if self.moves < self.temperature_moves:
                a = int(res.extra["chosen"])
            elif self.final_pick == "q":
                a = int(res.best_action_q(self.q_min_frac))
            else:
                a = int(np.argmax(res.improved_policy))
        else:
            res = self.mcts.run(state, add_noise=self.add_noise)
            self.last_result = res
            temp = self.temperature if self.moves < self.temperature_moves else 0.0
            if temp > 0:
                a = res.sample_action(self.rng, temp)
            elif self.final_pick == "q":
                a = res.best_action_q(self.q_min_frac)
            else:
                a = res.best_action()
        self.moves += 1
        # The tree is advanced only along the agent's OWN action; the opponent's moves and the
        # chance outcomes are recovered by MCTS._reusable_root at the next call (state key match).
        self.mcts.advance(a)
        return a
