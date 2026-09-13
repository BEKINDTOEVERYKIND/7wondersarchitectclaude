"""Play matches between agents through an :class:`Environment` factory."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..agents.base import Agent
from ..game import Environment
from .elo import wilson_interval

EnvFactory = Callable[[int], Environment]  # seed -> fresh environment


@dataclass
class MatchResult:
    wins: List[float] = field(default_factory=lambda: [0.0, 0.0])
    draws: int = 0
    games: int = 0
    score_diffs: List[int] = field(default_factory=list)
    plies: List[int] = field(default_factory=list)
    per_game: List[Tuple[int, int, int, int]] = field(default_factory=list)  # (seed, first_player_agent, s0, s1)
    outcomes: List[float] = field(default_factory=list)  # 1 / 0.5 / 0 for agent 0, using the game's returns (tie-break included)

    def score(self, agent_idx: int) -> float:
        return (self.wins[agent_idx] + 0.5 * self.draws) / max(1, self.games)

    def summary(self, names: Tuple[str, str]) -> str:
        lo, hi = wilson_interval(self.wins[0] + 0.5 * self.draws, self.games)
        return (f"{names[0]} vs {names[1]}: {self.wins[0]:.0f}-{self.wins[1]:.0f}-{self.draws} "
                f"({self.score(0) * 100:.1f}% [{lo * 100:.0f}-{hi * 100:.0f}]), mean margin "
                f"{np.mean(self.score_diffs) if self.score_diffs else 0:+.2f}, mean plies {np.mean(self.plies) if self.plies else 0:.1f}")


def play_game(env: Environment, agents: Tuple[Agent, Agent], max_plies: int = 10_000,
              on_move: Optional[Callable] = None) -> Tuple[Tuple[float, float], Tuple[int, int], int]:
    """Play one game; ``agents[i]`` controls player ``i``.  Returns (returns, scores, plies)."""
    for a in agents:
        a.new_game()
    plies = 0
    while not env.is_terminal() and plies < max_plies:
        p = env.to_move()
        obs = env.observe(p)
        action = agents[p].select_action(obs)
        if on_move is not None:
            on_move(p, obs, action)
        env.step(action)
        plies += 1
    return env.returns(), env.scores(), plies


def play_match(env_factory: EnvFactory, agents: Tuple[Agent, Agent], num_games: int, seed: int = 0,
               alternate: bool = True, log: Optional[Callable[[str], None]] = None) -> MatchResult:
    """Play ``num_games`` games, alternating who is player 0 (fair colour balance)."""
    res = MatchResult()
    for g in range(num_games):
        first = (g % 2) if alternate else 0  # index of the agent that plays as player 0
        pair = (agents[first], agents[1 - first])
        env = env_factory(seed + g)
        r, s, plies = play_game(env, pair)
        res.games += 1
        # map back to agent indices
        r_agent = (r[0], r[1]) if first == 0 else (r[1], r[0])
        s_agent = (s[0], s[1]) if first == 0 else (s[1], s[0])
        if r_agent[0] > r_agent[1]:
            res.wins[0] += 1
        elif r_agent[1] > r_agent[0]:
            res.wins[1] += 1
        else:
            res.draws += 1
        res.score_diffs.append(int(s_agent[0] - s_agent[1]))
        res.plies.append(plies)
        res.per_game.append((seed + g, first, int(s_agent[0]), int(s_agent[1])))
        res.outcomes.append(1.0 if r_agent[0] > r_agent[1] else (0.0 if r_agent[1] > r_agent[0] else 0.5))
        if log is not None and ((g + 1) % 10 == 0 or g + 1 == num_games):
            log(f"  game {g + 1}/{num_games}: {res.wins[0]:.0f}-{res.wins[1]:.0f}-{res.draws}")
    return res
