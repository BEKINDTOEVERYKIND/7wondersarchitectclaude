"""Build agents from short spec strings (used by the CLI and the pipeline).

Specs
-----
* ``random``
* ``heuristic`` or ``heuristic:<epsilon>``
* ``rollout:<sims>[:<playouts>[:<policy>]]``   – MCTS with playouts; policy ``heuristic`` (default) or ``random``
* ``net:<checkpoint.pt>[:<sims>]``            – neural MCTS (default 200 simulations)
* ``netraw:<checkpoint.pt>``                  – the network's policy alone (no search)
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from ..engine.actions import Actions
from ..search.mcts import MCTSConfig
from ..search.rollout import RolloutEvaluator, random_policy
from .mcts_agent import MCTSAgent
from .random_agent import RandomAgent


class RawNetAgent:
    """Greedy (or sampled) network policy without search."""

    def __init__(self, evaluator, name: str = "netraw", temperature: float = 0.0, rng=None):
        self.evaluator = evaluator
        self.name = name
        self.temperature = temperature
        self.rng = rng or np.random.default_rng()

    def new_game(self) -> None:
        pass

    def select_action(self, state) -> int:
        pols, _ = self.evaluator.evaluate([state])
        legal = list(state.legal_actions())
        p = np.array([pols[0][a] for a in legal], dtype=np.float64)
        if self.temperature <= 0:
            return legal[int(np.argmax(p))]
        p = p ** (1.0 / self.temperature)
        p /= p.sum()
        return legal[int(self.rng.choice(len(legal), p=p))]


def make_agent(spec: str, seed: Optional[int] = None, mcts_config: Optional[MCTSConfig] = None,
               net_threads: int = 1):
    rng = np.random.default_rng(seed)
    parts = spec.split(":")
    kind = parts[0]
    if kind == "random":
        return RandomAgent(rng=rng)
    if kind == "heuristic":
        from .heuristic import HeuristicAgent
        eps = float(parts[1]) if len(parts) > 1 else 0.0
        return HeuristicAgent(rng=rng, epsilon=eps)
    if kind == "rollout":
        sims = int(parts[1]) if len(parts) > 1 else None
        playouts = int(parts[2]) if len(parts) > 2 else 1
        pol_name = parts[3] if len(parts) > 3 else "heuristic"
        if pol_name == "heuristic":
            from .heuristic import heuristic_action, heuristic_prior
            policy_fn = heuristic_action
            prior_fn = heuristic_prior
        else:
            policy_fn = random_policy
            prior_fn = None
        ev = RolloutEvaluator(Actions.NUM, playouts_per_leaf=playouts, policy_fn=policy_fn, rng=rng, prior_fn=prior_fn)
        cfg = replace(mcts_config) if mcts_config is not None else MCTSConfig(num_simulations=200)
        if sims is not None:  # an explicit count in the spec wins over the caller's config
            cfg.num_simulations = sims
        cfg.batch_size = 1  # playout values are computed one leaf at a time
        return MCTSAgent(ev, cfg, name=f"rollout{cfg.num_simulations}x{playouts}-{pol_name}", rng=rng, num_actions=Actions.NUM)
    if kind in ("net", "netraw"):
        import torch
        from ..nn.evaluator import TorchEvaluator
        from ..nn.features import encode
        from ..nn.model import PolicyValueNet
        path = parts[1]
        net = PolicyValueNet.load(path)
        ev = TorchEvaluator(net, encode, num_threads=net_threads)
        if kind == "netraw":
            return RawNetAgent(ev, name=f"netraw({path})", rng=rng)
        cfg = replace(mcts_config) if mcts_config is not None else MCTSConfig(num_simulations=200)
        if len(parts) > 2:
            cfg.num_simulations = int(parts[2])
        return MCTSAgent(ev, cfg, name=f"net{cfg.num_simulations}({path})", rng=rng, num_actions=Actions.NUM)
    raise ValueError(f"unknown agent spec: {spec}")
