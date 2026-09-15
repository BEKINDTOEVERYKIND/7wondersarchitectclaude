"""Build agents from short spec strings (used by the CLI and the pipeline).

Specs
-----
* ``random``
* ``heuristic`` or ``heuristic:<epsilon>``
* ``rollout:<sims>[:<playouts>[:<policy>]]``   – MCTS with playouts; policy ``heuristic`` (default) or ``random``
* ``net:<checkpoint.pt>[:<sims>[:<c_puct>[:<value_T>]]]`` – neural MCTS (default 200 simulations; optional
  PUCT constant and value-calibration temperature, e.g. ``net:m.pt:300:3.0:1.45``)
* ``netraw:<checkpoint.pt>``                  – the network's policy alone (no search)
* ``hybrid:<checkpoint.pt>[:<sims>[:<lam>[:<playouts>]]]`` – neural MCTS whose leaf values are blended
  with heuristic playouts: value = (1-lam)·net + lam·playout (default sims 100, lam 0.5, 1 playout)

Options
-------
Any token after the kind (and, for ``net`` / ``netraw`` / ``hybrid``, after the checkpoint path) that
contains ``=`` is a ``key=value`` option; the other tokens keep their positional meaning, e.g.
``hybrid:models/imitation_v7.pt:300:0.5:pt=1.5:floor=0.05:root=gumbel:pick=q:batch=8:margin=0.5``.
Every option defaults to today's behaviour when absent:

* ``pt=<T>``      policy temperature of the network priors, softmax(logits / T)  (net, netraw, hybrid)
* ``floor=<f>``   root prior floor: p <- (1-f)·p + f·uniform over the legal moves  (search agents)
* ``root=puct|gumbel``  root procedure: PUCT visit counts or Sequential Halving with Gumbel-Top-k
* ``pick=visits|q``     final move: most visits (default) or best Q among the well-visited children
* ``batch=<n>``   leaves per evaluator call (``rollout`` / ``hybrid`` default to 1, ``net`` to 8)
* ``margin=<w>``  hybrid playout value = (1-w)·result + w·tanh(margin/8)  (hybrid only)
* ``cpuct=<c>``   PUCT exploration constant;  ``fpu=<r>``  first-play-urgency reduction
* ``reuse=chance|all``  tree reuse only through resolved chance nodes (default) or also across
  the opponent's decisions
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..engine.actions import Actions
from ..search.mcts import MCTSConfig
from ..search.rollout import RolloutEvaluator, random_policy
from .mcts_agent import MCTSAgent
from .random_agent import RandomAgent

SEARCH_KEYS = ("floor", "root", "pick", "batch", "cpuct", "fpu", "reuse")
ALLOWED_KEYS = {
    "random": (),
    "heuristic": (),
    "rollout": SEARCH_KEYS,
    "net": ("pt",) + SEARCH_KEYS,
    "netraw": ("pt",),
    "hybrid": ("pt", "margin") + SEARCH_KEYS,
}


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


def parse_spec(spec: str) -> Tuple[str, List[str], Dict[str, str]]:
    """Split ``spec`` into ``(kind, positional tokens, options)``.

    Tokens containing ``=`` are options (validated against the kind's allowed keys); for the
    checkpoint kinds the first token after the kind is always the path.
    """
    parts = spec.split(":")
    kind = parts[0]
    if kind not in ALLOWED_KEYS:
        raise ValueError(f"unknown agent spec: {spec}")
    tokens = parts[1:]
    positional: List[str] = []
    if kind in ("net", "netraw", "hybrid"):
        if not tokens:
            raise ValueError(f"{kind} spec needs a checkpoint path: {spec}")
        positional.append(tokens[0])
        tokens = tokens[1:]
    opts: Dict[str, str] = {}
    for t in tokens:
        if "=" in t:
            k, v = t.split("=", 1)
            if k not in ALLOWED_KEYS[kind]:
                raise ValueError(f"option {k!r} is not valid for {kind!r} specs "
                                 f"(allowed: {', '.join(ALLOWED_KEYS[kind]) or 'none'}): {spec}")
            if k in opts:
                raise ValueError(f"duplicate option {k!r}: {spec}")
            opts[k] = v
        else:
            positional.append(t)
    return kind, positional, opts


def _apply_search_options(cfg: MCTSConfig, opts: Dict[str, str], spec: str) -> Dict[str, object]:
    """Write the search options into ``cfg``; returns the MCTSAgent keyword arguments."""
    agent_kwargs: Dict[str, object] = {}
    if "floor" in opts:
        f = float(opts["floor"])
        if not 0.0 <= f <= 1.0:
            raise ValueError(f"floor must be in [0, 1]: {spec}")
        cfg.prior_floor = f
    if "batch" in opts:
        b = int(opts["batch"])
        if b < 1:
            raise ValueError(f"batch must be >= 1: {spec}")
        cfg.batch_size = b
    if "cpuct" in opts:
        cfg.c_puct = float(opts["cpuct"])
    if "fpu" in opts:
        cfg.fpu_reduction = float(opts["fpu"])
    if "reuse" in opts:
        if opts["reuse"] not in ("chance", "all"):
            raise ValueError(f"reuse must be 'chance' or 'all': {spec}")
        cfg.reuse_across_opponent = opts["reuse"] == "all"
    if "root" in opts:
        if opts["root"] not in ("puct", "gumbel"):
            raise ValueError(f"root must be 'puct' or 'gumbel': {spec}")
        agent_kwargs["root_mode"] = opts["root"]
    if "pick" in opts:
        if opts["pick"] not in ("visits", "q"):
            raise ValueError(f"pick must be 'visits' or 'q': {spec}")
        agent_kwargs["final_pick"] = opts["pick"]
    return agent_kwargs


def _unit(opts: Dict[str, str], key: str, default: float, spec: str) -> float:
    if key not in opts:
        return default
    v = float(opts[key])
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{key} must be in [0, 1]: {spec}")
    return v


def _tag(opts: Dict[str, str]) -> str:
    return "[" + ",".join(f"{k}={v}" for k, v in opts.items()) + "]" if opts else ""


def make_agent(spec: str, seed: Optional[int] = None, mcts_config: Optional[MCTSConfig] = None,
               net_threads: int = 1):
    rng = np.random.default_rng(seed)
    kind, parts, opts = parse_spec(spec)
    parts = [kind] + parts  # positional tokens with the kind at index 0 (as before)
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
        cfg.batch_size = 1  # playout values are computed one leaf at a time (``batch=`` overrides)
        kw = _apply_search_options(cfg, opts, spec)
        return MCTSAgent(ev, cfg, name=f"rollout{cfg.num_simulations}x{playouts}-{pol_name}{_tag(opts)}", rng=rng,
                         num_actions=Actions.NUM, **kw)
    if kind == "hybrid":
        from ..nn.evaluator import TorchEvaluator
        from ..nn.features import encode
        from ..nn.model import PolicyValueNet
        from ..search.rollout import HybridEvaluator
        path = parts[1]
        net = PolicyValueNet.load(path)
        lam = float(parts[3]) if len(parts) > 3 else 0.5
        playouts = int(parts[4]) if len(parts) > 4 else 1
        pt = float(opts.get("pt", 1.0))
        margin_w = _unit(opts, "margin", 0.0, spec)
        ev = HybridEvaluator(TorchEvaluator(net, encode, num_threads=net_threads, policy_temperature=pt), Actions.NUM,
                             lam=lam, playouts_per_leaf=playouts, rng=rng, margin_weight=margin_w)
        cfg = replace(mcts_config) if mcts_config is not None else MCTSConfig(num_simulations=100)
        if len(parts) > 2:
            cfg.num_simulations = int(parts[2])
        cfg.batch_size = 1  # ``batch=`` overrides
        kw = _apply_search_options(cfg, opts, spec)
        return MCTSAgent(ev, cfg, name=f"hybrid{cfg.num_simulations}(lam={lam},{path}){_tag(opts)}", rng=rng,
                         num_actions=Actions.NUM, **kw)
    if kind in ("net", "netraw"):
        from ..nn.evaluator import TorchEvaluator
        from ..nn.features import encode
        from ..nn.model import PolicyValueNet
        path = parts[1]
        net = PolicyValueNet.load(path)
        value_t = float(parts[4]) if len(parts) > 4 else 1.0
        pt = float(opts.get("pt", 1.0))
        ev = TorchEvaluator(net, encode, num_threads=net_threads, value_temperature=value_t, policy_temperature=pt)
        if kind == "netraw":
            return RawNetAgent(ev, name=f"netraw({path}){_tag(opts)}", rng=rng)
        cfg = replace(mcts_config) if mcts_config is not None else MCTSConfig(num_simulations=200)
        if len(parts) > 2:
            cfg.num_simulations = int(parts[2])
        if len(parts) > 3:
            cfg.c_puct = float(parts[3])
        kw = _apply_search_options(cfg, opts, spec)  # ``cpuct=`` wins over the positional constant
        tag = f"net{cfg.num_simulations}" + (f"c{cfg.c_puct:g}" if len(parts) > 3 else "") + (f"T{value_t:g}" if len(parts) > 4 else "")
        return MCTSAgent(ev, cfg, name=f"{tag}({path}){_tag(opts)}", rng=rng, num_actions=Actions.NUM, **kw)
    raise ValueError(f"unknown agent spec: {spec}")
