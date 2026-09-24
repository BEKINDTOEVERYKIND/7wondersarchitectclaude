"""Self-play game generation (single process function + multiprocessing driver).

Each decision of a game yields one training sample: the mover-centric features, the legal
mask, the search's visit distribution (policy target), and — filled in at the end of the game —
the result ``z`` and final score margin from the mover's perspective.

Feature versions (``sevenwa.nn.features``): a network is always served the encoding of its
``cfg.input_dim``, and the samples a self-play worker writes use the version of the checkpoint
that plays (so expert iteration fine-tunes it on matching features); without a checkpoint
(rollout evaluator) and in the imitation worker the version is an argument (default: latest).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..engine.actions import Actions
from ..engine.env import Environment
from ..engine.rules import RulesConfig
from ..game import CHANCE
from ..nn.features import (EXPERT_PRIOR_SLICE, FEATURE_VERSION_LATEST, encoder_for_dim, encoders, feature_size,
                           legal_mask)
from ..search.gumbel import gumbel_search
from ..search.mcts import MCTS, MCTSConfig
from ..search.rollout import RolloutEvaluator


@dataclass
class SelfPlayConfig:
    num_simulations: int = 160
    batch_size: int = 8
    c_puct: float = 1.75
    fpu_reduction: float = 0.25
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0
    temperature_moves: int = 20       # number of decisions (per game) played with temperature
    final_temperature: float = 0.0
    evaluator: str = "net"             # "net" | "rollout" | "hybrid" (net priors, values blended with playouts)
    rollout_playouts: int = 1
    rollout_policy: str = "heuristic"  # "heuristic" | "random"
    hybrid_lambda: float = 0.5         # hybrid: value = (1-lam)*v_net + lam*mean(playouts)
    net_threads: int = 1
    max_plies: int = 2000
    root_mode: str = "puct"            # "puct" (visit-count targets) | "gumbel" (sequential halving, completed-Q targets)
    gumbel_max_considered: int = 16
    value_mix: float = 0.0             # value target = (1-mix)*game outcome + mix*search root value (lower variance)


@dataclass
class GameRecord:
    feats: np.ndarray
    mask: np.ndarray
    pi: np.ndarray
    z: np.ndarray
    margin: np.ndarray
    scores: Tuple[int, int]
    wonders: Tuple[int, int]
    plies: int
    seconds: float


def _make_evaluator_versioned(cfg: SelfPlayConfig, checkpoint: Optional[str],
                              rng: np.random.Generator) -> Tuple[object, Optional[int]]:
    """``(evaluator, feature version)``: the version is the checkpoint's (from its ``cfg.input_dim``)
    for the net / hybrid evaluators and ``None`` for the rollout evaluator (no network)."""
    if cfg.evaluator in ("net", "hybrid"):
        import torch
        from ..nn.evaluator import TorchEvaluator
        from ..nn.model import PolicyValueNet
        torch.set_num_threads(cfg.net_threads)
        net = PolicyValueNet.load(checkpoint)
        version, encode, _ = encoder_for_dim(net.cfg.input_dim)
        ev = TorchEvaluator(net, encode, num_threads=cfg.net_threads)
        if cfg.evaluator == "net":
            return ev, version
        from ..agents.heuristic import heuristic_action
        from ..search.rollout import HybridEvaluator
        return HybridEvaluator(ev, Actions.NUM, lam=cfg.hybrid_lambda, playouts_per_leaf=cfg.rollout_playouts,
                               policy_fn=heuristic_action, rng=rng), version
    if cfg.rollout_policy == "heuristic":
        from ..agents.heuristic import heuristic_action, heuristic_prior
        return RolloutEvaluator(Actions.NUM, playouts_per_leaf=cfg.rollout_playouts, policy_fn=heuristic_action,
                                rng=rng, prior_fn=heuristic_prior), None
    return RolloutEvaluator(Actions.NUM, playouts_per_leaf=cfg.rollout_playouts, rng=rng), None


def _make_evaluator(cfg: SelfPlayConfig, checkpoint: Optional[str], rng: np.random.Generator):
    return _make_evaluator_versioned(cfg, checkpoint, rng)[0]


def play_selfplay_game(mcts: MCTS, cfg: SelfPlayConfig, seed: int, rng: np.random.Generator,
                       rules: Optional[RulesConfig] = None, mcts_other: Optional[MCTS] = None,
                       feature_version: int = FEATURE_VERSION_LATEST) -> GameRecord:
    """Play one self-play game.  Each seat searches with its own tree (``mcts`` for player 0 and
    ``mcts_other`` for player 1) because the two belief states differ (Cat peek), so a subtree built
    under one observer must not be reused by the other.  The recorded features use encoding
    ``feature_version`` (pass the version of the network that searches)."""
    t0 = time.time()
    _, encode_state = encoders(feature_version)
    env = Environment(seed=seed, rules=rules)
    feats, masks, pis, movers, roots = [], [], [], [], []
    trees = [mcts, mcts_other if mcts_other is not None else MCTS(mcts.evaluator, mcts.cfg, rng=rng, num_actions=mcts.num_actions)]
    for t in trees:
        t.reset()
    plies = 0
    while not env.is_terminal() and plies < cfg.max_plies:
        p = env.to_move()
        obs = env.observe(p)
        mcts = trees[p]
        if cfg.root_mode == "gumbel":
            res = gumbel_search(mcts, obs, cfg.num_simulations, max_considered=cfg.gumbel_max_considered)
            a = int(res.extra["chosen"]) if plies < cfg.temperature_moves else int(np.argmax(res.improved_policy))
        else:
            res = mcts.run(obs, add_noise=True)
            temp = cfg.temperature if plies < cfg.temperature_moves else cfg.final_temperature
            a = res.sample_action(rng, temp) if temp > 0 else res.best_action()
        feats.append(encode_state(obs))
        masks.append(legal_mask(obs))
        pis.append(res.policy_target)
        movers.append(p)
        roots.append(float(res.root_value))
        env.step(a)
        for t in trees:
            t.advance(a)
        plies += 1
    r = env.returns()
    s = env.scores()
    z = np.array([r[m] for m in movers], dtype=np.float32)
    if cfg.value_mix > 0:
        z = (1.0 - cfg.value_mix) * z + cfg.value_mix * np.clip(np.array(roots, dtype=np.float32), -1.0, 1.0)
    margin = np.array([(s[m] - s[1 - m]) for m in movers], dtype=np.float32)
    return GameRecord(np.stack(feats) if feats else np.zeros((0, feature_size(feature_version)), np.float32),
                      np.stack(masks) if masks else np.zeros((0, Actions.NUM), bool),
                      np.stack(pis) if pis else np.zeros((0, Actions.NUM), np.float32),
                      z, margin, s, env.wonders, plies, time.time() - t0)


def selfplay_worker(args) -> Dict:
    """Entry point for a worker process.  ``args`` is a dict (picklable).

    The samples are encoded with the feature version of the checkpoint that plays; without a network
    (rollout evaluator) with ``args["feature_version"]`` (default: the latest version).  An explicit
    ``feature_version`` that contradicts the checkpoint is an error."""
    cfg = SelfPlayConfig(**args["config"])
    checkpoint = args.get("checkpoint")
    seeds: List[int] = args["seeds"]
    out_path: Optional[str] = args.get("out_path")
    generation = int(args.get("generation", 0))
    rng = np.random.default_rng(args.get("rng_seed", seeds[0] if seeds else 0))
    evaluator, net_version = _make_evaluator_versioned(cfg, checkpoint, rng)
    requested = args.get("feature_version")
    if net_version is not None and requested is not None and int(requested) != net_version:
        raise ValueError(f"feature_version {requested} requested, but {checkpoint} uses feature version {net_version}")
    version = net_version if net_version is not None else int(requested or FEATURE_VERSION_LATEST)
    feature_size(version)  # validates
    mcfg = MCTSConfig(num_simulations=cfg.num_simulations, batch_size=cfg.batch_size, c_puct=cfg.c_puct,
                      fpu_reduction=cfg.fpu_reduction, dirichlet_alpha=cfg.dirichlet_alpha,
                      dirichlet_epsilon=cfg.dirichlet_epsilon, add_root_noise=True)
    if cfg.evaluator != "net":
        mcfg.batch_size = 1
    mcts = MCTS(evaluator, mcfg, rng=rng, num_actions=Actions.NUM)
    mcts_other = MCTS(evaluator, mcfg, rng=rng, num_actions=Actions.NUM)
    records = [play_selfplay_game(mcts, cfg, seed, rng, mcts_other=mcts_other, feature_version=version) for seed in seeds]
    feats = np.concatenate([r.feats for r in records]) if records else np.zeros((0, feature_size(version)), np.float32)
    mask = np.concatenate([r.mask for r in records])
    pi = np.concatenate([r.pi for r in records])
    z = np.concatenate([r.z for r in records])
    margin = np.concatenate([r.margin for r in records])
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, feats=feats, mask=mask, pi=pi, z=z, margin=margin,
                            gen=np.full(len(z), generation, np.int32))
    return {
        "games": len(records), "samples": int(len(z)), "out_path": out_path, "feature_version": version,
        "mean_plies": float(np.mean([r.plies for r in records])) if records else 0.0,
        "seconds": float(sum(r.seconds for r in records)),
        "p0_wins": int(sum(1 for r in records if r.z[0] > 0 and r.z.size)),
        "draws": int(sum(1 for r in records if r.z.size and r.z[0] == 0)),
        "mean_score": float(np.mean([np.mean(r.scores) for r in records])) if records else 0.0,
    }


def run_selfplay(cfg: SelfPlayConfig, num_games: int, checkpoint: Optional[str], out_dir: str, generation: int,
                 num_workers: int = 4, seed: int = 0, log=print, feature_version: Optional[int] = None) -> List[Dict]:
    """Generate ``num_games`` games across ``num_workers`` processes; returns per-worker stats.

    The shards use the checkpoint's feature version; ``feature_version`` only matters without a network
    (rollout evaluator; default: the latest version)."""
    import multiprocessing as mp
    seeds = [seed + i for i in range(num_games)]
    chunks = [seeds[i::num_workers] for i in range(num_workers)]
    jobs = []
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        jobs.append({"config": asdict(cfg), "checkpoint": checkpoint, "seeds": chunk, "rng_seed": seed * 1000 + i,
                     "out_path": os.path.join(out_dir, f"gen{generation:04d}_w{i}.npz"), "generation": generation,
                     "feature_version": feature_version})
    t0 = time.time()
    if num_workers <= 1 or len(jobs) == 1:
        results = [selfplay_worker(j) for j in jobs]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(len(jobs)) as pool:
            results = pool.map(selfplay_worker, jobs)
    games = sum(r["games"] for r in results)
    samples = sum(r["samples"] for r in results)
    dt = time.time() - t0
    log(f"self-play gen {generation}: {games} games, {samples} samples, {dt:.1f}s ({games / max(dt, 1e-9) * 60:.1f} games/min), "
        f"mean plies {np.mean([r['mean_plies'] for r in results]):.1f}, mean score {np.mean([r['mean_score'] for r in results]):.1f}")
    return results


# ---------------------------------------------------------------------------
# Imitation bootstrap: fast heuristic self-play with soft heuristic policy targets
# ---------------------------------------------------------------------------
def imitation_worker(args) -> Dict:
    """Generate heuristic-vs-heuristic games (ε-greedy for diversity) and record, per decision,
    the heuristic's soft policy (``heuristic_prior``) as the target plus the game outcome and
    margin.  ~100 games/s per process, so tens of thousands of games are cheap; the resulting
    network is a heuristic-level policy with a value head trained on far more outcomes than
    self-play can produce early on.

    ``args["feature_version"]`` (default: the latest) selects the feature encoding of the shards.  From
    version 2 on the features contain the heuristic prior itself, so with the default temperature the
    policy target is taken from them instead of being computed a second time (identical values)."""
    from ..agents.heuristic import HeuristicParams, heuristic_action, heuristic_prior
    from dataclasses import replace as _replace
    seeds: List[int] = args["seeds"]
    eps = float(args.get("epsilon", 0.1))
    temperature = args.get("temperature")
    version = int(args.get("feature_version") or FEATURE_VERSION_LATEST)
    _, encode_state = encoders(version)
    reuse_prior = version >= 2 and not temperature
    out_path = args.get("out_path")
    generation = int(args.get("generation", 0))
    rng = np.random.default_rng(args.get("rng_seed", 0))
    params = _replace(HeuristicParams(), epsilon=eps)
    feats, masks, pis, zs, margins = [], [], [], [], []
    t0 = time.time()
    plies_total = 0
    for seed in seeds:
        env = Environment(seed=seed)
        f_g, m_g, p_g, movers = [], [], [], []
        while not env.is_terminal():
            p = env.to_move()
            obs = env.observe(p)
            f = encode_state(obs)
            f_g.append(f)
            m_g.append(legal_mask(obs))
            pr = None
            if reuse_prior and len(obs.legal_actions()) >= 2:
                pr = f[EXPERT_PRIOR_SLICE[0]:EXPERT_PRIOR_SLICE[1]].copy()
                if not pr.sum() > 0.5:  # the EXPERT block fell back to zeros: compute the target directly
                    pr = None
            if pr is None:
                pr = heuristic_prior(obs, temperature=temperature) if temperature else heuristic_prior(obs)
            p_g.append(pr)
            movers.append(p)
            env.step(heuristic_action(obs, rng, params))
        r = env.returns()
        s = env.scores()
        feats.extend(f_g); masks.extend(m_g); pis.extend(p_g)
        zs.extend(r[m] for m in movers)
        margins.extend(s[m] - s[1 - m] for m in movers)
        plies_total += len(movers)
    feats = np.stack(feats).astype(np.float32); masks = np.stack(masks); pis = np.stack(pis).astype(np.float32)
    z = np.array(zs, np.float32); margin = np.array(margins, np.float32)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, feats=feats, mask=masks, pi=pis, z=z, margin=margin,
                            gen=np.full(len(z), generation, np.int32))
    return {"games": len(seeds), "samples": int(len(z)), "out_path": out_path, "seconds": time.time() - t0,
            "mean_plies": plies_total / max(1, len(seeds)), "feature_version": version}


def run_imitation(num_games: int, out_dir: str, generation: int = 0, num_workers: int = 4, seed: int = 0,
                  epsilon: float = 0.1, log=print, games_per_job: int = 500,
                  feature_version: int = FEATURE_VERSION_LATEST) -> List[Dict]:
    """Generate imitation data in small jobs (``games_per_job`` games each, one shard per job) so that a
    worker never holds more than ~30k samples in memory and a lost job costs little.  The shards use
    feature encoding ``feature_version`` (default: the latest)."""
    feature_size(feature_version)  # validates before any worker starts
    import multiprocessing as mp
    seeds = [seed + i for i in range(num_games)]
    chunks = [seeds[i:i + games_per_job] for i in range(0, num_games, games_per_job)]
    jobs = [{"seeds": c, "epsilon": epsilon, "rng_seed": seed * 1000 + i, "generation": generation,
             "feature_version": int(feature_version),
             "out_path": os.path.join(out_dir, f"gen{generation:04d}_j{i:04d}.npz")} for i, c in enumerate(chunks) if c]
    t0 = time.time()
    if num_workers <= 1 or len(jobs) == 1:
        results = [imitation_worker(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(min(num_workers, len(jobs))) as pool:
            results = pool.map(imitation_worker, jobs, chunksize=1)
    games = sum(r["games"] for r in results)
    log(f"imitation data: {games} heuristic games, {sum(r['samples'] for r in results)} samples "
        f"(feature version {feature_version}), {time.time() - t0:.1f}s")
    return results
