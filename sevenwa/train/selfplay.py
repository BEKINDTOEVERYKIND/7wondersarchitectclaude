"""Self-play game generation (single process function + multiprocessing driver).

Each decision of a game yields one training sample: the mover-centric features, the legal
mask, the search's visit distribution (policy target), and — filled in at the end of the game —
the result ``z`` and final score margin from the mover's perspective.
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
from ..nn.features import FEATURE_SIZE, encode, encode_state, legal_mask
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
    evaluator: str = "net"             # "net" | "rollout"
    rollout_playouts: int = 1
    rollout_policy: str = "heuristic"  # "heuristic" | "random"
    net_threads: int = 1
    max_plies: int = 2000
    root_mode: str = "puct"            # "puct" (visit-count targets) | "gumbel" (sequential halving, completed-Q targets)
    gumbel_max_considered: int = 16


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


def _make_evaluator(cfg: SelfPlayConfig, checkpoint: Optional[str], rng: np.random.Generator):
    if cfg.evaluator == "net":
        import torch
        from ..nn.evaluator import TorchEvaluator
        from ..nn.model import PolicyValueNet
        torch.set_num_threads(cfg.net_threads)
        net = PolicyValueNet.load(checkpoint)
        return TorchEvaluator(net, encode, num_threads=cfg.net_threads)
    if cfg.rollout_policy == "heuristic":
        from ..agents.heuristic import heuristic_action, heuristic_prior
        return RolloutEvaluator(Actions.NUM, playouts_per_leaf=cfg.rollout_playouts, policy_fn=heuristic_action,
                                rng=rng, prior_fn=heuristic_prior)
    return RolloutEvaluator(Actions.NUM, playouts_per_leaf=cfg.rollout_playouts, rng=rng)


def play_selfplay_game(mcts: MCTS, cfg: SelfPlayConfig, seed: int, rng: np.random.Generator,
                       rules: Optional[RulesConfig] = None, mcts_other: Optional[MCTS] = None) -> GameRecord:
    """Play one self-play game.  Each seat searches with its own tree (``mcts`` for player 0 and
    ``mcts_other`` for player 1) because the two belief states differ (Cat peek), so a subtree built
    under one observer must not be reused by the other."""
    t0 = time.time()
    env = Environment(seed=seed, rules=rules)
    feats, masks, pis, movers = [], [], [], []
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
        env.step(a)
        for t in trees:
            t.advance(a)
        plies += 1
    r = env.returns()
    s = env.scores()
    z = np.array([r[m] for m in movers], dtype=np.float32)
    margin = np.array([(s[m] - s[1 - m]) for m in movers], dtype=np.float32)
    return GameRecord(np.stack(feats) if feats else np.zeros((0, FEATURE_SIZE), np.float32),
                      np.stack(masks) if masks else np.zeros((0, Actions.NUM), bool),
                      np.stack(pis) if pis else np.zeros((0, Actions.NUM), np.float32),
                      z, margin, s, env.wonders, plies, time.time() - t0)


def selfplay_worker(args) -> Dict:
    """Entry point for a worker process.  ``args`` is a dict (picklable)."""
    cfg = SelfPlayConfig(**args["config"])
    checkpoint = args.get("checkpoint")
    seeds: List[int] = args["seeds"]
    out_path: Optional[str] = args.get("out_path")
    generation = int(args.get("generation", 0))
    rng = np.random.default_rng(args.get("rng_seed", seeds[0] if seeds else 0))
    evaluator = _make_evaluator(cfg, checkpoint, rng)
    mcfg = MCTSConfig(num_simulations=cfg.num_simulations, batch_size=cfg.batch_size, c_puct=cfg.c_puct,
                      fpu_reduction=cfg.fpu_reduction, dirichlet_alpha=cfg.dirichlet_alpha,
                      dirichlet_epsilon=cfg.dirichlet_epsilon, add_root_noise=True)
    if cfg.evaluator != "net":
        mcfg.batch_size = 1
    mcts = MCTS(evaluator, mcfg, rng=rng, num_actions=Actions.NUM)
    mcts_other = MCTS(evaluator, mcfg, rng=rng, num_actions=Actions.NUM)
    records = [play_selfplay_game(mcts, cfg, seed, rng, mcts_other=mcts_other) for seed in seeds]
    feats = np.concatenate([r.feats for r in records]) if records else np.zeros((0, FEATURE_SIZE), np.float32)
    mask = np.concatenate([r.mask for r in records])
    pi = np.concatenate([r.pi for r in records])
    z = np.concatenate([r.z for r in records])
    margin = np.concatenate([r.margin for r in records])
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, feats=feats, mask=mask, pi=pi, z=z, margin=margin,
                            gen=np.full(len(z), generation, np.int32))
    return {
        "games": len(records), "samples": int(len(z)), "out_path": out_path,
        "mean_plies": float(np.mean([r.plies for r in records])) if records else 0.0,
        "seconds": float(sum(r.seconds for r in records)),
        "p0_wins": int(sum(1 for r in records if r.z[0] > 0 and r.z.size)),
        "draws": int(sum(1 for r in records if r.z.size and r.z[0] == 0)),
        "mean_score": float(np.mean([np.mean(r.scores) for r in records])) if records else 0.0,
    }


def run_selfplay(cfg: SelfPlayConfig, num_games: int, checkpoint: Optional[str], out_dir: str, generation: int,
                 num_workers: int = 4, seed: int = 0, log=print) -> List[Dict]:
    """Generate ``num_games`` games across ``num_workers`` processes; returns per-worker stats."""
    import multiprocessing as mp
    seeds = [seed + i for i in range(num_games)]
    chunks = [seeds[i::num_workers] for i in range(num_workers)]
    jobs = []
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        jobs.append({"config": asdict(cfg), "checkpoint": checkpoint, "seeds": chunk, "rng_seed": seed * 1000 + i,
                     "out_path": os.path.join(out_dir, f"gen{generation:04d}_w{i}.npz"), "generation": generation})
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
    self-play can produce early on."""
    from ..agents.heuristic import HeuristicParams, heuristic_action, heuristic_prior
    from dataclasses import replace as _replace
    seeds: List[int] = args["seeds"]
    eps = float(args.get("epsilon", 0.1))
    temperature = args.get("temperature")
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
            f_g.append(encode_state(obs))
            m_g.append(legal_mask(obs))
            p_g.append(heuristic_prior(obs, temperature=temperature) if temperature else heuristic_prior(obs))
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
            "mean_plies": plies_total / max(1, len(seeds))}


def run_imitation(num_games: int, out_dir: str, generation: int = 0, num_workers: int = 4, seed: int = 0,
                  epsilon: float = 0.1, log=print) -> List[Dict]:
    import multiprocessing as mp
    seeds = [seed + i for i in range(num_games)]
    chunks = [seeds[i::num_workers] for i in range(num_workers)]
    jobs = [{"seeds": c, "epsilon": epsilon, "rng_seed": seed * 1000 + i, "generation": generation,
             "out_path": os.path.join(out_dir, f"gen{generation:04d}_w{i}.npz")} for i, c in enumerate(chunks) if c]
    t0 = time.time()
    if num_workers <= 1 or len(jobs) == 1:
        results = [imitation_worker(j) for j in jobs]
    else:
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            results = pool.map(imitation_worker, jobs)
    games = sum(r["games"] for r in results)
    log(f"imitation data: {games} heuristic games, {sum(r['samples'] for r in results)} samples, {time.time() - t0:.1f}s")
    return results
