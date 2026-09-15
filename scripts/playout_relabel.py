#!/usr/bin/env python3
"""Imitation positions with *playout-averaged* value targets.

The imitation bootstrap labels every position with the single outcome of one ε-greedy game — a
very noisy value target.  Here positions are sampled from ε-greedy heuristic games (as in the
bootstrap) and each one is labelled with the mean result and mean score margin of ``K``
heuristic playouts started from the player's belief state (hidden cards re-sampled per playout),
i.e. the same estimate the hybrid evaluator blends into its leaf values.  Policy targets stay the
heuristic's soft prior.  Shards use the self-play format, so they train through
``scripts/expert_iteration.py --skip-generation --extra-search-glob '<out-dir>/*.npz'``.
"""
import argparse
import multiprocessing as mp
import os
import time

import numpy as np

from sevenwa.game import CHANCE
from sevenwa.agents.heuristic import HeuristicParams, heuristic_action, heuristic_prior
from sevenwa.engine.env import Environment
from sevenwa.nn.features import encode_state, legal_mask


def _playout(state, rng):
    s = state
    while not s.is_terminal():
        if s.to_move() == CHANCE:
            outs = s.chance_outcomes()
            probs = np.fromiter((p for _, p in outs), dtype=np.float64, count=len(outs))
            probs /= probs.sum()
            s = s.apply_chance(outs[int(rng.choice(len(outs), p=probs))][0])
        else:
            s = s.apply_action(heuristic_action(s, rng))
    return s.returns(), s.scores()


def worker(args):
    from dataclasses import replace
    seeds, eps, K, every, rng_seed, out_path = args
    rng = np.random.default_rng(rng_seed)
    params = replace(HeuristicParams(), epsilon=eps)
    feats, masks, pis, zs, margins = [], [], [], [], []
    t0 = time.time()
    n_games = 0
    for seed in seeds:
        env = Environment(seed=seed)
        k = 0
        while not env.is_terminal():
            p = env.to_move()
            obs = env.observe(p)
            legal = obs.legal_actions()
            if len(legal) > 1:
                k += 1
                if k % every == 0:
                    acc_z = 0.0
                    acc_m = 0.0
                    for _ in range(K):
                        r, sc = _playout(obs, rng)
                        acc_z += r[p]
                        acc_m += sc[p] - sc[1 - p]
                    feats.append(encode_state(obs))
                    masks.append(legal_mask(obs))
                    pis.append(heuristic_prior(obs))
                    zs.append(acc_z / K)
                    margins.append(acc_m / K)
            env.step(heuristic_action(obs, rng, params))
        n_games += 1
    feats = np.stack(feats).astype(np.float32)
    masks = np.stack(masks)
    pis = np.stack(pis).astype(np.float32)
    z = np.array(zs, np.float32)
    margin = np.array(margins, np.float32)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, feats=feats, mask=masks, pi=pis, z=z, margin=margin, gen=np.full(len(z), 1, np.int32))
    return {"games": n_games, "samples": int(len(z)), "seconds": time.time() - t0, "out_path": out_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--games", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--playouts", type=int, default=8, help="K playouts per labelled position")
    ap.add_argument("--every", type=int, default=3, help="label every n-th real decision of a game")
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=16, help="number of shards (jobs); >= workers so progress is saved often")
    args = ap.parse_args()
    seeds = list(range(args.seed * 1_000_000, args.seed * 1_000_000 + args.games))
    jobs = []
    for j in range(args.jobs):
        chunk = seeds[j::args.jobs]
        if chunk:
            jobs.append((chunk, args.epsilon, args.playouts, args.every, args.seed * 1000 + j,
                         os.path.join(args.out_dir, f"gen0001_j{j:02d}.npz")))
    t0 = time.time()
    total = 0
    with mp.get_context("fork").Pool(args.workers) as pool:
        for res in pool.imap_unordered(worker, jobs):
            total += res["samples"]
            print(f"[{time.time() - t0:6.0f}s] {res['out_path']}: {res['games']} games, {res['samples']} samples "
                  f"({res['seconds']:.0f}s)", flush=True)
    print(f"done: {total} labelled positions in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
