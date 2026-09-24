#!/usr/bin/env python3
"""Value training data from the search's OWN leaf positions (playout-labelled).

``scripts/playout_relabel.py`` labels positions of ε-greedy heuristic games; the network's value
is already about as accurate as a playout mean there, and the remaining gap to the hybrid
evaluator is *off-distribution*: the search visits lines the heuristic never plays.  This
script therefore plays games with the network-only MCTS agent (``net:<ckpt>:<sims>``, PUCT,
greedy pick) — half against itself, half against ``heuristic:<opp-epsilon>`` with the network's
seat alternating — and after every ``--every``-th search of a network agent it walks the search
tree (``MCTSAgent.last_result.root``; every ``Node`` stores its ``GameState``), collects the
visited *decision* nodes below the root with at least ``--min-visits`` visits and draws at most
``--leaves-per-decision`` of them without replacement with probability ∝ ``(1 + depth) / √N``
(deeper and rarer nodes preferred; ``depth`` counts decision edges from the root).  Each drawn
state is labelled from *its own mover's* perspective with the mean result and mean score margin
of ``--playouts`` heuristic playouts, its policy target is the heuristic prior
(``heuristic_prior``), and the samples are written in the self-play npz format (``feats, mask,
pi, z, margin, gen``), one shard per job.  A job whose shard already exists is skipped, so a
killed run can be resumed with the same command; the shards train through
``scripts/expert_iteration.py --skip-generation --extra-search-glob '<out-dir>/*.npz'``.

Feature versions: the search's evaluator always uses the checkpoint's own encoder (chosen by its
``cfg.input_dim``); the samples are written in ``--feature-version``, by default the checkpoint's own
version too (581 features for ``models/imitation_v7.pt``), so the shards train that network as
``expert_iteration.py --teacher <same checkpoint>``; pass another version only when the shards are
meant for a network with another encoder.

Usage:
  python scripts/leaf_relabel.py --net models/imitation_v7.pt --out-dir runs/leaf1/data \
      --games 400 --sims 96 --workers 4 --playouts 8 --leaves-per-decision 8 --every 2 --jobs 16
"""
import argparse
import multiprocessing as mp
import os
import time
from typing import Dict, List, Tuple

import numpy as np

from sevenwa.agents.heuristic import HeuristicAgent, heuristic_action, heuristic_prior
from sevenwa.engine.actions import Actions
from sevenwa.engine.env import Environment
from sevenwa.nn.features import encoders, feature_size, legal_mask
from sevenwa.search.mcts import Node
from sevenwa.search.rollout import playout


def tree_decision_nodes(root: Node, min_visits: int) -> List[Tuple[Node, int]]:
    """Visited decision nodes strictly below ``root`` as ``(node, depth)``; ``depth`` counts the
    decision edges from the root (chance nodes are passed through without adding depth)."""
    out: List[Tuple[Node, int]] = []
    stack: List[Tuple[Node, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        for child in node.children.values():
            if child.N <= 0:
                continue
            if child.is_terminal:
                continue
            if child.is_chance:
                stack.append((child, depth))
                continue
            if child.N >= min_visits:
                out.append((child, depth + 1))
            stack.append((child, depth + 1))
    return out


def sample_leaves(root: Node, k: int, min_visits: int, rng: np.random.Generator) -> List[Tuple[Node, int]]:
    """At most ``k`` decision nodes of the tree, drawn without replacement with weight (1 + depth) / sqrt(N)."""
    cands = tree_decision_nodes(root, min_visits)
    if not cands or k <= 0:
        return []
    if len(cands) <= k:
        return cands
    w = np.array([(1.0 + d) / np.sqrt(n.N) for n, d in cands], dtype=np.float64)
    w /= w.sum()
    idx = rng.choice(len(cands), size=k, replace=False, p=w)
    return [cands[int(i)] for i in idx]


def label_state(state, playouts: int, rng: np.random.Generator) -> Tuple[float, float]:
    """Mean result and mean final score margin of ``playouts`` heuristic playouts, from the
    perspective of the player to move in ``state``."""
    mover = state.to_move()
    acc_z = 0.0
    acc_m = 0.0
    for _ in range(playouts):
        r, m = playout(state, rng, heuristic_action, return_margin=True)
        acc_z += r[mover]
        acc_m += m if mover == 0 else -m
    return acc_z / playouts, acc_m / playouts


def worker(job: Dict) -> Dict:
    import torch
    from sevenwa.agents.mcts_agent import MCTSAgent
    from sevenwa.nn.evaluator import TorchEvaluator
    from sevenwa.nn.features import encoder_for_dim
    from sevenwa.nn.model import PolicyValueNet
    from sevenwa.search.mcts import MCTSConfig

    torch.set_num_threads(1)
    t0 = time.time()
    rng = np.random.default_rng(job["rng_seed"])
    net = PolicyValueNet.load(job["net"])
    net_version, net_encode, _ = encoder_for_dim(net.cfg.input_dim)  # the search: the checkpoint's own encoder
    version = int(job.get("feature_version") or net_version)  # default: the shards train this checkpoint
    _, encode_state = encoders(version)  # the written samples
    ev = TorchEvaluator(net, net_encode, num_threads=1)
    cfg = MCTSConfig(num_simulations=job["sims"], batch_size=8)
    net_agents = [MCTSAgent(ev, cfg, name="net", rng=np.random.default_rng(rng.integers(2 ** 31)),
                            num_actions=Actions.NUM, root_mode=job["root_mode"], temperature=1.0,
                            temperature_moves=job["temp_moves"]) for _ in range(2)]
    heur = HeuristicAgent(epsilon=job["opp_epsilon"], rng=np.random.default_rng(rng.integers(2 ** 31)))
    feats, masks, pis, zs, margins, depths, visits = [], [], [], [], [], [], []
    n_games = 0
    n_searches = 0
    net_wins = 0.0
    heur_games = 0
    for seed, mode, net_seat in job["games"]:
        env = Environment(seed=seed)
        if mode == "self":
            agents = [net_agents[0], net_agents[1]]
            is_net = [True, True]
        else:
            agents = [None, None]
            agents[net_seat] = net_agents[0]
            agents[1 - net_seat] = heur
            is_net = [net_seat == 0, net_seat == 1]
        for a in agents:
            a.new_game()
        k = 0
        while not env.is_terminal():
            p = env.to_move()
            obs = env.observe(p)
            a = agents[p].select_action(obs)
            if is_net[p]:
                n_searches += 1
                k += 1
                if k % job["every"] == 0:
                    root = agents[p].last_result.root
                    for node, depth in sample_leaves(root, job["leaves"], job["min_visits"], rng):
                        s = node.state
                        z, m = label_state(s, job["playouts"], rng)
                        feats.append(encode_state(s))
                        masks.append(legal_mask(s))
                        pis.append(heuristic_prior(s))
                        zs.append(z)
                        margins.append(m)
                        depths.append(depth)
                        visits.append(node.N)
            env.step(a)
        n_games += 1
        if mode != "self":
            heur_games += 1
            r = env.returns()
            net_wins += 1.0 if r[net_seat] > 0 else (0.5 if r[net_seat] == 0 else 0.0)
    out_path = job["out_path"]
    n = len(zs)
    arrs = {
        "feats": np.stack(feats).astype(np.float32) if n else np.zeros((0, feature_size(version)), np.float32),
        "mask": np.stack(masks) if n else np.zeros((0, Actions.NUM), bool),
        "pi": np.stack(pis).astype(np.float32) if n else np.zeros((0, Actions.NUM), np.float32),
        "z": np.array(zs, np.float32), "margin": np.array(margins, np.float32),
        "gen": np.full(n, job["generation"], np.int32),
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".partial.npz"
    np.savez_compressed(tmp, **arrs)
    os.replace(tmp, out_path)
    return {"games": n_games, "samples": n, "searches": n_searches, "seconds": time.time() - t0, "out_path": out_path,
            "feature_version": version,
            "mean_depth": float(np.mean(depths)) if depths else 0.0,
            "mean_visits": float(np.mean(visits)) if visits else 0.0,
            "net_score": net_wins / heur_games if heur_games else float("nan"), "heur_games": heur_games}


def make_games(games: int, seed: int) -> List[Tuple[int, str, int]]:
    """Global game list: (env seed, "self" | "heur", network seat); modes alternate and the
    network's seat against the heuristic alternates too."""
    out = []
    for i in range(games):
        mode = "self" if i % 2 == 0 else "heur"
        out.append((seed * 1_000_000 + i, mode, (i // 2) % 2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--sims", type=int, default=96)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--playouts", type=int, default=8, help="K heuristic playouts per labelled leaf")
    ap.add_argument("--leaves-per-decision", type=int, default=8, help="max tree positions drawn per harvested search")
    ap.add_argument("--every", type=int, default=2, help="harvest the tree after every n-th search of a network agent")
    ap.add_argument("--min-visits", type=int, default=2, help="only tree nodes with at least this many visits")
    ap.add_argument("--opp-epsilon", type=float, default=0.1, help="ε of the heuristic opponent")
    ap.add_argument("--root-mode", default="puct", choices=["puct", "gumbel"])
    ap.add_argument("--temp-moves", type=int, default=0,
                    help="network agents sample their move by visit count for the first n moves (0 = greedy, as net:<ckpt>)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=16, help="number of shards (jobs); >= workers so progress is saved often")
    ap.add_argument("--generation", type=int, default=1, help="``gen`` value written into the shards")
    ap.add_argument("--feature-version", type=int, default=None,
                    help="feature encoding of the WRITTEN samples (default: the checkpoint's own version; the search "
                         "always uses the checkpoint's own encoder)")
    args = ap.parse_args()
    if args.feature_version is not None:
        feature_size(args.feature_version)  # validates
    os.makedirs(args.out_dir, exist_ok=True)
    games = make_games(args.games, args.seed)
    jobs = []
    skipped = 0
    for j in range(args.jobs):
        chunk = games[j::args.jobs]
        if not chunk:
            continue
        out_path = os.path.join(args.out_dir, f"gen{args.generation:04d}_j{j:02d}.npz")
        if os.path.exists(out_path):
            skipped += 1
            continue
        jobs.append({"net": args.net, "games": chunk, "sims": args.sims, "playouts": args.playouts,
                     "leaves": args.leaves_per_decision, "every": args.every, "min_visits": args.min_visits,
                     "opp_epsilon": args.opp_epsilon, "root_mode": args.root_mode, "temp_moves": args.temp_moves,
                     "rng_seed": args.seed * 1000 + j, "out_path": out_path, "generation": args.generation,
                     "feature_version": args.feature_version})
    if skipped:
        print(f"skipping {skipped} finished jobs", flush=True)
    t0 = time.time()
    total = 0
    scores = []

    def report(res):
        nonlocal total
        total += res["samples"]
        if res["heur_games"]:
            scores.append((res["net_score"], res["heur_games"]))
        print(f"[{time.time() - t0:6.0f}s] {res['out_path']}: {res['games']} games, {res['searches']} searches, "
              f"{res['samples']} samples (mean depth {res['mean_depth']:.2f}, mean visits {res['mean_visits']:.1f}; "
              f"{res['seconds']:.0f}s)", flush=True)

    if args.workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            report(worker(job))
    else:
        with mp.get_context("spawn").Pool(min(args.workers, len(jobs))) as pool:
            for res in pool.imap_unordered(worker, jobs):
                report(res)
    if scores:
        n = sum(g for _, g in scores)
        print(f"network vs heuristic:{args.opp_epsilon:g}: {sum(s * g for s, g in scores) / n * 100:.1f}% over {n} games")
    print(f"done: {total} labelled leaf positions in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
