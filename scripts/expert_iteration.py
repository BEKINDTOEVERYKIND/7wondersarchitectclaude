#!/usr/bin/env python3
"""Expert iteration with a FIXED teacher and match-based selection.

One round:
  1. generate N search-play games with the teacher network (Gumbel MCTS, exploration for the
     first moves), recording the search's improved policy and the game outcome (optionally
     blended with the root value);
  2. fine-tune the teacher on (search games ⊕ a slice of the imitation data) with a low learning
     rate, holding out a fraction of the search games for loss metrics;
  3. play the candidate against the teacher at a high simulation count and keep it only if it
     wins clearly (the noisy per-generation gate of the pipeline is replaced by one decisive match).

Usage:
  python scripts/expert_iteration.py --teacher runs/v4/ckpt/champion.pt --run-dir runs/ei1 \
      --games 2400 --sims 96 --workers 3 --imitation-glob 'runs/v4/imitation/gen0000_*.npz'

Feature versions: the teacher's ``cfg.input_dim`` fixes the encoding of everything in the round — the
search games are generated with the teacher's encoder, the buffers are sized by it, and imitation /
extra shards written with another version are rejected with a clear error.
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch

from sevenwa.agents.factory import make_agent
from sevenwa.engine.actions import Actions
from sevenwa.engine.env import make_env
from sevenwa.nn.features import version_for_dim
from sevenwa.nn.model import PolicyValueNet, compute_loss
from sevenwa.search.mcts import MCTSConfig
from sevenwa.train.arena import play_match
from sevenwa.train.replay import ReplayBuffer
from sevenwa.train.pipeline import check_feature_width
from sevenwa.train.selfplay import SelfPlayConfig, run_selfplay
from sevenwa.train.trainer import TrainConfig, Trainer


def holdout_loss(net, buf: ReplayBuffer, n: int = 30000):
    if len(buf) == 0:
        return {}
    b = buf.sample(min(n, len(buf)), np.random.default_rng(0))
    net.eval()
    with torch.no_grad():
        _, parts = compute_loss(net, torch.from_numpy(b.feats), torch.from_numpy(b.mask), torch.from_numpy(b.pi),
                                torch.from_numpy(b.z), torch.from_numpy(b.margin))
    return {k: round(float(v), 3) for k, v in parts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--run-dir", default="runs/ei1")
    ap.add_argument("--games", type=int, default=2400)
    ap.add_argument("--sims", type=int, default=96)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--value-mix", type=float, default=0.3)
    ap.add_argument("--evaluator", default="net", choices=["net", "hybrid"],
                    help="leaf evaluator of the teacher's search: the network alone or its values blended with heuristic playouts")
    ap.add_argument("--hybrid-lambda", type=float, default=0.5)
    ap.add_argument("--playouts", type=int, default=1)
    ap.add_argument("--temp-moves", type=int, default=30)
    ap.add_argument("--imitation-glob", default=None, help="imitation shards to mix in (a random slice is used)")
    ap.add_argument("--imitation-fraction", type=float, default=0.5, help="share of imitation samples in the training set")
    ap.add_argument("--holdout-fraction", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--match-games", type=int, default=40)
    ap.add_argument("--match-sims", type=int, default=300)
    ap.add_argument("--accept", type=float, default=0.55)
    ap.add_argument("--skip-generation", action="store_true", help="reuse existing shards in run-dir/data")
    ap.add_argument("--extra-search-glob", action="append", default=[],
                    help="search-play shards from earlier rounds to include in training (repeatable)")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.run_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(args.run_dir, "ckpt"), exist_ok=True)
    log_path = os.path.join(args.run_dir, "log.txt")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    t0 = time.time()
    teacher = PolicyValueNet.load(args.teacher)
    n_feats = teacher.cfg.input_dim
    log(f"teacher {args.teacher}: {n_feats} input features (feature version {version_for_dim(n_feats)})")
    data_dir = os.path.join(args.run_dir, "data")
    if not args.skip_generation:
        cfg = SelfPlayConfig(num_simulations=args.sims, batch_size=8, root_mode="gumbel", temperature_moves=args.temp_moves,
                             value_mix=args.value_mix, evaluator=args.evaluator, hybrid_lambda=args.hybrid_lambda,
                             rollout_playouts=args.playouts)
        run_selfplay(cfg, args.games, args.teacher, data_dir, 1, args.workers, seed=args.seed, log=log)
    shards = sorted(glob.glob(os.path.join(data_dir, "gen0001_*.npz")))
    for g in args.extra_search_glob:
        shards += sorted(glob.glob(g))
    # hold out whole worker shards' tail: simplest reproducible split is by sample index
    train = ReplayBuffer(n_feats, Actions.NUM)
    hold = ReplayBuffer(n_feats, Actions.NUM)
    rng = np.random.default_rng(args.seed)
    for p in shards:
        d = np.load(p)
        n = len(d["z"])
        if d["feats"].ndim != 2 or d["feats"].shape[1] != n_feats:
            raise ValueError(f"{p}: {d['feats'].shape[-1]} features, but the teacher expects {n_feats} "
                             f"(feature version {version_for_dim(n_feats)})")
        cut = int(n * (1 - args.holdout_fraction))
        train.add_game(d["feats"][:cut], d["mask"][:cut], d["pi"][:cut], d["z"][:cut], d["margin"][:cut], 1)
        hold.add_game(d["feats"][cut:], d["mask"][cut:], d["pi"][cut:], d["z"][cut:], d["margin"][cut:], 1)
    n_search = len(train)
    if args.imitation_glob:
        imit = ReplayBuffer(n_feats, Actions.NUM)
        imit.load_shards(args.imitation_glob)
        check_feature_width(imit, n_feats, args.imitation_glob)
        want = int(n_search * args.imitation_fraction / max(1e-9, 1 - args.imitation_fraction))
        b = imit.sample(min(want, len(imit)), rng)
        train.add_game(b.feats, b.mask, b.pi, b.z, b.margin, 0)
        log(f"training set: {n_search} search samples + {len(b.z)} imitation samples; holdout {len(hold)} search samples")
    else:
        log(f"training set: {n_search} search samples; holdout {len(hold)}")

    log(f"teacher on search holdout: {holdout_loss(teacher, hold)}")
    cand = PolicyValueNet.load(args.teacher)
    torch.set_num_threads(4)
    trainer = Trainer(cand, TrainConfig(batch_size=args.batch, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                                        num_threads=4))
    chunks = int(max(1, round(args.epochs / 0.5)))
    for c in range(chunks):
        trainer.cfg.epochs = 0.5
        st = trainer.train(train, rng, log=lambda m: None)
        log(f"after {0.5 * (c + 1):.1f} epochs: train policy {st['policy']:.3f} value {st['value']:.3f} acc {st['acc']:.3f} | "
            f"search holdout {holdout_loss(cand, hold)}")
    cand_path = os.path.join(args.run_dir, "ckpt", "candidate.pt")
    cand.save(cand_path, extra={"teacher": args.teacher, "games": args.games})

    # decisive match at high simulation count
    a = make_agent(f"net:{cand_path}:{args.match_sims}", seed=args.seed + 1)
    b = make_agent(f"net:{args.teacher}:{args.match_sims}", seed=args.seed + 2)
    res = play_match(make_env, (a, b), args.match_games, seed=args.seed + 1000)
    log("match: " + res.summary(("candidate", "teacher")))
    accepted = res.score(0) >= args.accept
    if accepted:
        best = os.path.join(args.run_dir, "ckpt", "best.pt")
        cand.save(best, extra={"teacher": args.teacher, "games": args.games, "match": res.summary(("candidate", "teacher"))})
        log(f"ACCEPTED -> {best}")
    else:
        log("rejected (teacher stays)")
    with open(os.path.join(args.run_dir, "result.json"), "w") as f:
        json.dump({"accepted": accepted, "score": res.score(0), "wins": res.wins, "draws": res.draws,
                   "seconds": time.time() - t0}, f, indent=2)


if __name__ == "__main__":
    main()
