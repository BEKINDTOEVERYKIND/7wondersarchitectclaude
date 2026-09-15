#!/usr/bin/env python3
"""Resumable search self-play data generation in chunks (one directory of shards per chunk).

Each chunk is written to ``<out-dir>/c<k>/gen0001_w<i>.npz`` when it completes, so a killed run
loses at most one chunk; re-running the same command skips finished chunks.  Feed the shards to
``scripts/expert_iteration.py --skip-generation --extra-search-glob '<out-dir>/c*/gen0001_*.npz'``.
"""
import argparse
import glob
import os
import time

from sevenwa.train.selfplay import SelfPlayConfig, run_selfplay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--games-per-chunk", type=int, default=200)
    ap.add_argument("--sims", type=int, default=96)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--evaluator", default="hybrid", choices=["net", "hybrid"])
    ap.add_argument("--hybrid-lambda", type=float, default=0.5)
    ap.add_argument("--playouts", type=int, default=1)
    ap.add_argument("--value-mix", type=float, default=0.5)
    ap.add_argument("--temp-moves", type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "chunks.log")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    cfg = SelfPlayConfig(num_simulations=args.sims, batch_size=8, root_mode="gumbel", temperature_moves=args.temp_moves,
                         value_mix=args.value_mix, evaluator=args.evaluator, hybrid_lambda=args.hybrid_lambda,
                         rollout_playouts=args.playouts)
    for k in range(args.chunks):
        d = os.path.join(args.out_dir, f"c{k:02d}")
        if glob.glob(os.path.join(d, "gen0001_*.npz")):
            log(f"chunk {k}: already done, skipping")
            continue
        tmp = d + ".partial"
        os.makedirs(tmp, exist_ok=True)
        t0 = time.time()
        run_selfplay(cfg, args.games_per_chunk, args.teacher, tmp, 1, args.workers, seed=args.seed * 100 + k, log=log)
        os.rename(tmp, d)
        log(f"chunk {k}: {args.games_per_chunk} games in {time.time() - t0:.0f}s -> {d}")
    log("all chunks done")


if __name__ == "__main__":
    main()
