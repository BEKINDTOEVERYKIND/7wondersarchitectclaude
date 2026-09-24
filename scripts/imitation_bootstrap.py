#!/usr/bin/env python3
"""Imitation bootstrap: heuristic self-play data -> network, with a held-out set and evaluation.

Usage: python scripts/imitation_bootstrap.py --run-dir runs/v3 --games 20000 --holdout-games 1000

``--feature-version`` (default: the latest, 2) selects the input encoding of the data and of the new
network; ``--feature-version 1`` reproduces the 581-feature networks such as ``models/imitation_v7.pt``.
"""
import argparse
import glob
import os
import time

import numpy as np
import torch

from sevenwa.engine.actions import Actions
from sevenwa.nn.features import FEATURE_VERSION_LATEST, feature_size
from sevenwa.nn.model import compute_loss
from sevenwa.train.pipeline import PipelineConfig, check_feature_width, new_network
from sevenwa.train.replay import ReplayBuffer
from sevenwa.train.selfplay import run_imitation
from sevenwa.train.trainer import TrainConfig, Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/v3")
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--holdout-games", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feature-version", type=int, default=FEATURE_VERSION_LATEST,
                    help="input feature encoding of the data and the network (1 = 581 features, 2 = + EXPERT block)")
    args = ap.parse_args()
    n_feats = feature_size(args.feature_version)
    data_dir = os.path.join(args.run_dir, "data")
    hold_dir = os.path.join(args.run_dir, "holdout")
    os.makedirs(os.path.join(args.run_dir, "ckpt"), exist_ok=True)
    t0 = time.time()
    run_imitation(args.games, data_dir, 0, args.workers, seed=args.seed, epsilon=args.epsilon,
                  feature_version=args.feature_version)
    run_imitation(args.holdout_games, hold_dir, 0, args.workers, seed=args.seed + 10_000_000, epsilon=args.epsilon,
                  feature_version=args.feature_version)
    buf = ReplayBuffer(n_feats, Actions.NUM)
    n = buf.load_shards(os.path.join(data_dir, "gen0000_*.npz"))
    check_feature_width(buf, n_feats, data_dir)
    ho = ReplayBuffer(n_feats, Actions.NUM)
    ho.load_shards(os.path.join(hold_dir, "gen0000_*.npz"))
    check_feature_width(ho, n_feats, hold_dir)
    hb = ho.sample(min(len(ho), 20000), np.random.default_rng(0))
    print(f"train samples {n}, holdout {len(ho)} ({time.time() - t0:.0f}s)", flush=True)

    def evaluate(net):
        net.eval()
        with torch.no_grad():
            _, parts = compute_loss(net, torch.from_numpy(hb.feats), torch.from_numpy(hb.mask), torch.from_numpy(hb.pi),
                                    torch.from_numpy(hb.z), torch.from_numpy(hb.margin))
        return {k: round(v, 3) for k, v in parts.items()}

    net = new_network(PipelineConfig(net_width=args.width, net_depth=args.depth, net_dropout=args.dropout,
                                     feature_version=args.feature_version))
    torch.set_num_threads(4)
    trainer = Trainer(net, TrainConfig(batch_size=args.batch, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                                       num_threads=4))
    rng = np.random.default_rng(args.seed)
    # train in half-epoch chunks so the holdout curve is visible
    chunks = int(max(1, round(args.epochs / 0.5)))
    for c in range(chunks):
        trainer.cfg.epochs = 0.5
        st = trainer.train(buf, rng, log=lambda m: None)
        print(f"after {0.5 * (c + 1):.1f} epochs: train policy {st['policy']:.3f} value {st['value']:.3f} acc {st['acc']:.3f} | "
              f"holdout {evaluate(net)} ({time.time() - t0:.0f}s)", flush=True)
        net.save(os.path.join(args.run_dir, "ckpt", "champion.pt"), extra={"generation": 0, "imitation_epochs": 0.5 * (c + 1)})
    print("saved", os.path.join(args.run_dir, "ckpt", "champion.pt"), flush=True)


if __name__ == "__main__":
    main()
