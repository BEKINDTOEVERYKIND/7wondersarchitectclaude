#!/usr/bin/env python3
"""Paired A/B test of HeuristicParams variants: same deals, both seats, vs the default parameters.

Usage: python scripts/heuristic_ab.py --games 600 cat_value=1.6 cat_steal=1.6 | deny=0.8 | ...
Each argument group separated by '|' is one variant (fields set together).
"""
import argparse
import sys
import time
from dataclasses import replace

import numpy as np

from sevenwa.agents.heuristic import HeuristicAgent, HeuristicParams
from sevenwa.engine.env import make_env
from sevenwa.train.arena import play_game
from sevenwa.train.elo import wilson_interval


def paired_match(pa: HeuristicParams, pb: HeuristicParams, games: int, seed: int):
    a = HeuristicAgent(params=pa, seed=seed)
    b = HeuristicAgent(params=pb, seed=seed + 1)
    wins = 0.0
    margins = []
    for g in range(games):
        for first in (0, 1):  # both seats on the same deal
            env = make_env(seed + g)
            pair = (a, b) if first == 0 else (b, a)
            r, s, _ = play_game(env, pair)
            ra = r[0] if first == 0 else r[1]
            sa = (s[0] - s[1]) if first == 0 else (s[1] - s[0])
            wins += 1.0 if ra > 0 else (0.5 if ra == 0 else 0.0)
            margins.append(sa)
    n = 2 * games
    lo, hi = wilson_interval(wins, n)
    return wins / n, (lo, hi), float(np.mean(margins))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("variants", nargs="*")
    args = ap.parse_args()
    spec = " ".join(args.variants)
    variants = [v.strip() for v in spec.split("|") if v.strip()]
    base = HeuristicParams()
    for v in variants:
        kv = {}
        for item in v.split():
            k, _, val = item.partition("=")
            kv[k] = type(getattr(base, k))(float(val)) if not isinstance(getattr(base, k), bool) else val.lower() == "true"
        pa = replace(base, **kv)
        t = time.time()
        score, (lo, hi), margin = paired_match(pa, base, args.games, args.seed)
        print(f"{v:40s} score {score * 100:5.1f}% [{lo * 100:4.1f}-{hi * 100:4.1f}]  margin {margin:+.2f}  ({2 * args.games} games, {time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
