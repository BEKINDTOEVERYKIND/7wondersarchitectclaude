#!/usr/bin/env python3
"""Paired A/B between two agent spec strings: every deal is played from both seats.

Usage: python scripts/spec_ab.py --a <spec> --b <spec> --games 10 --seed 0
Prints one line per game and a final score line for A (win = 1, draw = 0.5).
"""
import argparse
import time

import numpy as np

from sevenwa.agents.factory import make_agent
from sevenwa.engine.env import make_env
from sevenwa.train.arena import play_game
from sevenwa.train.elo import wilson_interval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--games", type=int, default=10, help="deals; each is played from both seats")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    a = make_agent(args.a, seed=args.seed * 2 + 1)
    b = make_agent(args.b, seed=args.seed * 2 + 2)
    t0 = time.time()
    wins = 0.0
    margins = []
    for g in range(args.games):
        for first in (0, 1):
            env = make_env(args.seed * 1000 + g)
            pair = (a, b) if first == 0 else (b, a)
            r, s, plies = play_game(env, pair)
            ra = r[0] if first == 0 else r[1]
            sa = (s[0] - s[1]) if first == 0 else (s[1] - s[0])
            wins += 1.0 if ra > 0 else (0.5 if ra == 0 else 0.0)
            margins.append(sa)
            print(f"  deal {g} A-seat {first}: {'W' if ra > 0 else ('D' if ra == 0 else 'L')} {sa:+d} ({plies} plies, {time.time() - t0:.0f}s)", flush=True)
    n = 2 * args.games
    lo, hi = wilson_interval(wins, n)
    print(f"A={args.a}\nB={args.b}\nscore {wins / n * 100:5.1f}% [{lo * 100:4.1f}-{hi * 100:4.1f}]  margin {np.mean(margins):+.2f}  ({n} games, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
