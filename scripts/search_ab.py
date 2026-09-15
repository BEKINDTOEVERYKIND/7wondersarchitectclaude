#!/usr/bin/env python3
"""Search-level A/B of heuristic knowledge: rollout-MCTS (heuristic prior + heuristic playouts) built from a
VARIANT HeuristicParams against the same search built from the defaults, on paired deals (both seats).

Unlike scripts/heuristic_ab.py this tests whether a change in the *knowledge* still wins once both sides
search with it (the playouts and priors of each side use that side's parameters).

Usage: python scripts/search_ab.py --games 10 --sims 200 "green_early_bonus=3 | res_unit=0.5"
"""
import argparse
import sys
import time
from dataclasses import replace

import numpy as np

from sevenwa.agents.heuristic import HeuristicParams, heuristic_action, heuristic_prior
from sevenwa.agents.mcts_agent import MCTSAgent
from sevenwa.engine.actions import Actions
from sevenwa.engine.env import make_env
from sevenwa.search.mcts import MCTSConfig
from sevenwa.search.rollout import RolloutEvaluator
from sevenwa.train.arena import play_game
from sevenwa.train.elo import wilson_interval


def make(params: HeuristicParams, sims: int, playouts: int, seed: int, name: str) -> MCTSAgent:
    rng = np.random.default_rng(seed)
    ev = RolloutEvaluator(Actions.NUM, playouts_per_leaf=playouts,
                          policy_fn=lambda s, r: heuristic_action(s, r, params), rng=rng,
                          prior_fn=lambda s: heuristic_prior(s, params))
    return MCTSAgent(ev, MCTSConfig(num_simulations=sims), name=name, rng=rng, num_actions=Actions.NUM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10, help="deals; each is played from both seats")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--playouts", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("variants", nargs="*")
    args = ap.parse_args()
    base = HeuristicParams()
    for v in [x.strip() for x in " ".join(args.variants).split("|") if x.strip()]:
        kv = {}
        for item in v.split():
            k, _, val = item.partition("=")
            kv[k] = type(getattr(base, k))(float(val)) if not isinstance(getattr(base, k), bool) else val.lower() == "true"
        pa = replace(base, **kv)
        a = make(pa, args.sims, args.playouts, args.seed, "variant")
        b = make(base, args.sims, args.playouts, args.seed + 1, "default")
        t0 = time.time()
        wins = 0.0
        margins = []
        for g in range(args.games):
            for first in (0, 1):
                env = make_env(args.seed * 100 + g)
                pair = (a, b) if first == 0 else (b, a)
                r, s, _ = play_game(env, pair)
                ra = r[0] if first == 0 else r[1]
                sa = (s[0] - s[1]) if first == 0 else (s[1] - s[0])
                wins += 1.0 if ra > 0 else (0.5 if ra == 0 else 0.0)
                margins.append(sa)
                print(f"  game {g} seat {first}: {'W' if ra > 0 else ('D' if ra == 0 else 'L')} {sa:+d}", flush=True)
        n = 2 * args.games
        lo, hi = wilson_interval(wins, n)
        print(f"{v:40s} score {wins / n * 100:5.1f}% [{lo * 100:4.1f}-{hi * 100:4.1f}]  margin {np.mean(margins):+.2f}  "
              f"({n} games at {args.sims} sims, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
