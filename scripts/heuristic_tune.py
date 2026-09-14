#!/usr/bin/env python3
"""Coordinate-descent tuning of HeuristicParams with paired self-play A/B tests.

For each parameter, try scaled variants against the current best; accept a variant if it scores
>= ``--accept`` over ``--games`` paired games and confirms at 2x games.  Writes the best parameter
set to ``--out`` (JSON) after every acceptance.
"""
import argparse
import json
import sys
import time
from dataclasses import asdict, fields, replace

sys.path.insert(0, ".")
from scripts.heuristic_ab import paired_match  # noqa: E402
from sevenwa.agents.heuristic import HeuristicParams  # noqa: E402

SKIP = {"epsilon", "tie_noise", "prior_temperature", "max_turns"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--accept", type=float, default=0.535)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--out", default="runs/heuristic_tuned.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    best = HeuristicParams()
    history = []
    names = [f.name for f in fields(HeuristicParams) if f.name not in SKIP]
    seed = args.seed
    for p in range(args.passes):
        improved = False
        for name in names:
            cur = getattr(best, name)
            if isinstance(cur, bool):
                continue
            if isinstance(cur, int):
                cands = [max(1, cur - 1), cur + 1]
            else:
                cands = [cur * 0.6, cur * 1.4] if cur != 0 else [0.2, -0.2]
            for val in cands:
                trial = replace(best, **{name: type(cur)(val)})
                seed += 7919
                score, ci, margin = paired_match(trial, best, args.games, seed)
                line = f"pass {p} {name}={val:g} (was {cur:g}): {score * 100:.1f}% [{ci[0] * 100:.1f}-{ci[1] * 100:.1f}]"
                if score >= args.accept:
                    seed += 7919
                    score2, ci2, _ = paired_match(trial, best, 2 * args.games, seed)
                    line += f" -> confirm {score2 * 100:.1f}%"
                    if score2 >= args.accept - 0.01:
                        best = trial
                        improved = True
                        line += "  ACCEPTED"
                        with open(args.out, "w") as f:
                            json.dump(asdict(best), f, indent=2)
                print(line, flush=True)
                history.append(line)
        if not improved:
            break
    print("final:", json.dumps({k: v for k, v in asdict(best).items() if v != getattr(HeuristicParams(), k)}))


if __name__ == "__main__":
    main()
