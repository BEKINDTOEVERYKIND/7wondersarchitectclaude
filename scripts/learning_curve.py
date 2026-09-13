#!/usr/bin/env python3
"""Print the learning curve of a run from runs/<name>/metrics.jsonl."""
import json
import sys


def main(path: str) -> None:
    print(f"{'gen':>4} {'policy':>7} {'value':>6} {'acc':>5} | {'fresh_v':>7} {'fresh_p':>7} | {'gate':>5} {'prom':>4} | evals")
    for line in open(path):
        d = json.loads(line)
        t = d.get("train", {})
        f = d.get("fresh") or {}
        g = d.get("gate") or {}
        ev = ", ".join(f"{e['b'][:10]} {e['wins'][0]:.0f}-{e['wins'][1]:.0f}" for e in d.get("evals", []))
        print(f"{d['generation']:>4} {t.get('policy', float('nan')):7.3f} {t.get('value', float('nan')):6.3f} {t.get('acc', float('nan')):5.2f} | "
              f"{f.get('value', float('nan')):7.3f} {f.get('policy', float('nan')):7.3f} | {g.get('score_a', float('nan')):5.2f} "
              f"{'yes' if d.get('promoted') else 'no':>4} | {ev}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/default/metrics.jsonl")
