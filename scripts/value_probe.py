#!/usr/bin/env python3
"""Off-distribution value probe: how good is a value head away from the imitation distribution?

Decision positions are sampled from ε-greedy heuristic games at each requested epsilon (as in
``scripts/playout_relabel.py``): a small ε gives the imitation distribution, a large one
(``0.5``) gives positions the heuristic would never reach on its own, i.e. the kind of lines a
search visits.  Every position is labelled with the mean result and mean score margin of ``K``
heuristic playouts from the mover's belief state — the estimate the hybrid evaluator uses at
its leaves — and each network's value head (and score head) is compared against that label:

* value MSE, Pearson correlation, mean bias (prediction − target) and sign agreement (share of
  positions with a non-zero target whose predicted sign matches), overall and broken down by
  turn bucket (0-10 / 10-20 / 20-30 / 30+), by ``min(cards_needed)`` over the two players (the
  heuristic's count of cards still needed to finish the Wonder) and by decision kind;
* ``z_var`` (mean target²) is the MSE of the constant-0 predictor, and ``margin MSE / corr``
  compare the score head against the playout margin.

The playouts dominate the cost (~5 ms each): 1 000 positions × 8 playouts ≈ 40 s per epsilon.
Each network is queried with its own feature encoding (chosen by its ``cfg.input_dim``), so
581-feature (version 1) and later networks can be compared on the same positions.

Usage:
  python scripts/value_probe.py --nets models/imitation_v7.pt,models/value_v8.pt \
      --epsilons 0.1,0.5 --positions 1000 --playouts 8 --every 4 --seed 0
"""
import argparse
import multiprocessing as mp
import time
from dataclasses import replace
from typing import Dict, List, Optional, Sequence

import numpy as np

from sevenwa.agents.heuristic import HeuristicParams, heuristic_action
from sevenwa.engine.env import Environment
from sevenwa.engine.state import DECISION_NAMES
from sevenwa.nn.features import FEATURE_VERSION_LATEST, encoders, feature_size, version_for_dim
from sevenwa.search.rollout import playout

try:  # private helper of the heuristic: skipped (bucket omitted) if it disappears
    from sevenwa.agents.heuristic import _make_view
except ImportError:  # pragma: no cover
    _make_view = None

TURN_EDGES = [10, 20, 30]
TURN_LABELS = ["turn 0-10", "turn 10-20", "turn 20-30", "turn 30+"]
NEED_EDGES = [3, 6, 10]
NEED_LABELS = ["need 0-2", "need 3-5", "need 6-9", "need 10+"]


def _bucket(x: int, edges: List[int]) -> int:
    for i, e in enumerate(edges):
        if x < e:
            return i
    return len(edges)


def cards_needed(state) -> Optional[int]:
    """``min`` over both players of the heuristic's ``cards_needed`` (None if unavailable)."""
    if _make_view is None:
        return None
    try:
        return int(min(_make_view(state, 0).cards_needed, _make_view(state, 1).cards_needed))
    except Exception:  # pragma: no cover - the helper's contract changed
        return None


def feats_key(version: int) -> str:
    return f"feats_v{int(version)}"


def sample_positions(n_positions: int, eps: float, playouts: int, every: int, seed: int,
                     versions: Sequence[int] = (FEATURE_VERSION_LATEST,)) -> Dict[str, np.ndarray]:
    """Sample ``n_positions`` labelled decision positions from ε-greedy heuristic games.

    Returns arrays: ``feats_v<k>`` (N, F_k) for every requested feature version ``k``, ``z`` and
    ``margin`` (playout means from the mover's perspective), ``turn``, ``need`` (-1 when
    unavailable), ``dkind``.
    """
    rng = np.random.default_rng(seed)
    params = replace(HeuristicParams(), epsilon=eps)
    versions = sorted({int(v) for v in versions})
    enc = {v: encoders(v)[1] for v in versions}
    feats = {v: [] for v in versions}
    zs, ms, turns, needs, kinds = [], [], [], [], []
    game = 0
    while len(zs) < n_positions:
        env = Environment(seed=seed * 1_000_003 + game)
        game += 1
        k = 0
        while not env.is_terminal() and len(zs) < n_positions:
            p = env.to_move()
            obs = env.observe(p)
            if len(obs.legal_actions()) > 1:
                k += 1
                if k % every == 0:
                    acc_z = 0.0
                    acc_m = 0.0
                    for _ in range(playouts):
                        r, m = playout(obs, rng, heuristic_action, return_margin=True)
                        acc_z += r[p]
                        acc_m += m if p == 0 else -m
                    for v in versions:
                        feats[v].append(enc[v](obs))
                    zs.append(acc_z / playouts)
                    ms.append(acc_m / playouts)
                    turns.append(int(obs.turn))
                    need = cards_needed(obs)
                    needs.append(-1 if need is None else need)
                    kinds.append(int(obs.dkind))
            env.step(heuristic_action(obs, rng, params))
    out = {feats_key(v): (np.stack(feats[v]).astype(np.float32) if feats[v] else np.zeros((0, feature_size(v)), np.float32))
           for v in versions}
    out.update({"z": np.array(zs, np.float32), "margin": np.array(ms, np.float32), "turn": np.array(turns, np.int32),
                "need": np.array(needs, np.int32), "dkind": np.array(kinds, np.int32), "games": game})
    return out


def _sample_job(args):
    n, eps, playouts, every, seed, versions = args
    return sample_positions(n, eps, playouts, every, seed, versions)


def predict(net, feats: np.ndarray):
    """Value-head and score-head (in points) predictions for a feature batch."""
    import torch
    torch.set_num_threads(1)
    net.eval()
    out_v, out_m = [], []
    with torch.no_grad():
        for i in range(0, len(feats), 1024):
            _, v, s = net(torch.from_numpy(feats[i:i + 1024]))
            out_v.append(v.numpy())
            out_m.append(s.numpy() * float(net.cfg.score_scale))
    return np.concatenate(out_v), np.concatenate(out_m)


def metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """MSE, Pearson correlation, mean bias and sign agreement of ``pred`` against ``target``."""
    n = len(target)
    if n == 0:
        return {"n": 0, "mse": float("nan"), "corr": float("nan"), "bias": float("nan"), "sign": float("nan"),
                "n_sign": 0, "z_var": float("nan")}
    err = pred - target
    if n > 1 and pred.std() > 1e-9 and target.std() > 1e-9:
        corr = float(np.corrcoef(pred, target)[0, 1])
    else:
        corr = float("nan")
    nz = target != 0
    sign = float(np.mean(np.sign(pred[nz]) == np.sign(target[nz]))) if nz.any() else float("nan")
    return {"n": int(n), "mse": float(np.mean(err ** 2)), "corr": corr, "bias": float(np.mean(err)),
            "sign": sign, "n_sign": int(nz.sum()), "z_var": float(np.mean(target ** 2))}


def fmt(label: str, m: Dict[str, float]) -> str:
    return (f"  {label:<14} n={m['n']:5d}  mse={m['mse']:.3f}  corr={m['corr']:+.3f}  bias={m['bias']:+.3f}  "
            f"sign={m['sign'] * 100:5.1f}% (n={m['n_sign']})  z_var={m['z_var']:.3f}")


def report(name: str, eps: float, data: Dict[str, np.ndarray], v_pred: np.ndarray, m_pred: np.ndarray) -> Dict:
    z, margin = data["z"], data["margin"]
    overall = metrics(v_pred, z)
    mm = metrics(m_pred, margin)
    print(f"{name} @ eps={eps:g}:")
    print(fmt("overall", overall) + f"  | margin mse={mm['mse']:.2f} corr={mm['corr']:+.3f} bias={mm['bias']:+.2f}")
    by = {"overall": overall, "margin": mm}
    tb = np.array([_bucket(t, TURN_EDGES) for t in data["turn"]])
    for i, lab in enumerate(TURN_LABELS):
        sel = tb == i
        if sel.any():
            by[lab] = metrics(v_pred[sel], z[sel])
            print(fmt(lab, by[lab]))
    if (data["need"] >= 0).all():
        nb = np.array([_bucket(x, NEED_EDGES) for x in data["need"]])
        for i, lab in enumerate(NEED_LABELS):
            sel = nb == i
            if sel.any():
                by[lab] = metrics(v_pred[sel], z[sel])
                print(fmt(lab, by[lab]))
    for k, lab in enumerate(DECISION_NAMES):
        sel = data["dkind"] == k
        if sel.any():
            by[lab] = metrics(v_pred[sel], z[sel])
            print(fmt(lab, by[lab]))
    return by


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nets", required=True, help="comma-separated checkpoints")
    ap.add_argument("--epsilons", default="0.1,0.5", help="comma-separated ε of the ε-greedy sampling games")
    ap.add_argument("--positions", type=int, default=1000, help="positions per epsilon")
    ap.add_argument("--playouts", type=int, default=8, help="K heuristic playouts per position (the label)")
    ap.add_argument("--every", type=int, default=4, help="sample every n-th real decision of a game")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1,
                    help="processes for the playout labelling (positions are split into this many chunks)")
    args = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    from sevenwa.nn.model import PolicyValueNet

    nets = [p for p in args.nets.split(",") if p]
    epsilons = [float(e) for e in args.epsilons.split(",") if e]
    loaded = {path: PolicyValueNet.load(path) for path in nets}
    net_version = {path: version_for_dim(net.cfg.input_dim) for path, net in loaded.items()}  # each net: its own encoder
    versions = sorted(set(net_version.values())) or [FEATURE_VERSION_LATEST]
    t0 = time.time()
    datasets = {}
    for ei, eps in enumerate(epsilons):
        chunks = max(1, min(args.workers, args.positions))
        per = [args.positions // chunks + (1 if i < args.positions % chunks else 0) for i in range(chunks)]
        jobs = [(n, eps, args.playouts, args.every, args.seed * 100 + ei * 10 + i, versions)
                for i, n in enumerate(per) if n > 0]
        if len(jobs) == 1:
            parts = [_sample_job(jobs[0])]
        else:
            with mp.get_context("fork").Pool(len(jobs)) as pool:
                parts = pool.map(_sample_job, jobs)
        keys = [feats_key(v) for v in versions] + ["z", "margin", "turn", "need", "dkind"]
        data = {k: np.concatenate([p[k] for p in parts]) for k in keys}
        data["games"] = sum(p["games"] for p in parts)
        datasets[eps] = data
        print(f"[{time.time() - t0:5.0f}s] eps={eps:g}: {len(data['z'])} positions from {data['games']} games, "
              f"{args.playouts} playouts each; mean target {data['z'].mean():+.3f}, z_var {np.mean(data['z'] ** 2):.3f}",
              flush=True)
    for path in nets:
        net = loaded[path]
        for eps in epsilons:
            data = datasets[eps]
            v_pred, m_pred = predict(net, data[feats_key(net_version[path])])
            report(path, eps, data, v_pred, m_pred)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
