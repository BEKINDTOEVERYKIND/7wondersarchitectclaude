"""Elo estimation from pairwise results (Bradley–Terry via simple iterative MM updates)."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, Tuple


def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def fit_elo(results: Iterable[Tuple[str, str, float]], anchor: str = "random", anchor_rating: float = 0.0,
            iterations: int = 2000, tol: float = 1e-9) -> Dict[str, float]:
    """``results`` are ``(name_a, name_b, score_a)`` with score in {0, 0.5, 1}.

    Bradley–Terry maximum likelihood via Hunter's minorisation–maximisation iteration
    (γ_i ← W_i / Σ_j n_ij / (γ_i + γ_j)), which converges monotonically; draws count as half a
    win for each side.  Ratings are 400·log10(γ), shifted so that ``anchor`` has ``anchor_rating``.
    A tiny prior pseudo-game against every opponent keeps γ finite for undefeated players.
    """
    wins = defaultdict(float)  # name -> (fractional) wins
    games = defaultdict(float)  # (a,b) with a<b -> games played
    names = set()
    for a, b, sc in results:
        names.add(a); names.add(b)
        wins[a] += sc
        wins[b] += 1.0 - sc
        key = (a, b) if a <= b else (b, a)
        games[key] += 1.0
    names = sorted(names)
    if not names:
        return {}
    eps = 0.01  # prior: 0.01 of a drawn game against every opponent
    for a in names:
        for b in names:
            if a < b:
                games[(a, b)] += 2 * eps
                wins[a] += eps
                wins[b] += eps
    gamma = {n: 1.0 for n in names}
    for _ in range(iterations):
        new = {}
        for i in names:
            denom = 0.0
            for j in names:
                if i == j:
                    continue
                key = (i, j) if i <= j else (j, i)
                n_ij = games.get(key, 0.0)
                if n_ij:
                    denom += n_ij / (gamma[i] + gamma[j])
            new[i] = wins[i] / denom if denom > 0 else gamma[i]
        norm = sum(new.values()) / len(new)
        new = {k: v / norm for k, v in new.items()}
        delta = max(abs(math.log(new[k]) - math.log(gamma[k])) for k in names)
        gamma = new
        if delta < tol:
            break
    rating = {n: 400.0 * math.log10(gamma[n]) for n in names}
    shift = anchor_rating - rating.get(anchor, 0.0)
    return {n: r + shift for n, r in rating.items()}


def wilson_interval(wins: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)
