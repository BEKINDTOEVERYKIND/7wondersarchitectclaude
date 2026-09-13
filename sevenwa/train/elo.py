"""Elo estimation from pairwise results (Bradley–Terry via simple iterative MM updates)."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, Tuple


def expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def fit_elo(results: Iterable[Tuple[str, str, float]], anchor: str = "random", anchor_rating: float = 0.0,
            iterations: int = 500) -> Dict[str, float]:
    """``results`` are ``(name_a, name_b, score_a)`` with score in {0, 0.5, 1}.

    Ratings are fitted by minimising the logistic log-loss with a fixed anchor.
    """
    games = defaultdict(lambda: [0.0, 0.0])  # (a,b) -> [score_a_sum, count]
    names = set()
    for a, b, s in results:
        names.add(a); names.add(b)
        games[(a, b)][0] += s; games[(a, b)][1] += 1
    rating = {n: 0.0 for n in names}
    if anchor in rating:
        rating[anchor] = anchor_rating
    lr = 20.0
    for _ in range(iterations):
        grad = defaultdict(float)
        for (a, b), (sa, n) in games.items():
            e = expected(rating[a], rating[b]) * n
            grad[a] += sa - e
            grad[b] -= sa - e
        for nme in rating:
            if nme == anchor:
                continue
            rating[nme] += lr * grad[nme] / max(1.0, sum(c for (x, y), (_, c) in games.items() if nme in (x, y)))
        lr = max(1.0, lr * 0.995)
    return rating


def wilson_interval(wins: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / denom, (centre + half) / denom)
