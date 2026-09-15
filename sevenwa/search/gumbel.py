"""Gumbel root policy: Sequential Halving with Gumbel-Top-k (Danihelka et al., ICLR 2022).

Compared with visit-count targets, the *completed-Q* improved policy is a much better
training target at small simulation budgets (16–128), and action selection via
``argmax(g + logits + σ(q))`` guarantees policy improvement in expectation.  This module
adds the root procedure on top of :class:`sevenwa.search.mcts.MCTS` (which handles the
tree, chance nodes and batched evaluation).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from .mcts import MCTS, Node, SearchResult


def _sigma(q: np.ndarray, max_visits: int, c_visit: float, c_scale: float) -> np.ndarray:
    return (c_visit + max_visits) * c_scale * q


def gumbel_search(mcts: MCTS, state, num_simulations: int, max_considered: int = 16, c_visit: float = 50.0,
                  c_scale: float = 1.0, root: Optional[Node] = None) -> SearchResult:
    rng = mcts.rng
    if mcts.num_actions is None:
        mcts.num_actions = state.num_actions()
    root = root if root is not None else mcts._reusable_root(state)
    if root is None:
        root = Node(state)
    if root.is_chance or root.is_terminal:
        raise ValueError("gumbel_search requires a non-terminal decision node")
    if not root.expanded:
        mcts.expand_root(root)
    mcts.apply_prior_floor(root)  # MCTSConfig.prior_floor (0 = off): bounds the root logits from below
    sign = 1 if root.to_move == 0 else -1
    actions = list(root.priors.keys())
    logits = np.log(np.array([root.priors[a] for a in actions], dtype=np.float64) + 1e-12)
    g = rng.gumbel(size=len(actions))
    K = min(max_considered, len(actions))
    order = np.argsort(-(g + logits))
    remaining = [int(i) for i in order[:K]]

    def q_of(idx: int) -> Optional[float]:
        child = root.children.get(actions[idx])
        if child is None or child.N == 0:
            return None
        return sign * child.W / child.N

    # Sequential Halving schedule as in the paper / mctx: each phase gives every remaining action
    # floor(n / (log2(K) * |remaining|)) visits (using the TOTAL budget n), halves the set (never below
    # 2 actions) and keeps cycling until the budget is exhausted, so the last visits always compare
    # the two best candidates.
    budget = num_simulations
    phases = max(1, int(math.ceil(math.log2(max(2, K)))))

    def score(idx: int, max_n: int) -> float:
        q = q_of(idx)
        return g[idx] + logits[idx] + (_sigma(np.array([q]), max_n, c_visit, c_scale)[0] if q is not None else 0.0)

    if len(remaining) == 1 and budget > 0:
        _simulate_forced(mcts, root, actions[remaining[0]], budget)
        budget = 0
    while budget > 0:
        per = max(1, num_simulations // (phases * len(remaining)))
        per = min(per, max(1, -(-budget // len(remaining))))
        for idx in remaining:
            n = min(per, budget)
            if n <= 0:
                break
            _simulate_forced(mcts, root, actions[idx], n)
            budget -= n
        max_n = max((root.children[actions[i]].N for i in remaining if actions[i] in root.children), default=0)
        order = np.argsort(-np.array([score(i, max_n) for i in remaining]))
        keep = max(2, len(remaining) // 2) if len(remaining) > 2 else len(remaining)
        remaining = [remaining[i] for i in order[:keep]]
    max_n = max((root.children[actions[i]].N for i in remaining if actions[i] in root.children), default=0)
    chosen = actions[max(remaining, key=lambda i: score(i, max_n))]

    # completed Q-values and the improved policy over all legal actions
    max_n = max((c.N for c in root.children.values()), default=0)
    visited_q = [q_of(i) for i in range(len(actions))]
    if any(q is not None for q in visited_q):
        # value mixing: v_mix = (v_root + (ΣN/ΣN_visited) Σ π(a) q(a)) / (1 + ΣN)
        v_root = sign * (root.W / root.N) if root.N > 0 else 0.0
        pri = np.array([root.priors[a] for a in actions])
        vis = np.array([q if q is not None else 0.0 for q in visited_q])
        mask = np.array([q is not None for q in visited_q])
        n_vis = np.array([root.children[a].N if a in root.children else 0 for a in actions], dtype=np.float64)
        tot = n_vis.sum()
        w = (pri * mask).sum()
        v_mix = (v_root + (tot / max(w, 1e-12)) * (pri * mask * vis).sum()) / (1.0 + tot) if w > 0 else v_root
    else:
        v_mix = 0.0
    completed = np.array([q if q is not None else v_mix for q in visited_q])
    improved_logits = logits + _sigma(completed, max_n, c_visit, c_scale)
    improved_logits -= improved_logits.max()
    improved = np.exp(improved_logits)
    improved /= improved.sum()
    policy = np.zeros(mcts.num_actions, dtype=np.float32)
    for k, a in enumerate(actions):
        policy[a] = improved[k]
    counts = {a: (root.children[a].N if a in root.children else 0) for a in actions}
    mcts._root = root
    root_value = sign * (root.W / root.N) if root.N > 0 else 0.0
    res = SearchResult(root=root, visit_counts=counts, root_value=root_value, policy_target=policy,
                       improved_policy=policy, extra={"chosen": chosen, "q": {actions[i]: visited_q[i] for i in range(len(actions)) if visited_q[i] is not None}})
    return res


def _simulate_forced(mcts: MCTS, root: Node, action: int, n: int) -> None:
    """Run ``n`` simulations that all start with ``action`` at the root."""
    done = 0
    while done < n:
        batch = []
        want = min(mcts.cfg.batch_size, n - done)
        for _ in range(want):
            child = root.children.get(action)
            if child is None:
                child = Node(root.state.apply_action(action))
                root.children[action] = child
            leaf, path = mcts._select(root, start=child, prefix=[(root, mcts._sign_of(root))])
            if leaf.is_terminal:
                mcts._backup(path, leaf.terminal_value, remove_virtual=False)
                done += 1
                continue
            mcts._apply_virtual_loss(path)
            batch.append((leaf, path))
        if batch:
            mcts._expand_batch(batch)
            done += len(batch)
