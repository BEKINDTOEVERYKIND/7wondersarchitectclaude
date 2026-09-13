"""Chance-aware Monte-Carlo Tree Search (PUCT) with batched leaf evaluation.

Design
------
* The tree contains **decision nodes** (a player picks an action) and **chance
  nodes** (the environment reveals a card / token).  Chance nodes are expanded
  eagerly (no network call) and traversed by *sampling* an outcome from the
  engine-provided distribution, so their value estimate converges to the true
  expectation over the unseen-card multiset.  This is the "expectimax MCTS" used
  for backgammon-style games and is the correct treatment for 7 Wonders:
  Architects, whose only stochasticity is card/token reveals.
* All statistics are stored in the **absolute** frame (value for player 0).  A
  player ``p`` selects with ``sign = +1 if p == 0 else -1``.
* Leaves are collected in mini-batches with *virtual loss* and evaluated in a
  single network forward pass (:class:`Evaluator`), which is what makes the
  CPU-only pipeline fast enough.
* First-play urgency (FPU): unvisited children inherit the parent's Q minus a
  small reduction instead of a fixed 0, which behaves much better than
  AlphaZero's original choice at low simulation counts.
* Root Dirichlet noise and visit-count temperature are supported for self-play.
* An optional *Gumbel* root policy (Sequential Halving with Gumbel-Top-k, as in
  "Policy improvement by planning with Gumbel", Danihelka et al. 2022) is
  provided in :mod:`sevenwa.search.gumbel`; it gives better policy-improvement
  targets at small simulation budgets.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from ..game import CHANCE, GameState


class Evaluator(Protocol):
    """Batched state evaluator.

    ``evaluate(states)`` returns ``(policies, values)`` where ``policies`` has
    shape ``(B, num_actions)`` (probabilities; illegal entries may be anything,
    they are masked and renormalised by the search) and ``values`` has shape
    ``(B,)`` and is expressed **from the perspective of the player to move** in
    each state, in ``[-1, 1]``.
    """

    def evaluate(self, states: Sequence[GameState]) -> Tuple[np.ndarray, np.ndarray]: ...


@dataclass
class MCTSConfig:
    num_simulations: int = 200
    c_puct: float = 1.75
    c_puct_base: float = 19652.0  # AlphaZero's log-growth of exploration
    c_puct_init: float = 1.25
    use_puct_schedule: bool = False  # if True use c_puct_init/base schedule instead of c_puct
    fpu_reduction: float = 0.25  # first-play urgency reduction (in Q units)
    fpu_root_reduction: float = 0.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    add_root_noise: bool = False
    batch_size: int = 8  # leaves evaluated per network call
    virtual_loss: float = 1.0
    max_chance_depth: int = 64  # guard against pathological chance chains
    value_scale: float = 1.0
    reuse_tree: bool = True


class Node:
    """A search-tree node.  ``children`` are keyed by action (decision) or outcome (chance)."""

    __slots__ = (
        "state", "to_move", "is_chance", "is_terminal", "children", "priors",
        "outcome_ids", "outcome_probs", "N", "W", "vN", "vW", "expanded", "terminal_value",
    )

    def __init__(self, state: GameState):
        self.state = state
        self.to_move = state.to_move()
        self.is_chance = self.to_move == CHANCE
        self.is_terminal = state.is_terminal()
        self.children: Dict[int, "Node"] = {}
        self.priors: Dict[int, float] = {}
        self.outcome_ids: Optional[np.ndarray] = None
        self.outcome_probs: Optional[np.ndarray] = None
        self.N: int = 0
        self.W: float = 0.0  # sum of absolute (player-0 frame) values
        self.vN: int = 0  # virtual-loss visits in flight
        self.vW: float = 0.0
        self.expanded = False
        self.terminal_value: Optional[float] = None
        if self.is_terminal:
            self.terminal_value = float(state.returns()[0])
            self.expanded = True
        elif self.is_chance:
            outs = state.chance_outcomes()
            self.outcome_ids = np.fromiter((o for o, _ in outs), dtype=np.int64, count=len(outs))
            probs = np.fromiter((p for _, p in outs), dtype=np.float64, count=len(outs))
            self.outcome_probs = probs / probs.sum()
            self.expanded = True

    # absolute-frame mean value including virtual losses
    def q_abs(self) -> float:
        n = self.N + self.vN
        return (self.W + self.vW) / n if n > 0 else 0.0

    def visits(self) -> int:
        return self.N


@dataclass
class SearchResult:
    root: Node
    visit_counts: Dict[int, int]
    root_value: float  # from the perspective of the player to move at the root
    policy_target: np.ndarray  # (num_actions,) normalised visit distribution
    improved_policy: Optional[np.ndarray] = None  # Gumbel completed-Q policy if computed
    extra: dict = field(default_factory=dict)

    def best_action(self) -> int:
        return max(self.visit_counts.items(), key=lambda kv: kv[1])[0]

    def sample_action(self, rng: np.random.Generator, temperature: float = 1.0) -> int:
        actions = list(self.visit_counts.keys())
        counts = np.array([self.visit_counts[a] for a in actions], dtype=np.float64)
        if temperature <= 1e-6:
            return actions[int(np.argmax(counts))]
        logits = np.log(counts + 1e-12) / temperature
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return actions[int(rng.choice(len(actions), p=probs))]


class MCTS:
    """PUCT search over a :class:`GameState` tree with chance nodes."""

    def __init__(self, evaluator: Evaluator, config: Optional[MCTSConfig] = None,
                 rng: Optional[np.random.Generator] = None, num_actions: Optional[int] = None):
        self.evaluator = evaluator
        self.cfg = config or MCTSConfig()
        self.rng = rng or np.random.default_rng()
        self.num_actions = num_actions
        self._root: Optional[Node] = None

    # ------------------------------------------------------------------ public
    def run(self, state: GameState, add_noise: Optional[bool] = None,
            num_simulations: Optional[int] = None, root: Optional[Node] = None) -> SearchResult:
        cfg = self.cfg
        n_sims = cfg.num_simulations if num_simulations is None else num_simulations
        add_noise = cfg.add_root_noise if add_noise is None else add_noise
        if self.num_actions is None:
            self.num_actions = state.num_actions()

        root = root if root is not None else self._reusable_root(state)
        if root is None:
            root = Node(state)
        if root.is_chance or root.is_terminal:
            raise ValueError("MCTS.run requires a non-terminal decision node at the root")
        if not root.expanded:
            self._expand_batch([(root, [(root, 1)])])
        if add_noise:
            self._add_dirichlet_noise(root)

        sims_done = 0
        while sims_done < n_sims:
            batch: List[Tuple[Node, List[Tuple[Node, int]]]] = []
            want = min(cfg.batch_size, n_sims - sims_done)
            for _ in range(want):
                leaf, path = self._select(root)
                if leaf.is_terminal:
                    self._backup(path, leaf.terminal_value, remove_virtual=False)
                    sims_done += 1
                    continue
                self._apply_virtual_loss(path)
                batch.append((leaf, path))
            if batch:
                self._expand_batch(batch)
                sims_done += len(batch)

        self._root = root
        return self._result(root)

    def advance(self, action_or_outcome: int) -> None:
        """Move the retained root down one edge (for tree reuse between moves)."""
        if self._root is None or not self.cfg.reuse_tree:
            self._root = None
            return
        child = self._root.children.get(action_or_outcome)
        self._root = child if child is not None else None

    def reset(self) -> None:
        self._root = None

    # ------------------------------------------------------------ internals
    def _reusable_root(self, state: GameState) -> Optional[Node]:
        if not self.cfg.reuse_tree or self._root is None:
            return None
        node = self._root
        # Walk through chance nodes that the environment already resolved: we cannot
        # know which outcome happened, so match by observable key.
        target = state.key()
        frontier = [node]
        for _ in range(self.cfg.max_chance_depth):
            nxt = []
            for n in frontier:
                if n.state.key() == target and not n.is_chance:
                    return n
                if n.is_chance:
                    nxt.extend(n.children.values())
            if not nxt:
                break
            frontier = nxt
        return None

    def _c_puct(self, parent_n: int) -> float:
        cfg = self.cfg
        if cfg.use_puct_schedule:
            return cfg.c_puct_init + math.log((parent_n + cfg.c_puct_base + 1) / cfg.c_puct_base)
        return cfg.c_puct

    def _select(self, root: Node) -> Tuple[Node, List[Tuple[Node, int]]]:
        """Descend from the root to a leaf; returns the leaf and the path of (node, sign)."""
        node = root
        path: List[Tuple[Node, int]] = [(root, 1 if root.to_move == 0 else -1)]
        depth_guard = 0
        while True:
            if node.is_terminal:
                return node, path
            if node.is_chance:
                depth_guard += 1
                if depth_guard > self.cfg.max_chance_depth:
                    raise RuntimeError("chance chain too deep; engine bug?")
                idx = int(self.rng.choice(len(node.outcome_ids), p=node.outcome_probs))
                outcome = int(node.outcome_ids[idx])
                child = node.children.get(outcome)
                if child is None:
                    child = Node(node.state.apply_chance(outcome))
                    node.children[outcome] = child
                node = child
                path.append((node, 0))  # sign irrelevant at chance nodes
                continue
            if not node.expanded:
                return node, path
            action = self._select_action(node, is_root=node is root)
            child = node.children.get(action)
            if child is None:
                child = Node(node.state.apply_action(action))
                node.children[action] = child
            node = child
            path.append((node, 1 if node.to_move == 0 else (-1 if node.to_move == 1 else 0)))

    def _select_action(self, node: Node, is_root: bool) -> int:
        sign = 1 if node.to_move == 0 else -1
        n_parent = node.N + node.vN
        sqrt_n = math.sqrt(max(1, n_parent))
        c = self._c_puct(n_parent)
        parent_q = sign * node.q_abs()
        fpu = parent_q - (self.cfg.fpu_root_reduction if is_root else self.cfg.fpu_reduction)
        best_score = -1e18
        best_action = -1
        for action, prior in node.priors.items():
            child = node.children.get(action)
            if child is None or (child.N + child.vN) == 0:
                q = fpu
                n_child = 0
            else:
                q = sign * child.q_abs()
                n_child = child.N + child.vN
            score = q + c * prior * sqrt_n / (1 + n_child)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _apply_virtual_loss(self, path: List[Tuple[Node, int]]) -> None:
        vl = self.cfg.virtual_loss
        for node, _ in path:
            node.vN += 1
        # A virtual "loss" for whoever moves at the parent: emulate by pushing the
        # absolute value towards the opponent of the mover at each node's parent.
        for i, (node, _) in enumerate(path):
            if i == 0:
                continue
            parent_sign = path[i - 1][1]
            if parent_sign == 0:  # parent is a chance node: use the nearest decision ancestor
                j = i - 1
                while j >= 0 and path[j][1] == 0:
                    j -= 1
                parent_sign = path[j][1] if j >= 0 else 1
            node.vW += -parent_sign * vl
        # root itself
        path[0][0].vW += 0.0

    def _remove_virtual_loss(self, path: List[Tuple[Node, int]]) -> None:
        vl = self.cfg.virtual_loss
        for i, (node, _) in enumerate(path):
            node.vN -= 1
            if i == 0:
                continue
            parent_sign = path[i - 1][1]
            if parent_sign == 0:
                j = i - 1
                while j >= 0 and path[j][1] == 0:
                    j -= 1
                parent_sign = path[j][1] if j >= 0 else 1
            node.vW -= -parent_sign * vl

    def _backup(self, path: List[Tuple[Node, int]], value_abs: float, remove_virtual: bool) -> None:
        if remove_virtual:
            self._remove_virtual_loss(path)
        for node, _ in path:
            node.N += 1
            node.W += value_abs

    def _expand_batch(self, batch: List[Tuple[Node, List[Tuple[Node, int]]]]) -> None:
        states = [leaf.state for leaf, _ in batch]
        policies, values = self.evaluator.evaluate(states)
        for i, (leaf, path) in enumerate(batch):
            if not leaf.expanded:  # a leaf may be reached twice within a batch
                legal = list(leaf.state.legal_actions())
                p = np.asarray(policies[i], dtype=np.float64)
                pl = np.array([max(p[a], 0.0) for a in legal], dtype=np.float64)
                s = pl.sum()
                if not np.isfinite(s) or s <= 1e-12:
                    pl = np.full(len(legal), 1.0 / len(legal))
                else:
                    pl /= s
                leaf.priors = {a: float(pl[k]) for k, a in enumerate(legal)}
                leaf.expanded = True
            v_mover = float(values[i]) * self.cfg.value_scale
            sign = 1 if leaf.to_move == 0 else -1
            self._backup(path, sign * v_mover, remove_virtual=True)

    def _add_dirichlet_noise(self, root: Node) -> None:
        cfg = self.cfg
        actions = list(root.priors.keys())
        if not actions:
            return
        noise = self.rng.dirichlet([cfg.dirichlet_alpha] * len(actions))
        for k, a in enumerate(actions):
            root.priors[a] = (1 - cfg.dirichlet_epsilon) * root.priors[a] + cfg.dirichlet_epsilon * float(noise[k])

    def _result(self, root: Node) -> SearchResult:
        counts = {a: (root.children[a].N if a in root.children else 0) for a in root.priors}
        total = sum(counts.values())
        policy = np.zeros(self.num_actions, dtype=np.float32)
        if total > 0:
            for a, n in counts.items():
                policy[a] = n / total
        else:
            for a, p in root.priors.items():
                policy[a] = p
        sign = 1 if root.to_move == 0 else -1
        root_value = sign * (root.W / root.N if root.N > 0 else 0.0)
        # Q-values per action from the mover's perspective (useful for diagnostics / Gumbel)
        q = {a: (sign * root.children[a].W / root.children[a].N) for a in counts if a in root.children and root.children[a].N > 0}
        return SearchResult(root=root, visit_counts=counts, root_value=root_value,
                            policy_target=policy, extra={"q": q})
