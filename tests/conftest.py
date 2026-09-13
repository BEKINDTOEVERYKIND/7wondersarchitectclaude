"""Shared helpers for the rules-fidelity test-suite (``test_engine.py`` / ``test_env.py``).

Two ways of building states are used throughout the suite:

1. **Public API only.**  :func:`first_pick` resolves the initial chance nodes of
   ``GameState.initial`` with *chosen* outcomes (the two visible deck tops and the three face-up
   Progress tokens), which gives a deterministic state at player 0's first main pick.  Further
   play goes through ``apply_action`` / ``apply_chance`` (see :func:`resolve`, :func:`settle`).

2. **Scenario editing.**  For targeted rule tests a *copy* of a state is edited directly
   (tableaus, deck tops, stages, tokens, conflict...).  Every helper below *moves* cards and
   tokens between the places the engine tracks (tableaus, deck unseen multisets, deck tops, the
   discard pile, the token stack) and never creates them, so the conservation invariants checked
   by :func:`assert_invariants` stay valid for scenario states too.  :func:`run` then re-enters
   the engine's work loop with an explicit queue (for example ``("check",), ("end_turn",)`` to
   trigger the mandatory-construction / science check of the mover, or ``("turn_start",)`` for a
   full turn boundary) and stops at the next real decision / chance node, exactly like the engine
   does after ``apply_action``.  :func:`take_card` is the most common entry point: it simulates
   "the mover has just taken card *k* from deck *d*" with all of its consequences.

Nothing in here modifies the engine; private fields are only read/edited on copies.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from sevenwa.engine import Actions as A
from sevenwa.engine import GameState, RulesConfig
from sevenwa.engine.cards import KINDS, NUM_KINDS, central_deck_counts, wonder_deck_counts
from sevenwa.engine.env import Environment
from sevenwa.engine.state import (C_DRAW_CENTRAL, C_HALI_REVEAL, C_PEEK, C_REVEAL, C_TOKEN_REVEAL, CENTRAL, D_HALI_CHOOSE,
                                  D_PAY, D_PICK, N_CHANCE, N_DECISION, N_TERMINAL, PAY_COIN2_CODE, PAY_COIN_CODE)
from sevenwa.engine.tokens import TOKENS, TOTAL_TOKEN_COPIES
from sevenwa.engine.wonders import WONDERS
from sevenwa.game import CHANCE

# --------------------------------------------------------------------------- name spaces
#: card kind ids by name: ``K.wood``, ``K.coin``, ``K.civ2cat``, ``K.shield_h2`` ...
K = SimpleNamespace(**{k.name: k.id for k in KINDS})
#: Progress-token type ids by name: ``T.Urbanism``, ``T.Culture`` ...
T = SimpleNamespace(**{t.name: t.id for t in TOKENS})
#: Wonder ids by name: ``W.Giza``, ``W.Rhodes`` ...
W = SimpleNamespace(**{w.name: w.id for w in WONDERS})

#: Face-up tokens used by default: end-game VP tokens, which never influence play.
DEFAULT_FACEUP: Tuple[int, int, int] = (T.Strategy, T.Education, T.Decor)
GREEN_KINDS = (K.tablet, K.gear, K.compass)


# --------------------------------------------------------------------------- public-API drivers
def first_pick(w0: int = W.Giza, w1: int = W.Rhodes, rules: Optional[RulesConfig] = None,
               tops: Tuple[int, int] = (K.wood, K.stone), faceup: Sequence[int] = DEFAULT_FACEUP) -> GameState:
    """Deterministic state at player 0's first main pick, built only through the public API.

    The initial chance nodes are resolved in the engine's order: reveal of deck 0, reveal of
    deck 1, then the face-up Progress tokens.
    """
    s = GameState.initial((w0, w1), rules)
    assert s.is_chance() and s.ckind == C_REVEAL and s.cctx == 0
    s = s.apply_chance(tops[0])
    assert s.is_chance() and s.ckind == C_REVEAL and s.cctx == 1
    s = s.apply_chance(tops[1])
    for t in faceup:
        assert s.is_chance() and s.ckind == C_TOKEN_REVEAL, s.describe_node()
        s = s.apply_chance(t)
    assert s.to_move() == 0 and s.dkind == D_PICK and s.dctx[0] == "main", s.describe_node()
    return s


def resolve(s: GameState, *outcomes: int) -> GameState:
    """Apply the given chance outcomes in order (each must be at a chance node)."""
    for o in outcomes:
        assert s.is_chance(), f"expected a chance node, got {s.describe_node()}"
        assert o in dict(s.chance_outcomes()), f"outcome {o} impossible at {s.describe_node()}"
        s = s.apply_chance(o)
    return s


def most_likely(s: GameState) -> int:
    """Deterministic default outcome: highest probability, ties broken by the lowest id."""
    outs = s.chance_outcomes()
    best = max(p for _, p in outs)
    return min(o for o, p in outs if p == best)


def settle(s: GameState, choose: Optional[Callable[[GameState], int]] = None,
           trace: Optional[List[GameState]] = None) -> GameState:
    """Resolve chance nodes (with ``choose`` or :func:`most_likely`) until a decision/terminal."""
    while s.is_chance():
        if trace is not None:
            trace.append(s)
        s = s.apply_chance((choose or most_likely)(s))
    return s


def pay_through(s: GameState, choose: Optional[Callable[[Sequence[int]], int]] = None) -> Tuple[GameState, List[int]]:
    """Follow consecutive payment decisions; returns the state after them and the actions taken."""
    taken: List[int] = []
    while s.to_move() != CHANCE and not s.is_terminal() and s.dkind == D_PAY:
        a = (choose or (lambda legal: legal[0]))(list(s.legal_actions()))
        taken.append(a)
        s = s.apply_action(a)
    return s, taken


def payment_leaves(s: GameState) -> List[GameState]:
    """States at the end of *every* payment branch reachable from ``s`` (depth-first).

    Payments are sequences of codes in canonical (non-decreasing) order, so each leaf corresponds
    to one multiset of cards (or to a dead end, if the engine offered a code that cannot be
    completed).
    """
    if s.to_move() != CHANCE and not s.is_terminal() and s.dkind == D_PAY:
        out: List[GameState] = []
        for a in s.legal_actions():
            out.extend(payment_leaves(s.apply_action(a)))
        return out
    return [s]


def is_pick(s: GameState, reason: Optional[str] = None, player: Optional[int] = None) -> bool:
    """True if ``s`` is a card-pick decision (optionally with the given reason / player)."""
    if s.node_type != N_DECISION or s.dkind != D_PICK:
        return False
    if reason is not None and s.dctx[0] != reason:
        return False
    return player is None or s.mover == player


def is_main_pick_of(s: GameState, player: int) -> bool:
    return is_pick(s, "main", player)


# --------------------------------------------------------------------------- scenario editing
def edit(s: GameState) -> GameState:
    """A private copy that may be edited freely before :func:`run` (or stepped as-is)."""
    c = s._copy()
    c._legal = s._legal
    c._outcomes = s._outcomes
    return c


def _take_from_unseen(s: GameState, d: int, kind: int, n: int = 1) -> None:
    assert s.unseen[d][kind] >= n, f"deck {d} has only {s.unseen[d][kind]} unseen {KINDS[kind].name}"
    s.unseen[d][kind] -= n
    s.deck_size[d] -= n


def give(s: GameState, p: int, kind: int, n: int = 1, frm: int = CENTRAL) -> GameState:
    """Move ``n`` unseen cards of ``kind`` from deck ``frm`` into player ``p``'s tableau."""
    _take_from_unseen(s, frm, kind, n)
    s.cards[p][kind] += n
    return s


def give_many(s: GameState, p: int, kinds: Iterable[int], frm: int = CENTRAL) -> GameState:
    for k in kinds:
        give(s, p, k, 1, frm)
    return s


def set_top(s: GameState, d: int, kind: int) -> GameState:
    """Make ``kind`` the visible top card of deck ``d`` (the previous top returns to unseen)."""
    if s.deck_top[d] >= 0:  # the old top goes back among the unseen cards (deck size unchanged)
        s.unseen[d][s.deck_top[d]] += 1
        s.deck_top[d] = -1
    _take_from_unseen(s, d, kind)
    s.deck_top[d] = kind
    s.deck_size[d] += 1
    return s


def drain_deck(s: GameState, d: int) -> GameState:
    """Empty deck ``d`` (top + unseen go to the discard pile, which is inert)."""
    if s.deck_top[d] >= 0:
        s.discard[s.deck_top[d]] += 1
        s.deck_top[d] = -1
    for k in range(NUM_KINDS):
        s.discard[k] += s.unseen[d][k]
        s.unseen[d][k] = 0
    s.deck_size[d] = 0
    if d == CENTRAL:
        s.central_known_to = 0
    return s


def _pull_from_anywhere(s: GameState, kind: int, exclude: int = -1) -> None:
    """Remove one card of ``kind`` from the discard pile, or else from another deck's unseen cards."""
    if s.discard[kind] > 0:
        s.discard[kind] -= 1
        return
    for d in range(3):
        if d != exclude and s.unseen[d][kind] > 0:
            _take_from_unseen(s, d, kind)
            return
    raise AssertionError(f"no spare {KINDS[kind].name} anywhere")


def set_deck(s: GameState, d: int, top: Optional[int], unseen: Sequence[int] = ()) -> GameState:
    """Give deck ``d`` exactly the composition ``[top] + unseen`` (``top=None`` for a hidden top)."""
    drain_deck(s, d)
    for k in unseen:
        _pull_from_anywhere(s, k, exclude=d)
        s.unseen[d][k] += 1
        s.deck_size[d] += 1
    if top is not None:
        _pull_from_anywhere(s, top, exclude=d)
        if d == CENTRAL:  # the central top is only ever known through a peek
            s.unseen[d][top] += 1
        else:
            s.deck_top[d] = top
        s.deck_size[d] += 1
    return s


def give_token(s: GameState, p: int, tid: int, n: int = 1) -> GameState:
    """Give player ``p`` ``n`` copies of token ``tid`` (from the stack, else from the face-up row)."""
    for _ in range(n):
        if s.prog_unseen[tid] > 0:
            s.prog_unseen[tid] -= 1
            s.prog_stack -= 1
        else:
            assert tid in s.faceup, f"token {TOKENS[tid].name} unavailable"
            s.faceup.remove(tid)
        s.tokens[p][tid] += 1
    return s


def set_faceup(s: GameState, tids: Sequence[int]) -> GameState:
    """Replace the face-up token row (the old row goes back into the stack)."""
    for t in s.faceup:
        s.prog_unseen[t] += 1
        s.prog_stack += 1
    s.faceup = []
    for t in tids:
        assert s.prog_unseen[t] > 0
        s.prog_unseen[t] -= 1
        s.prog_stack -= 1
        s.faceup.append(t)
    return s


def exhaust_tokens(s: GameState, to: int = 1) -> GameState:
    """Move every remaining Progress token (face-up + stack) into player ``to``'s collection."""
    for t in s.faceup:
        s.tokens[to][t] += 1
    s.faceup = []
    for t in range(len(s.prog_unseen)):
        s.tokens[to][t] += s.prog_unseen[t]
        s.prog_unseen[t] = 0
    s.prog_stack = 0
    return s


def run(s: GameState, *queue) -> GameState:
    """Re-enter the engine's work loop on a copy of ``s`` with the given queue items.

    The queue should end with ``("end_turn",)`` (or be ``("turn_start",)``) so that the game can
    continue after the tested items; ``_end_turn`` schedules the next ``turn_start`` itself.
    """
    c = s._copy()
    c.queue = tuple(queue)
    c.node_type = -1
    c._legal = None
    c._outcomes = None
    c._run()
    return c


def take_card(s: GameState, kind: int, d: int = CENTRAL, reason: str = "main", p: Optional[int] = None) -> GameState:
    """Simulate "the mover has just taken card ``kind`` from deck ``d``" (card moved out of the
    deck's unseen cards, then the engine's ``placed`` handler runs followed by ``end_turn``)."""
    c = edit(s)
    if p is not None:
        c.mover = p
    _take_from_unseen(c, d, kind)
    return run(c, ("placed", kind, d, reason), ("end_turn",))


def check(s: GameState, p: Optional[int] = None) -> GameState:
    """Run the mandatory construction / science check of the mover (or of ``p``), then end the turn."""
    if p is not None:
        s = edit(s)
        s.mover = p
    return run(s, ("check",), ("end_turn",))


# --------------------------------------------------------------------------- invariants
def initial_composition(s: GameState) -> List[int]:
    comp = [0] * NUM_KINDS
    for v in (wonder_deck_counts(WONDERS[s.wonder[0]].name), wonder_deck_counts(WONDERS[s.wonder[1]].name),
              central_deck_counts()):
        for k in range(NUM_KINDS):
            comp[k] += v[k]
    return comp


def hali_revealed(s: GameState) -> Tuple[int, ...]:
    """Cards currently lifted off a deck by Halicarnassus (in neither deck nor tableau)."""
    if s.node_type == N_CHANCE and s.ckind == C_HALI_REVEAL:
        return tuple(s.cctx[2])
    if s.node_type == N_DECISION and s.dkind == D_HALI_CHOOSE:
        return tuple(s.dctx[1])
    for item in s.queue:
        if item[0] == "hali_reveal":
            return tuple(item[3])
    return ()


def hali_deck(s: GameState) -> int:
    if s.node_type == N_CHANCE and s.ckind == C_HALI_REVEAL:
        return s.cctx[0]
    if s.node_type == N_DECISION and s.dkind == D_HALI_CHOOSE:
        return s.dctx[0]
    for item in s.queue:
        if item[0] == "hali_reveal":
            return item[1]
    return -1


def in_transit(s: GameState) -> List[int]:
    """Kinds of cards taken from a deck but not yet placed (a ``placed`` item queued behind a
    reveal chance node)."""
    return [item[1] for item in s.queue if item[0] == "placed"]


def card_census(s: GameState) -> List[int]:
    """Every card the belief state accounts for, per kind."""
    c = [0] * NUM_KINDS
    for p in (0, 1):
        for k in range(NUM_KINDS):
            c[k] += s.cards[p][k]
    for d in range(3):
        for k in range(NUM_KINDS):
            c[k] += s.unseen[d][k]
        if s.deck_top[d] >= 0:
            c[s.deck_top[d]] += 1
    for k in range(NUM_KINDS):
        c[k] += s.discard[k]
    for k in hali_revealed(s):
        c[k] += 1
    for k in in_transit(s):
        c[k] += 1
    return c


def assert_cards_conserved(s: GameState, comp: Optional[List[int]] = None) -> None:
    comp = comp or initial_composition(s)
    census = card_census(s)
    assert census == comp, f"card conservation broken: {census} != {comp}\n{s.describe()}"


def assert_tokens_conserved(s: GameState) -> None:
    owned = sum(s.tokens[0]) + sum(s.tokens[1])
    assert owned + len(s.faceup) + sum(s.prog_unseen) == TOTAL_TOKEN_COPIES, s.describe()
    assert s.prog_stack == sum(s.prog_unseen)
    assert all(c >= 0 for c in s.prog_unseen)
    assert len(s.faceup) <= s.rules.faceup_progress_tokens


def assert_deck_sizes(s: GameState) -> None:
    rev_d, rev = hali_deck(s), hali_revealed(s)
    for d in range(3):
        expect = sum(s.unseen[d]) + (1 if s.deck_top[d] >= 0 else 0) + (len(rev) if d == rev_d else 0)
        assert s.deck_size[d] == expect, f"deck {d}: size {s.deck_size[d]} != {expect}\n{s.describe()}"
        assert all(c >= 0 for c in s.unseen[d])
    # the central top is only ever known through a Cat peek (``central_known_to`` is a bitmask)
    assert (s.deck_top[CENTRAL] >= 0) == (s.central_known_to != 0)
    assert 0 <= s.central_known_to <= 3
    for d in (0, 1):  # wonder decks are face up: the top is visible unless a reveal is pending
        if s.deck_size[d] > 0 and s.deck_top[d] < 0:
            pending = (s.node_type == N_CHANCE and s.ckind == C_REVEAL and s.cctx == d) or \
                      any(item[0] == "reveal" and item[1] == d for item in s.queue) or d == rev_d
            assert pending, f"deck {d} has cards but no visible top\n{s.describe()}"


def assert_node_sane(s: GameState) -> None:
    """Structural checks of the exposed node (real choices only, normalised chance)."""
    if s.node_type == N_DECISION:
        legal = list(s.legal_actions())
        assert len(legal) > 1, f"decision node with a single option: {s.describe_node()}"
        assert len(set(legal)) == len(legal)
        assert all(0 <= a < A.NUM for a in legal)
        assert s.to_move() in (0, 1) and not s.is_chance() and not s.is_terminal()
        assert s.chance_outcomes() == ()
        if s.dkind == D_PAY:
            cost = s._stage().cost
            value = s._pay_state(tuple(s.dctx))[2]
            assert value < cost, "payment decision after the cost was already covered"
            for a in legal:
                inc = 2 if a == A.PAY_COIN2 else 1
                assert value + inc <= cost, f"option {A.name(a)} would overpay ({value}+{inc} > {cost})"
    elif s.node_type == N_CHANCE:
        outs = list(s.chance_outcomes())
        assert len(outs) > 1, f"chance node with a single outcome: {s.describe_node()}"
        assert abs(sum(p for _, p in outs) - 1.0) < 1e-12
        assert all(p > 0 for _, p in outs)
        assert len({o for o, _ in outs}) == len(outs)
        assert s.to_move() == CHANCE and s.legal_actions() == ()
        if s.ckind in (C_REVEAL, C_HALI_REVEAL):
            d = s.cctx if s.ckind == C_REVEAL else s.cctx[0]
            src = s.unseen[d]
        elif s.ckind in (C_DRAW_CENTRAL, C_PEEK):
            src = s.unseen[CENTRAL]
        else:
            src = s.prog_unseen
        total = sum(src)
        for o, p in outs:
            assert abs(p - src[o] / total) < 1e-12, "chance probability not proportional to the unseen counts"
        assert {o for o, _ in outs} == {i for i, c in enumerate(src) if c > 0}
    else:
        assert s.node_type == N_TERMINAL and s.is_terminal() and s.to_move() == -2
        assert s.legal_actions() == () and s.chance_outcomes() == ()


def assert_invariants(s: GameState, comp: Optional[List[int]] = None) -> None:
    assert_cards_conserved(s, comp)
    assert_tokens_conserved(s)
    assert_deck_sizes(s)
    assert_node_sane(s)
    assert 0 <= s.conflict <= s.rules.conflict_tokens
    assert all(0 <= x <= 5 for x in s.stages)
    assert all(c >= 0 for p in (0, 1) for c in s.cards[p])
    assert all(c >= 0 for c in s.discard)


# --------------------------------------------------------------------------- random play
def random_game(seed: int, wonders: Optional[Tuple[int, int]] = None, rules: Optional[RulesConfig] = None,
                on_step: Optional[Callable[[GameState, GameState], None]] = None, max_plies: int = 5000) -> GameState:
    """Play a whole game on the belief state (chance nodes sampled from their distribution)."""
    rng = np.random.default_rng(seed)
    if wonders is None:
        w = rng.choice(len(WONDERS), size=2, replace=False)
        wonders = (int(w[0]), int(w[1]))
    s = GameState.initial(wonders, rules)
    for _ in range(max_plies):
        if s.is_terminal():
            return s
        if s.is_chance():
            outs = s.chance_outcomes()
            ids = [o for o, _ in outs]
            probs = np.array([p for _, p in outs])
            nxt = s.apply_chance(int(rng.choice(ids, p=probs / probs.sum())))
        else:
            legal = list(s.legal_actions())
            nxt = s.apply_action(int(rng.choice(legal)))
        if on_step is not None:
            on_step(s, nxt)
        s = nxt
    raise AssertionError("game did not terminate")


def random_env_game(seed: int, wonders: Optional[Tuple[int, int]] = None, rules: Optional[RulesConfig] = None,
                    on_step: Optional[Callable[[Environment, int], None]] = None, max_plies: int = 5000) -> Environment:
    """Play a whole game in the Environment with uniformly random legal actions."""
    env = Environment(wonders=wonders, rules=rules, seed=seed)
    rng = np.random.default_rng(seed + 1_000_003)
    for _ in range(max_plies):
        if env.is_terminal():
            return env
        obs = env.observe(env.to_move())
        a = int(rng.choice(list(obs.legal_actions())))
        if on_step is not None:
            on_step(env, a)
        env.step(a)
    raise AssertionError("game did not terminate")


def sync_env_to_state(env: Environment, s: GameState, seed: int = 0) -> None:
    """Install belief state ``s`` into ``env`` with physical decks/token stack consistent with it."""
    rng = np.random.default_rng(seed)
    decks: List[List[int]] = []
    for d in range(3):
        rest = [k for k in range(NUM_KINDS) for _ in range(s.unseen[d][k])]
        rng.shuffle(rest)
        decks.append(([s.deck_top[d]] if s.deck_top[d] >= 0 else []) + [int(x) for x in rest])
        assert len(decks[d]) == s.deck_size[d]
    env.decks = decks
    stack = [t for t in range(len(s.prog_unseen)) for _ in range(s.prog_unseen[t])]
    rng.shuffle(stack)
    env.token_stack = [int(x) for x in stack]
    env.state = s
    env.rules = s.rules
    env.wonders = s.wonder


def physical_matches_belief(env: Environment) -> None:
    """The true decks / token stack must be permutations of what the belief says is there."""
    s = env.state
    for d in range(3):
        deck = env.decks[d]
        assert len(deck) == s.deck_size[d], f"deck {d}: physical {len(deck)} vs belief {s.deck_size[d]}"
        counts = [0] * NUM_KINDS
        for k in deck:
            counts[k] += 1
        expect = list(s.unseen[d])
        if s.deck_top[d] >= 0:
            expect[s.deck_top[d]] += 1
            assert deck[0] == s.deck_top[d], f"deck {d}: visible top {s.deck_top[d]} is not the physical top {deck[0]}"
        for k in hali_revealed(s) if hali_deck(s) == d else ():
            expect[k] += 1
        assert counts == expect, f"deck {d}: physical multiset {counts} != belief {expect}"
    stack = [0] * len(s.prog_unseen)
    for t in env.token_stack:
        stack[t] += 1
    assert stack == list(s.prog_unseen)


# --------------------------------------------------------------------------- fixtures
@pytest.fixture
def base() -> GameState:
    """Giza (P0) vs Rhodes (P1), tops wood / stone, face-up VP tokens, player 0 to pick."""
    return first_pick()


PAY_CODES = {"coin": PAY_COIN_CODE, "coin2": PAY_COIN2_CODE}
