"""Tests for the hand-crafted expert policy (``sevenwa.agents.heuristic``)."""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

from sevenwa.agents.base import Agent
from sevenwa.agents.heuristic import (HeuristicAgent, HeuristicParams, heuristic_action, heuristic_prior,
                                      score_actions)
from sevenwa.agents.random_agent import RandomAgent
from sevenwa.engine.actions import Actions as A
from sevenwa.engine.cards import KIND_BY_NAME
from sevenwa.engine.env import make_env
from sevenwa.engine.state import (CENTRAL, D_HALI_CHOOSE, D_HALI_DECK, D_PAY, D_PICK, D_SCIENCE, D_TOKEN,
                                  DECISION_NAMES, GameState)
from sevenwa.engine.tokens import TOKEN_BY_NAME
from sevenwa.engine.wonders import WONDER_BY_NAME

from conftest import first_n  # noqa: E402  (tests/conftest.py)
from sevenwa.train.arena import play_match

K = KIND_BY_NAME
GIZA = WONDER_BY_NAME["Giza"].id
RHODES = WONDER_BY_NAME["Rhodes"].id
HALI = WONDER_BY_NAME["Halicarnassus"].id


# --------------------------------------------------------------------------- helpers
def _collect_states(num_games: int, seed: int, wonders=None, mix: str = "mixed"):
    """Play games and return every decision state seen (observer = player to move)."""
    rng = np.random.default_rng(seed)
    states = []
    for g in range(num_games):
        env = make_env(seed=seed * 1000 + g, wonders=wonders)
        while not env.is_terminal():
            p = env.to_move()
            obs = env.observe(p)
            states.append(obs)
            legal = obs.legal_actions()
            if mix == "random" or (mix == "mixed" and (g + p) % 2 == 0):
                a = int(legal[rng.integers(len(legal))])
            else:
                a = heuristic_action(obs, rng)
            env.step(a)
    return states


def _p0_with_cards(counts, stage: int = 0, tokens=()):
    """Synthetic state: player 0 (Giza, at the given stage) holds the given cards (and Progress
    tokens, by name) at the first decision, and the mandatory checks (science sets /
    construction) have been re-run.

    Cards are moved from the central deck into the tableau and tokens out of the face-up row /
    stack so the state stays consistent.
    """
    s = GameState.initial((GIZA, RHODES))
    # resolve chance nodes with fixed outcomes until the first decision
    while s.is_chance():
        s = s.apply_chance(s.chance_outcomes()[0][0])
    assert s.to_move() == 0 and s.dkind == D_PICK
    t = s._copy()
    t.built[0] = first_n(stage)
    for name, c in counts.items():
        k = K[name].id
        assert t.unseen[CENTRAL][k] >= c, name
        t.cards[0][k] += c
        t.unseen[CENTRAL][k] -= c
        t.deck_size[CENTRAL] -= c
    for name in tokens:
        tid = TOKEN_BY_NAME[name].id
        if tid in t.faceup:
            t.faceup.remove(tid)
        else:
            assert t.prog_unseen[tid] > 0, name
            t.prog_unseen[tid] -= 1
            t.prog_stack -= 1
        t.tokens[0][tid] += 1
    t.node_type = -1
    t.queue = (("check",),) + t.queue
    t._run()
    return t


def _science_state(counts):
    """Player 0 holds the given science symbols at the first decision.

    The engine only exposes a *choice* of science set when two sets are possible at once
    (e.g. {tablet, tablet, gear, gear}), which needs the Science token's extra pick in real
    play; we build it directly (see :func:`_p0_with_cards`).
    """
    return _p0_with_cards(counts)


# --------------------------------------------------------------------------- (a) strength
def test_heuristic_beats_random():
    res = play_match(make_env, (HeuristicAgent(seed=1), RandomAgent(seed=2)), 30, seed=100)
    assert res.games == 30
    assert res.score(0) >= 0.75, res.summary(("heuristic", "random"))
    assert np.mean(res.score_diffs) > 0


# --------------------------------------------------------------------------- (b) legality
def test_legal_on_random_and_mixed_playouts_all_decision_kinds():
    rng = np.random.default_rng(7)
    states = _collect_states(12, seed=3, mix="random") + _collect_states(12, seed=4, mix="mixed")
    # Halicarnassus games for the look-5 decisions
    states += _collect_states(8, seed=5, wonders=(HALI, GIZA), mix="mixed")
    states += _collect_states(8, seed=6, wonders=(RHODES, HALI), mix="mixed")
    assert len(states) > 500
    seen = set()
    for s in states:
        legal = s.legal_actions()
        a = heuristic_action(s, rng)
        assert a in legal, s.describe()
        seen.add(s.dkind)
        if len(legal) > 1:
            pr = heuristic_prior(s)
            assert pr.shape == (A.NUM,)
            assert pr.dtype == np.float32
            assert abs(float(pr.sum()) - 1.0) < 1e-5
            assert all(pr[x] == 0.0 for x in range(A.NUM) if x not in legal)
            assert all(pr[x] > 0.0 for x in legal)
    for dk in (D_PICK, D_PAY, D_TOKEN, D_HALI_DECK, D_HALI_CHOOSE):
        assert dk in seen, f"decision kind {DECISION_NAMES[dk]} not covered"
    # the science-set choice only appears with two simultaneous sets: synthetic coverage
    t = _science_state({"tablet": 2, "gear": 2})
    assert t.dkind == D_SCIENCE and len(t.legal_actions()) == 2
    a = heuristic_action(t, rng)
    assert a in t.legal_actions()
    t.apply_action(a)  # engine accepts it


def test_heuristic_is_deterministic_without_rng_and_prior_argmax_matches():
    states = [s for s in _collect_states(6, seed=11, mix="mixed") if len(s.legal_actions()) > 1]
    P = replace(HeuristicParams(), tie_noise=0.0)
    for s in states[:300]:
        a1 = heuristic_action(s, None, P)
        a2 = heuristic_action(s, np.random.default_rng(0), P)
        assert a1 == a2
        pr = heuristic_prior(s, P, temperature=0.05)
        assert pr[a1] == pr.max()  # greedy action has maximal prior (ties are possible)


def test_epsilon_greedy_stays_legal_and_agent_protocol():
    agent = HeuristicAgent(epsilon=0.5, seed=3)
    assert isinstance(agent, Agent)
    assert agent.name.startswith("heuristic")
    assert HeuristicAgent().name == "heuristic"
    agent.new_game()
    for s in _collect_states(3, seed=12, mix="random")[:200]:
        assert agent.select_action(s) in s.legal_actions()


# --------------------------------------------------------------------------- (c) speed
def test_heuristic_action_is_fast():
    rng = np.random.default_rng(0)
    states = [s for s in _collect_states(10, seed=21, mix="mixed") if len(s.legal_actions()) > 1][:600]
    for s in states:  # warm-up
        heuristic_action(s, rng)
    t0 = time.perf_counter()
    reps = 3
    for _ in range(reps):
        for s in states:
            heuristic_action(s, rng)
    per_call = (time.perf_counter() - t0) / (reps * len(states))
    # ~60-70 us on the development machine; generous bound for slow CI boxes
    assert per_call < 1e-3, f"heuristic_action too slow: {per_call * 1e6:.0f} us"


# --------------------------------------------------------------------------- behaviour
def _forced_start(deck0_top: str, deck1_top: str):
    """Initial Giza-vs-Rhodes state with chosen visible tops, resolved to P0's first pick."""
    s = GameState.initial((GIZA, RHODES))
    wanted = {0: K[deck0_top].id, 1: K[deck1_top].id}
    while s.is_chance():
        outs = s.chance_outcomes()
        pick = outs[0][0]
        if s.ckind == 0 and s.cctx in wanted:  # C_REVEAL of a wonder deck
            pick = wanted[s.cctx]
            assert any(o == pick for o, _ in outs)
        s = s.apply_chance(pick)
    assert s.to_move() == 0 and s.dkind == D_PICK
    return s


def _draw_center(s: GameState, kind_name: str) -> GameState:
    """Mover takes the central card, which turns out to be ``kind_name``; other chance = first outcome."""
    s = s.apply_action(A.PICK_CENTER)
    k = K[kind_name].id
    while s.is_chance():
        outs = s.chance_outcomes()
        pick = k if (s.ckind == 1 and any(o == k for o, _ in outs)) else outs[0][0]  # C_DRAW_CENTRAL
        s = s.apply_chance(pick)
    return s


def test_denial_prefers_card_that_completes_opponent_stage():
    # P0 takes a blue card, P1 draws WOOD (needs one more *different* resource for Rhodes stage 1);
    # P0 now sees STONE on deck 0 and CIV3 on deck 1.
    s = _forced_start("stone", "civ3")
    s_need = _draw_center(_draw_center(s, "civ3"), "wood")
    # control: identical except P1 drew a blue card instead (no urgent need for stone)
    s_ctrl = _draw_center(_draw_center(s, "civ3"), "civ3")
    for st in (s_need, s_ctrl):
        assert st.to_move() == 0 and st.dkind == D_PICK
        assert st.deck_top[0] == K["stone"].id and st.deck_top[1] == K["civ3"].id
    P = replace(HeuristicParams(), tie_noise=0.0)

    def gap(st):
        legal, sc = score_actions(st, P)
        d = dict(zip(legal, sc))
        return d[A.PICK_LEFT] - d[A.PICK_RIGHT]  # stone (own deck) minus civ3 (opponent's deck)

    assert gap(s_need) > gap(s_ctrl) + 0.5
    assert heuristic_action(s_need, None, P) == A.PICK_LEFT


def test_fifth_stage_completion_depends_on_winning():
    """Synthetic: P0 has 4 Giza stages and 3 different resources; PAPYRUS on deck 0 completes the
    Wonder (game over).  It must be taken when that wins and avoided when it loses."""
    base = _forced_start("papyrus", "civ3")
    P = replace(HeuristicParams(), tie_noise=0.0)

    def make(opp_blue: int) -> GameState:
        t = base._copy()
        t._legal = base._legal  # _copy() drops the cached legal list (as Environment.observe restores it)
        t.built = [first_n(4), 0]
        for name in ("wood", "stone", "clay"):
            t.cards[0][K[name].id] = 1
        t.cards[1][K["civ3"].id] = opp_blue
        return t

    giza4 = sum(WONDER_BY_NAME["Giza"].stages[i].vp for i in range(5))  # 30 VP when complete
    winning = make(opp_blue=(giza4 - 3) // 3)      # 27 < 30 -> completing wins
    losing = make(opp_blue=(giza4 + 3) // 3)       # 33 > 30 -> completing loses
    assert winning.scores()[1] < giza4 < losing.scores()[1]
    assert heuristic_action(winning, None, P) == A.PICK_LEFT
    assert heuristic_action(losing, None, P) != A.PICK_LEFT
    legal, sc = score_actions(winning, P)
    assert dict(zip(legal, sc))[A.PICK_LEFT] > P.win_bonus / 2
    legal, sc = score_actions(losing, P)
    assert dict(zip(legal, sc))[A.PICK_LEFT] < -P.lose_penalty / 2


def test_science_choice_keeps_future_potential():
    # {tablet x2, gear, compass}: pair(tablet) leaves {gear, compass} (any green completes a set),
    # triple leaves {tablet} -> the pair must be preferred.
    t = _science_state({"tablet": 2, "gear": 1, "compass": 1})
    assert t.dkind == D_SCIENCE
    legal = t.legal_actions()
    assert A.SCI_TRIPLE in legal and A.SCI_PAIR_BASE + 0 in legal
    P = replace(HeuristicParams(), tie_noise=0.0)
    assert heuristic_action(t, None, P) == A.SCI_PAIR_BASE + 0


def test_payment_prefers_spending_useless_grey_over_coins():
    """Coins are wild, so when a grey card and a coin are both legal first payments the grey is
    the better spend: keeping the coin never hurts the next stage more than keeping the grey.

    Deterministic Giza hands (stage costs: 2 different, 2 identical, 3 different, 3 identical,
    4 different) in which the engine offers both a grey and a plain coin first; the coin must be
    kept and, where only some greys keep the next stage affordable, one of those must be paid.
    """
    P = replace(HeuristicParams(), tie_noise=0.0)
    W, S, C = A.PAY_BASE + 0, A.PAY_BASE + 1, A.PAY_BASE + 2
    cases = [
        # (stage, hand, acceptable first payments)
        (0, {"wood": 1, "stone": 1, "coin": 2}, {W, S}),               # the 2 coins are the identical pair of stage 2
        (0, {"wood": 1, "stone": 1, "clay": 1, "coin": 2}, {W, S, C}),  # any grey: a grey + coin then pays stage 2
        (0, {"wood": 1, "stone": 2, "clay": 1, "coin": 2}, {W, C}),     # keep the stone pair *and* the coins
        (1, {"wood": 3, "coin": 2}, {W}),                              # wood pair now, wood + 2 coins for the 3 different
        (2, {"wood": 2, "stone": 1, "clay": 1, "coin": 3}, {W, S, C}),  # 3 coins alone would leave 2 wood for 3 identical
        (3, {"stone": 4, "coin": 3}, {S}),                             # stone x3 keeps stone + 3 coins for the 4 different
    ]
    for stage, hand, acceptable in cases:
        t = _p0_with_cards(hand, stage=stage)
        assert t.dkind == D_PAY and t.dctx[1] == (), (stage, hand, t.describe())
        legal = list(t.legal_actions())
        greys = [a for a in legal if A.PAY_BASE <= a < A.PAY_BASE + 5]
        assert greys and A.PAY_COIN in legal, (stage, hand, legal)
        a = heuristic_action(t, None, P)
        assert a in acceptable, (stage, hand, A.name(a))
        d = dict(zip(*score_actions(t, P)))
        assert max(d[g] for g in greys) > d[A.PAY_COIN] + P.pay_coin_penalty, (stage, hand, d)
    # The Economy doubled coin is the exception: it saves a card.  wood x2 + coin: pay coin x2 and keep
    # the wood pair for stage 2 (a wood + the coin would leave a single wood); but wood + stone + coin:
    # pay the greys, the doubled coin alone then pays stage 2.
    t = _p0_with_cards({"wood": 2, "coin": 1}, tokens=("Economy",))
    assert t.dkind == D_PAY and list(t.legal_actions()) == [W, A.PAY_COIN2]
    assert heuristic_action(t, None, P) == A.PAY_COIN2
    t = _p0_with_cards({"wood": 1, "stone": 1, "coin": 1}, tokens=("Economy",))
    assert t.dkind == D_PAY and list(t.legal_actions()) == [W, S, A.PAY_COIN2]
    assert heuristic_action(t, None, P) in (W, S)


def test_payment_choice_accounts_for_the_canonical_code_order():
    # Giza stage 1 (2 different) with wood, stone x2, clay x2; stage 2 needs 2 identical.  The engine
    # offers wood or stone first (clay, the highest code, cannot come first).  Wood first keeps a pair
    # (and stage 2 is built in the same turn); stone first can only be followed by clay, which leaves
    # one card of each resource.
    t = _p0_with_cards({"wood": 1, "stone": 2, "clay": 2})
    assert t.dkind == D_PAY and list(t.legal_actions()) == [A.PAY_BASE + 0, A.PAY_BASE + 1]
    P = replace(HeuristicParams(), tie_noise=0.0)
    assert heuristic_action(t, None, P) == A.PAY_BASE + 0
    s = t
    while s.to_move() == 0 and s.dkind == D_PAY:
        s = s.apply_action(heuristic_action(s, None, P))
    assert s.num_stages(0) == 2, s.describe()


def test_prior_temperature_controls_sharpness():
    states = [s for s in _collect_states(4, seed=41, mix="mixed") if s.dkind == D_PICK and len(s.legal_actions()) == 3]
    assert states
    s = states[0]
    sharp = heuristic_prior(s, temperature=0.1)
    flat = heuristic_prior(s, temperature=100.0)
    assert sharp.max() > flat.max()
    assert abs(float(flat[list(s.legal_actions())].min()) - 1.0 / 3.0) < 0.05
