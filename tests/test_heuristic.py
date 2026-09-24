"""Tests for the hand-crafted expert policy (``sevenwa.agents.heuristic``)."""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

from sevenwa.agents.base import Agent
from sevenwa.agents.heuristic import (HeuristicAgent, HeuristicParams, _Ctx, heuristic_action, heuristic_prior,
                                      score_actions)
from sevenwa.agents.random_agent import RandomAgent
from sevenwa.engine.actions import Actions as A
from sevenwa.engine.cards import KIND_BY_NAME
from sevenwa.engine.env import make_env
from sevenwa.engine.state import (C_PEEK, C_REVEAL, C_TOKEN_REVEAL, CENTRAL, D_HALI_CHOOSE, D_HALI_DECK, D_PAY,
                                  D_PICK, D_SCIENCE, D_TOKEN, DECISION_NAMES, GameState)
from sevenwa.engine.tokens import TOKEN_BY_NAME
from sevenwa.engine.wonders import WONDER_BY_NAME

from conftest import edit as edit_copy, first_n, first_pick, run, set_faceup, settle  # noqa: E402  (tests/conftest.py)
from sevenwa.train.arena import play_match

K = KIND_BY_NAME
GIZA = WONDER_BY_NAME["Giza"].id
RHODES = WONDER_BY_NAME["Rhodes"].id
HALI = WONDER_BY_NAME["Halicarnassus"].id
BABYLON = WONDER_BY_NAME["Babylon"].id


def _tok(name: str) -> int:
    """Token action for the face-up Progress token ``name``."""
    return A.TOKEN_BASE + TOKEN_BY_NAME[name].id


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


def _p0_with_cards(counts, stage: int = 0, tokens=(), faceup=None, wonders=(GIZA, RHODES), built=None, edit=None):
    """Synthetic state: player 0 (Giza, at the given stage) holds the given cards (and Progress
    tokens, by name) at the first decision, and the mandatory checks (science sets /
    construction) have been re-run.

    Cards are moved from the central deck into the tableau and tokens out of the face-up row /
    stack so the state stays consistent.  ``faceup`` (token names) replaces the face-up row,
    ``built`` overrides player 0's built-mask (default: the first ``stage`` stages) and ``edit(t)``
    may change anything else (the opponent's tableau...) before the checks run.
    """
    s = GameState.initial(wonders)
    # resolve chance nodes with fixed outcomes until the first decision
    while s.is_chance():
        s = s.apply_chance(s.chance_outcomes()[0][0])
    assert s.to_move() == 0 and s.dkind == D_PICK
    t = s._copy()
    t.built[0] = first_n(stage) if built is None else built
    for name, c in counts.items():
        k = K[name].id
        assert t.unseen[CENTRAL][k] >= c, name
        t.cards[0][k] += c
        t.unseen[CENTRAL][k] -= c
        t.deck_size[CENTRAL] -= c
    if faceup is not None:
        set_faceup(t, [TOKEN_BY_NAME[n].id for n in faceup])
    for name in tokens:
        tid = TOKEN_BY_NAME[name].id
        if tid in t.faceup:
            t.faceup.remove(tid)
        else:
            assert t.prog_unseen[tid] > 0, name
            t.prog_unseen[tid] -= 1
            t.prog_stack -= 1
        t.tokens[0][tid] += 1
    if edit is not None:
        edit(t)
    t.node_type = -1
    t.queue = (("check",),) + t.queue
    t._run()
    return t


def _give_p1(t: GameState, name: str, c: int = 1) -> None:
    """Move ``c`` cards of kind ``name`` from the central deck into player 1's tableau."""
    k = K[name].id
    assert t.unseen[CENTRAL][k] >= c, name
    t.cards[1][k] += c
    t.unseen[CENTRAL][k] -= c
    t.deck_size[CENTRAL] -= c


def _hold_token(t: GameState, p: int, name: str) -> None:
    """Give player ``p`` a copy of token ``name`` from the stack (the face-up row is untouched)."""
    tid = TOKEN_BY_NAME[name].id
    assert t.prog_unseen[tid] > 0, name
    t.prog_unseen[tid] -= 1
    t.prog_stack -= 1
    t.tokens[p][tid] += 1


def _peeked_state(holder: int, tc: int, tops=(K["wood"].id, K["stone"].id)) -> GameState:
    """``holder`` holds the Cat and peeks ``tc`` (the central top) at their turn start.  With
    ``holder == 1`` P1 then takes their own deck's top, so the result is P0's main pick with a
    central top known to P1 only (an in-tree state; ``Environment.observe(0)`` would hide it)."""
    s = first_pick(tops=tops)
    c = edit_copy(s)
    c.cat = holder
    c.mover = holder
    c = run(c, ("turn_start",))
    assert c.is_chance() and c.ckind == C_PEEK, c.describe_node()
    c = c.apply_chance(tc)
    assert c.dkind == D_PICK and c.mover == holder and c.deck_top[CENTRAL] == tc and c.knows_central(holder)
    if holder == 1:
        c = settle(c.apply_action(A.PICK_LEFT))
        assert c.dkind == D_PICK and c.mover == 0, c.describe_node()
        assert c.deck_top[CENTRAL] == tc and c.knows_central(1) and not c.knows_central(0)
    return c


def _hide_central(c: GameState) -> GameState:
    """Player 0's own observation of ``c``: the central top is hidden (as ``Environment.observe``)."""
    o = edit_copy(c)
    o.unseen[CENTRAL][o.deck_top[CENTRAL]] += 1
    o.deck_top[CENTRAL] = -1
    o.central_known_to = c.central_known_to & ~1
    return o


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
def _forced_start(deck0_top: str, deck1_top: str, wonders=(GIZA, RHODES), faceup=None):
    """Initial (Giza-vs-Rhodes) state with chosen visible tops (and face-up tokens, by name),
    resolved to P0's first pick."""
    s = GameState.initial(wonders)
    wanted = {0: K[deck0_top].id, 1: K[deck1_top].id}
    toks = [TOKEN_BY_NAME[n].id for n in (faceup or ())]
    while s.is_chance():
        outs = s.chance_outcomes()
        pick = outs[0][0]
        if s.ckind == C_REVEAL and s.cctx in wanted:  # C_REVEAL of a wonder deck
            pick = wanted[s.cctx]
        elif s.ckind == C_TOKEN_REVEAL and toks:
            pick = toks.pop(0)
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


# --------------------------------------------------------------------------- end-game tokens / stages
def test_token_that_forces_a_losing_fifth_stage_is_avoided():
    """P0 (Giza, 4 stages = 22 VP) holds wood x2 + stone x2: with Engineering the 4-different last stage
    becomes affordable, the build is mandatory and the game ends 30-31 -- a loss on the spot.  P1
    (Rhodes, 4 stages + civ3 x4 = 31) holds glass x4, so Engineering also hands *them* a winning
    build: the denial term alone would make P0 grab the token."""
    def opp(t):
        t.built[1] = first_n(4)
        _give_p1(t, "civ3", 4)
        _give_p1(t, "glass", 4)
    t = _p0_with_cards({"tablet": 2, "wood": 2, "stone": 2}, stage=4, faceup=("Engineering", "Urbanism", "Crafts"),
                       edit=opp)
    assert t.dkind == D_TOKEN and t.scores() == (22, 31)
    P = replace(HeuristicParams(), tie_noise=0.0)
    eng = _tok("Engineering")
    d = dict(zip(*score_actions(t, P)))
    others = [x for a, x in d.items() if a != eng]
    assert d[eng] < min(others) - P.lose_penalty / 4, d
    assert heuristic_action(t, None, P) != eng
    # control: ahead on points the forced build *wins* -> Engineering is by far the best token
    t = _p0_with_cards({"tablet": 2, "wood": 2, "stone": 2}, stage=4, faceup=("Engineering", "Urbanism", "Crafts"))
    assert t.dkind == D_TOKEN and t.scores() == (22, 0)
    d = dict(zip(*score_actions(t, P)))
    assert d[eng] > max(x for a, x in d.items() if a != eng) + P.win_bonus / 4
    assert heuristic_action(t, None, P) == eng


def test_babylon_last_stage_token_counts_towards_the_win():
    """Babylon's 3-identical stage (Progress token) may be constructed last; its token is chosen before
    the game ends.  P0 (S1, S2, S3, S5 = 15 VP) with stone x2 completes it with the stone on deck 0:
    20 VP against P1's 22 loses -- unless Decor (6 VP with the Wonder complete) is face-up."""
    P = replace(HeuristicParams(), tie_noise=0.0)
    for faceup, wins in ((("Decor", "Urbanism", "Crafts"), True), (("Engineering", "Urbanism", "Crafts"), False)):
        base = _forced_start("stone", "civ3", wonders=(BABYLON, RHODES), faceup=faceup)
        t = base._copy()
        t._legal = base._legal
        t.built = [0b10111, first_n(4)]
        t.cards[0][K["stone"].id] = 2
        _give_p1(t, "civ3", 1)
        assert t.scores() == (15, 22) and t.available_stages(0) == (3,)
        d = dict(zip(*score_actions(t, P)))
        if wins:
            assert d[A.PICK_LEFT] > P.win_bonus / 2, d
            assert heuristic_action(t, None, P) == A.PICK_LEFT
        else:
            assert d[A.PICK_LEFT] < -P.lose_penalty / 2, d
            assert heuristic_action(t, None, P) != A.PICK_LEFT


def test_endgame_token_choice_takes_the_points():
    """Once the Wonder is complete (Babylon's token stage built last) the token choice is the last
    decision of the game: a token is worth exactly the VP it scores, nothing else."""
    P = replace(HeuristicParams(), tie_noise=0.0)
    dec, cul, tac = _tok("Decor"), _tok("Culture"), _tok("Tactics")
    t = _p0_with_cards({"stone": 3}, built=0b10111, faceup=("Decor", "Culture", "Tactics"), wonders=(BABYLON, RHODES))
    assert t.dkind == D_TOKEN and t.wonder_done and t.dctx == "babylon"
    d = dict(zip(*score_actions(t, P)))
    assert (d[dec], d[cul], d[tac]) == (6.0, 4.0, 0.0)  # no Battle pending: Tactics scores nothing
    assert heuristic_action(t, None, P) == dec
    # the second Culture (12 - 4) beats Decor; Education adds 2 to every token
    def hold(t):
        _hold_token(t, 0, "Culture")
        _hold_token(t, 0, "Education")
    t = _p0_with_cards({"stone": 3}, built=0b10111, faceup=("Decor", "Culture", "Tactics"), wonders=(BABYLON, RHODES),
                       edit=hold)
    assert t.dkind == D_TOKEN and t.wonder_done
    d = dict(zip(*score_actions(t, P)))
    assert (d[dec], d[cul], d[tac]) == (8.0, 10.0, 2.0)
    assert heuristic_action(t, None, P) == cul


def test_decor_beats_culture_at_the_finish_line():
    """P0 (Giza, 4 stages, 22 VP vs 0) holds 3 different resources: one card completes a *winning*
    Wonder, so Decor is worth its full 6 VP, while the Culture pair bonus needs time that is gone."""
    P = replace(HeuristicParams(), tie_noise=0.0)
    dec, cul = _tok("Decor"), _tok("Culture")
    hand = {"tablet": 2, "wood": 1, "stone": 1, "clay": 1}
    t = _p0_with_cards(hand, stage=4, faceup=("Decor", "Culture", "Strategy"))
    assert t.dkind == D_TOKEN
    d = dict(zip(*score_actions(t, P)))
    assert d[dec] > d[cul] + 0.5, d
    assert heuristic_action(t, None, P) == dec
    # control: when completing loses (P1 leads 31-30) the race probability applies as before
    def opp(t):
        t.built[1] = first_n(4)
        _give_p1(t, "civ3", 4)
    t2 = _p0_with_cards(hand, stage=4, faceup=("Decor", "Culture", "Strategy"), edit=opp)
    assert t2.dkind == D_TOKEN and t2.scores() == (22, 31)
    d2 = dict(zip(*score_actions(t2, P)))
    assert d2[dec] < d[dec] - 0.5, (d, d2)


# --------------------------------------------------------------------------- hidden information
def test_pick_ignores_a_central_card_known_only_to_the_opponent():
    """In-tree states may carry a central top card sampled for the *opponent's* Cat peek.  The mover
    does not know it: the scores must equal those of the mover's own observation (top hidden),
    whatever the sampled card is."""
    P = replace(HeuristicParams(), tie_noise=0.0)
    ref = None
    for name in ("wood", "civ3", "shield_h2", "tablet"):
        s = _peeked_state(1, K[name].id)
        legal, sc = score_actions(s, P)
        legal_h, sc_h = score_actions(_hide_central(s), P)
        assert list(legal) == list(legal_h)
        assert sc == pytest.approx(sc_h), name
        if ref is None:
            ref = sc
        assert sc == pytest.approx(ref), name


# --------------------------------------------------------------------------- tunables
def test_new_tunables_default_to_the_previous_behaviour():
    P = replace(HeuristicParams(), tie_noise=0.0)
    assert (P.blind_token_discount, P.deny_far, P.arch_last_pick) == (1.0, 1.0, 1.0) and P.deny_hidden_central < 0
    hand = {"tablet": 2, "wood": 1, "stone": 1, "clay": 1}
    t = _p0_with_cards(hand, stage=4, faceup=("Decor", "Culture", "Strategy"))
    assert t.dkind == D_TOKEN and not any(t.cards[1][K[g].id] for g in ("tablet", "gear", "compass"))
    d = dict(zip(*score_actions(t, P)))
    # (1) blind_token_discount scales the blind-stack expectation (mine, and the opponent's reply)
    d1 = dict(zip(*score_actions(t, replace(P, blind_token_discount=0.5))))
    assert d1[A.TOKEN_BLIND] < d[A.TOKEN_BLIND] - 0.5
    assert all(d1[a] >= d[a] for a in d if a != A.TOKEN_BLIND)
    # (2) deny_far: the opponent holds no green card -> deny_far = 0 removes the denial altogether;
    #     with a pair in their tableau the denial is full whatever deny_far says
    d2 = dict(zip(*score_actions(t, replace(P, deny_far=0.0))))
    assert all(d2[a] > d[a] + 0.5 for a in d), (d, d2)
    tp = edit_copy(t)
    _give_p1(tp, "gear", 2)
    assert dict(zip(*score_actions(tp, P))) == dict(zip(*score_actions(tp, replace(P, deny_far=0.0))))
    # (3) deny_hidden_central: P0 peeked a wood the opponent (Rhodes, empty) wants more than the blue tops
    s = _peeked_state(0, K["wood"].id, tops=(K["civ3"].id, K["civ2cat"].id))
    d3 = dict(zip(*score_actions(s, P)))
    assert d3 == dict(zip(*score_actions(s, replace(P, deny_hidden_central=P.deny))))
    d3b = dict(zip(*score_actions(s, replace(P, deny_hidden_central=0.0))))
    assert d3b[A.PICK_CENTER] == d3[A.PICK_CENTER]
    assert d3b[A.PICK_LEFT] > d3[A.PICK_LEFT] + 0.5 and d3b[A.PICK_RIGHT] > d3[A.PICK_RIGHT] + 0.5, (d3, d3b)
    # (4) arch_last_pick: the pick after the *last* stage
    t4 = _p0_with_cards(hand, stage=4, faceup=("Architecture", "Decor", "Strategy"))
    assert t4.dkind == D_TOKEN
    arch = _tok("Architecture")
    d4 = dict(zip(*score_actions(t4, P)))
    d4b = dict(zip(*score_actions(t4, replace(P, arch_last_pick=0.0))))
    assert d4b[arch] < d4[arch] - 1.0 and d4b[_tok("Decor")] == d4[_tok("Decor")]
    assert dict(zip(*score_actions(t4, replace(P, arch_last_pick=1.0)))) == d4


# --------------------------------------------------------------------------- horizon / battle timing
def test_stall_horizon_ignores_a_player_whose_completion_loses():
    """P0 (Giza, 4 stages = 22 VP) holds 3 different resources: one card from the 5th stage, but
    completing it (30 VP) loses to P1 (Rhodes, 1 stage + 4 civ3 + 8 civ2cat = 32 VP), so P0 stalls and
    the game does not end within ~1 turn.  The previous horizon min(cards_needed) / pick_rate
    collapsed to ~1.4 turns for *both* players; the stall-aware one uses P1's 12 missing cards."""
    P = replace(HeuristicParams(), tie_noise=0.0)
    assert P.stall_horizon == 0.0  # default: the previous horizon (the A/B did not favour the stall-aware one)
    hand = {"wood": 1, "stone": 1, "clay": 1}

    def opp(t):
        t.built[1] = first_n(1)
        _give_p1(t, "civ3", 4)
        _give_p1(t, "civ2cat", 8)
    t = _p0_with_cards(hand, stage=4, edit=opp)  # the check ends P0's turn: P1 (the leader) is to move
    assert t.dkind == D_PICK and t.to_move() == 1 and t.scores() == (22, 32)
    old = _Ctx(t, replace(P, stall_horizon=0.0))
    new = _Ctx(t, replace(P, stall_horizon=1.0))
    assert old.views[0].cards_needed == 1 and old.views[1].cards_needed == 12
    assert not old.fifth_wins(0, old.views[0].tgt)
    assert old.tl == pytest.approx(1.0 / P.pick_rate) and _Ctx(t, P).tl == old.tl
    assert new.tl > 10.0 and new.sci_time == 1.0 and new.pb > old.pb
    # P1's time-scaled values follow the horizon (Urbanism: extra picks per remaining turn)
    urb = TOKEN_BY_NAME["Urbanism"].id
    assert new.token_values(1)[urb] > 5 * old.token_values(1)[urb]
    # a fractional weight interpolates the cards-needed bound
    half = _Ctx(t, replace(P, stall_horizon=0.5))
    assert half.tl == pytest.approx((1 + 0.5 * (12 - 1)) / P.pick_rate)
    # P0 still refuses the losing completion (the horizon does not touch the 5th-stage penalty)
    for k in ("papyrus", "glass", "coin"):
        assert new.card_values(0)[K[k].id] < -P.lose_penalty / 2, k
    # control: when completing *wins*, the 1-card horizon is right and nothing changes
    t2 = _p0_with_cards(hand, stage=4)
    assert t2.dkind == D_PICK and t2.to_move() == 1 and t2.scores() == (22, 0)
    a, b = _Ctx(t2, replace(P, stall_horizon=0.0)), _Ctx(t2, replace(P, stall_horizon=1.0))
    assert a.fifth_wins(0, a.views[0].tgt) and a.tl == b.tl == pytest.approx(1.0 / P.pick_rate)
    assert dict(zip(*score_actions(t2, replace(P, stall_horizon=1.0)))) == dict(zip(*score_actions(t2, P)))


def test_battle_response_scales_the_counterfactual_battle():
    """At conflict 2 a horn card fights the Battle this turn.  P1 holds a hornless shield, P0 none:
    taking the horn card turns a pb-weighted loss into a tie, worth o1 - pb * r * o0 with r =
    ``battle_response`` (1.0 = previous behaviour)."""
    base = _forced_start("shield_h1", "civ3")
    t = base._copy()
    t._legal = base._legal
    t.conflict = 2
    _give_p1(t, "shield", 1)
    P = replace(HeuristicParams(), tie_noise=0.0)
    assert P.battle_response == 1.0
    h1 = K["shield_h1"].id
    vals = {}
    for r in (1.0, 0.75, 0.5):
        c = _Ctx(t, replace(P, battle_response=r))
        vals[r] = (c.card_values(0)[h1], c.pb)
    pb = vals[1.0][1]
    assert 0.0 < pb < 1.0
    o0 = -t.rules.military_token_vp * (1 if t.rules.double_vs_zero_requires_two else 2)  # 0 vs 1 shield
    for r, (v, _) in vals.items():
        assert v == pytest.approx(0.0 - pb * r * o0), r
    assert vals[1.0][0] > vals[0.75][0] > vals[0.5][0] > 0.0
