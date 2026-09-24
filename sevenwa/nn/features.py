"""State -> fixed-size feature vector (mover-centric) and legal-action mask.

Two versioned layouts exist; a network's ``cfg.input_dim`` identifies the one it was trained on
(:func:`encoder_for_dim`), so every checkpoint keeps being served the features it expects.

Version 1 (``FEATURE_SIZE_V1`` = 581, all values roughly in [0, 1]):

* ``me`` block, ``opp`` block            – wonder, built/available stages (per stage), tableau, science, tokens, military
* ``deck_own``, ``deck_opp``, ``deck_center`` – size, visible top card, unseen-multiset histogram
* ``global`` block                         – progress tokens, conflict, cat, decision context

Version 2 (``FEATURE_SIZE_V2``, the latest) = version 1 followed by the **EXPERT** block
(``EXPERT_SLICE``): the race quantities the expert heuristic (``sevenwa.agents.heuristic``)
computes for the position, from one ``_Ctx`` per state, so that the value head does not have to
reconstruct them.  ``encode_state(s, version=2)[:FEATURE_SIZE_V1] == encode_state(s, version=1)``.
Per player (``me`` = the player encoded for, then ``opp``; ``EXPERT_PLAYER_FEATS`` each):

* per stage in cost order: cards still missing to pay it now / 4 (coins wild, Engineering /
  Economy as ``heuristic._maxval``; 0 once built)
* ``cards_needed`` / 14 (cards still missing to finish the Wonder), ``next_deficit`` / 4 of the
  primary target, number of available stages / 3
* the heuristic card values of the 14 kinds / 20, clipped to [-2, 4]
* best obtainable Progress token / 30, blind-token expectation / 30
* ``build_value`` of the primary target: plain / 20 and total / 60 (clipped to [-1.5, 1.5];
  the total includes the win bonus / loss penalty of a 5th stage)
* value of one permanent shield / 10

Global: turns-to-go ``tl`` / 16, battle probability ``pb``, future-battle weight ``nb`` / 2,
``sci_time``, and the heuristic's softmax prior over the global action space (as
``heuristic_prior``) when the state is a decision node with at least two legal actions.  States
that are not decision nodes get an all-zero EXPERT block.  The block only uses the belief of the
player encoded for: a central top card that player does not know is folded back into the unseen
multiset before the heuristic sees the state.

``ENTITY_SLICES`` (version 1) and ``ENTITY_SLICES_V2`` expose the block boundaries for the
attention trunk (:func:`entity_slices`).
"""
from __future__ import annotations

import warnings
from typing import Callable, List, Optional, Tuple

import numpy as np

from ..engine.actions import Actions as A
from ..engine.cards import BLUE, GREEN, GREY, KINDS, NUM_KINDS, RED, YELLOW, KIND_OF_SYMBOL, COIN_KIND, KIND_OF_RESOURCE
from ..engine.state import (CENTRAL, D_HALI_CHOOSE, D_HALI_DECK, D_PAY, D_PICK, D_SCIENCE, D_STAGE, D_TOKEN, GameState,
                            N_DECISION, PAY_COIN2_CODE, PAY_COIN_CODE)
from ..engine.tokens import NUM_TOKEN_TYPES, TOKENS
from ..engine.wonders import EFFECTS, IDENTICAL, NUM_STAGES, NUM_WONDERS, WONDERS

# ---- static card-kind feature table: kind one-hot | type one-hot | resource one-hot | vp | cat | symbol one-hot | shields | horns
KIND_FEATS = NUM_KINDS + 5 + 5 + 1 + 1 + 3 + 1 + 1
_KIND_TABLE = np.zeros((NUM_KINDS, KIND_FEATS), dtype=np.float32)
for _k in KINDS:
    row = _KIND_TABLE[_k.id]
    row[_k.id] = 1.0
    row[NUM_KINDS + _k.type] = 1.0
    if _k.resource >= 0:
        row[NUM_KINDS + 5 + _k.resource] = 1.0
    row[NUM_KINDS + 10] = _k.vp / 3.0
    row[NUM_KINDS + 11] = 1.0 if _k.cat else 0.0
    if _k.symbol >= 0:
        row[NUM_KINDS + 12 + _k.symbol] = 1.0
    row[NUM_KINDS + 15] = _k.shields / 2.0
    row[NUM_KINDS + 16] = _k.horns / 2.0

_TYPE_OF_KIND = np.array([k.type for k in KINDS], dtype=np.int64)
_TYPE_MASKS = [(_TYPE_OF_KIND == t).astype(np.float32) for t in range(5)]
_BLUE_VP = np.array([k.vp if k.type == BLUE else 0 for k in KINDS], dtype=np.float32)
_BLUE_CAT = np.array([1.0 if (k.type == BLUE and k.cat) else 0.0 for k in KINDS], dtype=np.float32)
_RED_SHIELDS = np.array([k.shields if k.type == RED else 0 for k in KINDS], dtype=np.float32)
_HORN_OF_KIND = np.array([1.0 if (k.type == RED and k.horns > 0) else 0.0 for k in KINDS], dtype=np.float32)
_EFFECT_INDEX = {e: i for i, e in enumerate(EFFECTS)}
_SHIELD_TOKEN_IDS = [t.id for t in TOKENS if t.shields > 0]
_SYMBOL_KINDS = [KIND_OF_SYMBOL[s] for s in range(3)]
_GREY_KIND_IDS = [KIND_OF_RESOURCE[r] for r in range(5)]

# static per-stage description: cost one-hot (2/3/4) | identical/different | vp/10 | effect one-hot
STAGE_STATIC = 3 + 2 + 1 + len(EFFECTS)
_STAGE_TABLE = np.zeros((NUM_WONDERS, NUM_STAGES, STAGE_STATIC), dtype=np.float32)
for _w in WONDERS:
    for _st in _w.stages:
        row = _STAGE_TABLE[_w.id, _st.index]
        row[min(_st.cost, 4) - 2] = 1.0
        row[3 + _st.kind] = 1.0
        row[5] = _st.vp / 10.0
        row[6 + _EFFECT_INDEX[_st.effect]] = 1.0
# per stage: built | available | affordable-now-with-own-cards | static description
STAGE_FEATS = 3 + STAGE_STATIC
PLAYER_FEATS = (NUM_WONDERS + 6 + NUM_STAGES * STAGE_FEATS + 3 + NUM_KINDS * 2 + 4 + 2 + 3 + 2
                + NUM_TOKEN_TYPES * 2 + 3 + 1 + 1)
DECK_FEATS = 3 + KIND_FEATS + NUM_KINDS + 6
PICK_REASONS = ["main", "token", "alexandria", "ephesus", "hali", "olympia"]
GLOBAL_FEATS = ((NUM_TOKEN_TYPES * 2 + 1) + (4 + 1) + 3 + 2 + 2 + 7 + (len(PICK_REASONS) + 1) + (1 + 7 + NUM_STAGES)
                + NUM_KINDS + 1)

ME_SLICE = (0, PLAYER_FEATS)
OPP_SLICE = (PLAYER_FEATS, 2 * PLAYER_FEATS)
DECK_OWN_SLICE = (2 * PLAYER_FEATS, 2 * PLAYER_FEATS + DECK_FEATS)
DECK_OPP_SLICE = (DECK_OWN_SLICE[1], DECK_OWN_SLICE[1] + DECK_FEATS)
DECK_CENTER_SLICE = (DECK_OPP_SLICE[1], DECK_OPP_SLICE[1] + DECK_FEATS)
GLOBAL_SLICE = (DECK_CENTER_SLICE[1], DECK_CENTER_SLICE[1] + GLOBAL_FEATS)
FEATURE_SIZE_V1 = GLOBAL_SLICE[1]
assert FEATURE_SIZE_V1 == 581, "the version-1 layout is frozen: checkpoints with input_dim 581 depend on it"
ENTITY_SLICES: List[Tuple[int, int]] = [ME_SLICE, OPP_SLICE, DECK_OWN_SLICE, DECK_OPP_SLICE, DECK_CENTER_SLICE, GLOBAL_SLICE]

# ---- version 2: version 1 + EXPERT block (see the module docstring)
# per player: stage deficits | cards_needed, next_deficit, #available | card values | best, blind token |
#             build plain, total | permanent shield
EXPERT_PLAYER_FEATS = NUM_STAGES + 3 + NUM_KINDS + 2 + 2 + 1
EXPERT_GLOBAL_FEATS = 4 + A.NUM  # tl, pb, nb, sci_time | heuristic prior over the global action space
EXPERT_FEATS = 2 * EXPERT_PLAYER_FEATS + EXPERT_GLOBAL_FEATS
EXPERT_SLICE = (FEATURE_SIZE_V1, FEATURE_SIZE_V1 + EXPERT_FEATS)
EXPERT_ME_SLICE = (EXPERT_SLICE[0], EXPERT_SLICE[0] + EXPERT_PLAYER_FEATS)
EXPERT_OPP_SLICE = (EXPERT_ME_SLICE[1], EXPERT_ME_SLICE[1] + EXPERT_PLAYER_FEATS)
EXPERT_GLOBAL_SLICE = (EXPERT_OPP_SLICE[1], EXPERT_OPP_SLICE[1] + 4)
EXPERT_PRIOR_SLICE = (EXPERT_GLOBAL_SLICE[1], EXPERT_GLOBAL_SLICE[1] + A.NUM)
FEATURE_SIZE_V2 = EXPERT_SLICE[1]
assert EXPERT_PRIOR_SLICE[1] == FEATURE_SIZE_V2
ENTITY_SLICES_V2: List[Tuple[int, int]] = ENTITY_SLICES + [EXPERT_SLICE]

FEATURE_VERSION_LATEST = 2
FEATURE_SIZES = {1: FEATURE_SIZE_V1, 2: FEATURE_SIZE_V2}
FEATURE_SIZE = FEATURE_SIZES[FEATURE_VERSION_LATEST]  # size of the LATEST version (new networks, new data)


def feature_size(version: int = FEATURE_VERSION_LATEST) -> int:
    """Length of the feature vector of encoding ``version``."""
    try:
        return FEATURE_SIZES[int(version)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"unknown feature version {version!r} (known: {sorted(FEATURE_SIZES)})") from None


def entity_slices(version: int = FEATURE_VERSION_LATEST) -> List[Tuple[int, int]]:
    """Entity boundaries (attention trunk) of encoding ``version``."""
    feature_size(version)  # validates
    return list(ENTITY_SLICES if int(version) == 1 else ENTITY_SLICES_V2)


def version_for_dim(input_dim: int) -> int:
    """Feature version whose vectors have ``input_dim`` entries (a network's ``cfg.input_dim``)."""
    for v, n in FEATURE_SIZES.items():
        if n == input_dim:
            return v
    raise ValueError(f"no feature encoding has {input_dim} features (known: "
                     + ", ".join(f"v{v}={n}" for v, n in sorted(FEATURE_SIZES.items()))
                     + "); the network was trained on a different encoder (e.g. before the 2026-09-14 rules "
                       "correction, when the layout had 437 features)")


_ENGINEERING_IDS = [t.id for t in TOKENS if t.kind == "engineering"]
_ECONOMY_IDS = [t.id for t in TOKENS if t.kind == "economy"]


def _affordable(s: GameState, p: int, stage) -> bool:
    """Could player ``p`` pay ``stage`` with the cards they hold right now (coins wild, Engineering/Economy)?"""
    cl = s.cards[p]
    greys = [cl[k] for k in _GREY_KIND_IDS]
    coins = cl[COIN_KIND]
    toks = s.tokens[p]
    econ = coins >= 1 and any(toks[t] for t in _ECONOMY_IDS) and not (p == s.mover and s.econ_used)
    cap = coins + (1 if econ else 0)
    if any(toks[t] for t in _ENGINEERING_IDS):
        value = sum(greys) + cap
    elif stage.kind == 0:
        value = max(greys) + cap
    else:
        value = sum(1 for g in greys if g > 0) + cap
    return value >= stage.cost


def _player_block(out: np.ndarray, off: int, s: GameState, p: int, is_mover: bool) -> None:
    w = WONDERS[s.wonder[p]]
    out[off + s.wonder[p]] = 1.0
    o = off + NUM_WONDERS
    built = s.built[p]
    n_built = s.num_stages(p)
    out[o + n_built] = 1.0
    o += 6
    avail = w.available(built)
    for i in range(NUM_STAGES):
        if (built >> i) & 1:
            out[o] = 1.0
        elif i in avail:
            out[o + 1] = 1.0
            if _affordable(s, p, w.stages[i]):
                out[o + 2] = 1.0
        out[o + 3:o + STAGE_FEATS] = _STAGE_TABLE[w.id, i]
        o += STAGE_FEATS
    out[o] = (5 - n_built) / 5.0
    out[o + 1] = w.cost_remaining(built) / 14.0
    out[o + 2] = (w.total_vp - w.vp_of_built(built)) / 30.0
    o += 3
    cl = s.cards[p]
    cards = np.array(cl, dtype=np.float32)
    out[o:o + NUM_KINDS] = cards * 0.25
    o += NUM_KINDS
    out[o:o + NUM_KINDS] = cards > 0
    o += NUM_KINDS
    greys = [cl[k] for k in _GREY_KIND_IDS]
    out[o] = sum(1 for g in greys if g > 0) / 5.0
    out[o + 1] = sum(greys) / 8.0
    out[o + 2] = cl[COIN_KIND] / 4.0
    out[o + 3] = max(greys) / 4.0
    o += 4
    out[o] = float(cards @ _BLUE_VP) / 20.0
    out[o + 1] = float(cards @ _BLUE_CAT) / 4.0
    o += 2
    sci = [cl[k] for k in _SYMBOL_KINDS]
    out[o] = sci[0] / 3.0
    out[o + 1] = sci[1] / 3.0
    out[o + 2] = sci[2] / 3.0
    o += 3
    out[o] = 1.0 if max(sci) >= 2 else 0.0
    out[o + 1] = 1.0 if min(sci) >= 1 else 0.0
    o += 2
    tl = s.tokens[p]
    out[o:o + NUM_TOKEN_TYPES] = np.array(tl, dtype=np.float32) * 0.5
    o += NUM_TOKEN_TYPES
    if is_mover:
        used = s.tokens_used
        for t in range(NUM_TOKEN_TYPES):
            if (used >> t) & 1:
                out[o + t] = 1.0
    o += NUM_TOKEN_TYPES
    out[o] = (float(cards @ _RED_SHIELDS) + s.wonder_shields[p] + sum(tl[t] * TOKENS[t].shields for t in _SHIELD_TOKEN_IDS)) / 6.0
    out[o + 1] = s.wonder_shields[p] / 2.0
    out[o + 2] = s.mil_tokens[p] / 6.0
    o += 3
    out[o] = 1.0 if s.cat == p else 0.0
    o += 1
    out[o] = s.score_of(p) / 60.0


def _deck_block(out: np.ndarray, off: int, s: GameState, d: int, mover: int) -> None:
    size = s.deck_size[d]
    cap = 60.0 if d == CENTRAL else 25.0
    out[off] = size / cap
    out[off + 1] = 1.0 if size == 0 else 0.0
    top = s.deck_top[d]
    known = top >= 0 and (d != CENTRAL or s.knows_central(mover))
    out[off + 2] = 1.0 if known else 0.0
    o = off + 3
    if known:
        out[o:o + KIND_FEATS] = _KIND_TABLE[top]
    o += KIND_FEATS
    ul = s.unseen[d]
    tot = sum(ul) + (1 if (top >= 0 and not known) else 0)
    if tot > 0:
        unseen = np.array(ul, dtype=np.float32)
        if top >= 0 and not known:  # hidden-to-mover top belongs to the belief multiset
            unseen[top] += 1.0
        probs = unseen / tot
        out[o:o + NUM_KINDS] = probs
        o += NUM_KINDS
        for t in range(5):
            out[o + t] = float(probs @ _TYPE_MASKS[t])
        out[o + 5] = float(probs @ _HORN_OF_KIND)


def _global_block(out: np.ndarray, off: int, s: GameState, mover: int) -> None:
    o = off
    for t in s.faceup:
        out[o + t] += 0.5
    o += NUM_TOKEN_TYPES
    out[o:o + NUM_TOKEN_TYPES] = np.asarray(s.prog_unseen, dtype=np.float32) / 2.0
    o += NUM_TOKEN_TYPES
    out[o] = s.prog_stack / 15.0
    o += 1
    out[o + min(s.conflict, 3)] = 1.0
    o += 4
    out[o] = 1.0 if s.battle_pending else 0.0
    o += 1
    out[o + (0 if s.cat < 0 else (1 if s.cat == mover else 2))] = 1.0
    o += 3
    out[o] = 1.0 if s.knows_central(mover) else 0.0
    out[o + 1] = 1.0 if s.has_peeked(1 - mover) else 0.0
    o += 2
    out[o] = min(s.turn, 80) / 80.0
    out[o + 1] = 1.0 if mover == 0 else 0.0
    o += 2
    if s.node_type == N_DECISION:
        out[o + s.dkind] = 1.0
    o += 7
    if s.node_type == N_DECISION and s.dkind == D_PICK:
        reason, optional, _ = s.dctx
        r = reason.split(":")[0]
        out[o + (PICK_REASONS.index(r) if r in PICK_REASONS else 0)] = 1.0
        out[o + len(PICK_REASONS)] = 1.0 if optional else 0.0
    o += len(PICK_REASONS) + 1
    if s.node_type == N_DECISION and s.dkind == D_PAY:
        stage_idx, chosen = s.dctx
        val = 0
        for c in chosen:
            out[o + 1 + c] += 1.0
            val += 2 if c == PAY_COIN2_CODE else 1
        out[o] = val / 4.0
        out[o + 8 + stage_idx] = 1.0
    elif s.node_type == N_DECISION and s.dkind == D_STAGE:
        for i in s.dctx:
            out[o + 8 + i] = 0.5
    o += 1 + 7 + NUM_STAGES
    if s.node_type == N_DECISION and s.dkind == D_HALI_CHOOSE:
        _, revealed = s.dctx
        for k in revealed:
            out[o + k] += 0.25
    o += NUM_KINDS
    out[o] = 1.0 if s.wonder_done else 0.0


# --------------------------------------------------------------------------- version 2: EXPERT block
# The heuristic is imported lazily (only the version-2 path needs it), so a refactor of its private
# helpers can never break the loading / serving of version-1 checkpoints.
# The v2 EXPERT block is computed by the heuristic with these parameter values, pinned on 2026-09-24 so
# that later tuning of HeuristicParams' defaults does not shift the inputs of trained v2 networks.
# (Changes to the heuristic's code still would: treat heuristic.py edits as a feature-version change.)
EXPERT_PARAM_VALUES = {
    'w_stage_vp': 1.0,
    'tempo_bonus': 5.0,
    'res_unit': 1.0,
    'res_discount': 0.85,
    'future_stage_factor': 0.45,
    'coin_flex': 0.8,
    'late_stage_penalty': 0.4,
    'win_bonus': 60.0,
    'lose_penalty': 30.0,
    'extra_card_value': 6.0,
    'w_blue': 1.0,
    'cat_value': 0.8,
    'cat_steal': 0.8,
    'cat_peek_per_turn': 0.06,
    'sci_phi1': 0.35,
    'sci_phi2': 0.62,
    'sci_token_factor': 0.9,
    'green_early_bonus': 0.0,
    'tok_extra_scale': 0.9,
    'builds_per_turn': 0.35,
    'arch_last_pick': 1.0,
    'blind_token_discount': 1.0,
    'res_tok_freq': 0.22,
    'green_tok_freq': 0.15,
    'horn_tok_freq': 0.12,
    'tok_eng_per_stage': 0.9,
    'tok_econ_per_stage': 0.6,
    'tok_immediate_build': 0.5,
    'tactics_future': 1.0,
    'future_tokens': 0.5,
    'future_cat_cards': 0.5,
    'p_complete_leader': 0.6,
    'p_complete_trailer': 0.3,
    'culture_second_bonus': 2.0,
    'deny_token': 0.35,
    'deny_far': 1.0,
    'pb_base': 0.2,
    'pb_per_turn': 0.04,
    'pb_per_conflict': 0.2,
    'pb_max': 0.5,
    'horn_accel': 0.12,
    'mil_lookahead': 2,
    'mil_step_discount': 0.8,
    'future_battle_weight': 0.5,
    'perm_shield_bonus': 0.4,
    'early_horn_discount': 0.0,
    'battle_response': 1.0,
    'deny': 0.55,
    'deny_reveal': 0.3,
    'deny_hidden_central': -1.0,
    'pick_rate': 0.7,
    'max_turns': 16.0,
    'stall_horizon': 0.0,
    'pay_coin_penalty': 0.3,
    'pay_econ_bonus': 0.8,
    'epsilon': 0.0,
    'tie_noise': 0.001,
    'prior_temperature': 1.5,
}

_HEUR = None
EXPERT_FAILURES = 0  # decision nodes whose EXPERT block fell back to zeros because the heuristic raised


def _heuristic():
    global _HEUR
    if _HEUR is None:
        from ..agents import heuristic as h
        scorers = {D_PICK: h._score_pick, D_PAY: h._score_pay, D_SCIENCE: h._score_science, D_TOKEN: h._score_token,
                   D_HALI_DECK: h._score_hali_deck, D_HALI_CHOOSE: h._score_hali_choose, D_STAGE: h._score_stage}
        _HEUR = (h._Ctx, h.HeuristicParams(**EXPERT_PARAM_VALUES), scorers, h._maxval)
    return _HEUR


def _belief_of(s: GameState, me: int) -> GameState:
    """``s`` as player ``me`` may see it: a central top card ``me`` does not know (a peek of the other
    player, kept by an in-tree sample or an omniscient environment state) goes back into the unseen
    multiset, exactly as the version-1 central-deck block treats it."""
    top = s.deck_top[CENTRAL]
    if top < 0 or s.knows_central(me):
        return s
    c = s._copy()
    c._legal = s._legal
    c._outcomes = s._outcomes
    c.deck_top[CENTRAL] = -1
    c.unseen[CENTRAL][top] += 1  # (the public "has peeked" bits stay: knows_central() is False without a top)
    return c


# positions (within the EXPERT block, before the prior) of the entries with their own clipping range
_CV_OFF = NUM_STAGES + 3
_TOTAL_OFF = _CV_OFF + NUM_KINDS + 3
_CV_IDX = np.array([q * EXPERT_PLAYER_FEATS + _CV_OFF + k for q in (0, 1) for k in range(NUM_KINDS)], dtype=np.int64)
_TOTAL_IDX = np.array([_TOTAL_OFF, EXPERT_PLAYER_FEATS + _TOTAL_OFF], dtype=np.int64)


def _expert_player(ctx, maxval, p: int, vals: List[float]) -> None:
    v = ctx.views[p]
    w = v.w
    built = v.built
    tgs = {tg.idx: tg for tg in v.targets}
    for i in range(NUM_STAGES):
        if (built >> i) & 1:
            vals.append(0.0)
            continue
        tg = tgs.get(i)
        if tg is not None:
            d = tg.deficit
        else:  # not available yet (prerequisites missing): same payable value as a heuristic _Target
            st = w.stages[i]
            mode = 2 if v.eng else (0 if st.kind == IDENTICAL else 1)
            mv = maxval(v.grey, v.coins, v.econ, mode)
            d = st.cost - mv if mv < st.cost else 0
        vals.append(d * 0.25)
    vals.append(v.cards_needed / 14.0)
    vals.append(v.next_deficit * 0.25)
    vals.append(len(v.targets) / 3.0)
    for x in ctx.card_values(p):
        vals.append(x / 20.0)  # clipped to [-2, 4] in _expert_block
    vals.append(ctx.best_token(p) / 30.0)
    vals.append(ctx.blind_token(p) / 30.0)
    plain, total = ctx.build_value(p)
    vals.append(plain / 20.0)
    vals.append(total / 60.0)  # clipped to [-1.5, 1.5] in _expert_block
    vals.append(ctx.perm_shield_value(p) / 10.0)


def _expert_block(out: np.ndarray, s: GameState, me: int) -> None:
    """Fill ``out[EXPERT_SLICE]`` (already zero) for the decision node ``s`` seen by player ``me``."""
    if s.node_type != N_DECISION:
        return
    Ctx, params, scorers, maxval = _heuristic()
    legal = s.legal_actions() or ()  # (a hand-edited copy may carry no legal list: no prior then)
    hs = _belief_of(s, me)
    ctx = Ctx(hs, params)
    off = EXPERT_SLICE[0]
    if len(legal) >= 2:
        # the heuristic prior (== heuristic_prior(hs)) through the scorer of score_actions, on the SAME
        # context as the other expert features (its lazily cached card / token / build values are shared)
        scorer = scorers.get(s.dkind)
        scores = scorer(ctx, legal) if scorer is not None else [0.0] * len(legal)
        temp = params.prior_temperature if params.prior_temperature > 0.0 else 1e-6
        sc = np.asarray(scores, dtype=np.float64) / temp
        sc -= sc.max()
        wts = np.exp(sc)
        wts /= wts.sum()
        out[EXPERT_PRIOR_SLICE[0] + np.asarray(legal, dtype=np.int64)] = wts
    vals: List[float] = []
    _expert_player(ctx, maxval, me, vals)
    _expert_player(ctx, maxval, 1 - me, vals)
    vals.append(ctx.tl / 16.0)
    vals.append(ctx.pb)
    vals.append(ctx.nb * 0.5)
    vals.append(ctx.sci_time)
    blk = np.asarray(vals, dtype=np.float32)
    blk[_CV_IDX] = np.clip(blk[_CV_IDX], -2.0, 4.0)
    blk[_TOTAL_IDX] = np.clip(blk[_TOTAL_IDX], -1.5, 1.5)
    np.clip(blk, -3.0, 4.0, out=blk)  # safety net: every entry stays in [-3, 4]
    out[off:EXPERT_PRIOR_SLICE[0]] = blk


def _expert_safe(out: np.ndarray, s: GameState, me: int) -> None:
    """``_expert_block`` that never raises: on any heuristic failure the block stays zero (as for a
    chance / terminal node), a warning is emitted once and ``EXPERT_FAILURES`` counts the event."""
    global EXPERT_FAILURES
    try:
        _expert_block(out, s, me)
        if not np.isfinite(out[EXPERT_SLICE[0]:EXPERT_SLICE[1]]).all():
            raise FloatingPointError("non-finite EXPERT feature")
    except Exception as e:  # noqa: BLE001 - encoding must never raise
        out[EXPERT_SLICE[0]:EXPERT_SLICE[1]] = 0.0
        EXPERT_FAILURES += 1
        if EXPERT_FAILURES == 1:
            warnings.warn(f"EXPERT feature block fell back to zeros ({type(e).__name__}: {e})", RuntimeWarning)


# --------------------------------------------------------------------------- public API
def encode_state(s: GameState, player: Optional[int] = None, version: int = FEATURE_VERSION_LATEST) -> np.ndarray:
    """Encode ``s`` from the perspective of ``player`` (default: the player to move) with encoding
    ``version`` (default: the latest; a network needs the version of its ``cfg.input_dim``, see
    :func:`encoder_for_dim`)."""
    mover = s.mover if player is None else player
    n = feature_size(version)  # validates ``version``
    out = np.zeros(n, dtype=np.float32)
    _player_block(out, ME_SLICE[0], s, mover, True)
    _player_block(out, OPP_SLICE[0], s, 1 - mover, False)
    _deck_block(out, DECK_OWN_SLICE[0], s, mover, mover)
    _deck_block(out, DECK_OPP_SLICE[0], s, 1 - mover, mover)
    _deck_block(out, DECK_CENTER_SLICE[0], s, CENTRAL, mover)
    _global_block(out, GLOBAL_SLICE[0], s, mover)
    if n > FEATURE_SIZE_V1:
        _expert_safe(out, s, mover)
    return out


def legal_mask(s: GameState) -> np.ndarray:
    m = np.zeros(A.NUM, dtype=bool)
    for a in s.legal_actions():
        m[a] = True
    return m


def encode(s: GameState, version: int = FEATURE_VERSION_LATEST) -> Tuple[np.ndarray, np.ndarray]:
    return encode_state(s, version=version), legal_mask(s)


# fixed-version entry points (picklable top-level functions for evaluators and workers)
def encode_state_v1(s: GameState, player: Optional[int] = None) -> np.ndarray:
    return encode_state(s, player, 1)


def encode_state_v2(s: GameState, player: Optional[int] = None) -> np.ndarray:
    return encode_state(s, player, 2)


def encode_v1(s: GameState) -> Tuple[np.ndarray, np.ndarray]:
    return encode_state(s, None, 1), legal_mask(s)


def encode_v2(s: GameState) -> Tuple[np.ndarray, np.ndarray]:
    return encode_state(s, None, 2), legal_mask(s)


EncodeFn = Callable[[GameState], Tuple[np.ndarray, np.ndarray]]
EncodeStateFn = Callable[..., np.ndarray]
_ENCODERS = {1: (encode_v1, encode_state_v1), 2: (encode_v2, encode_state_v2)}
assert set(_ENCODERS) == set(FEATURE_SIZES)


def encoders(version: int = FEATURE_VERSION_LATEST) -> Tuple[EncodeFn, EncodeStateFn]:
    """``(encode, encode_state)`` of encoding ``version``."""
    feature_size(version)  # validates
    return _ENCODERS[int(version)]


def encoder_for_dim(input_dim: int) -> Tuple[int, EncodeFn, EncodeStateFn]:
    """``(version, encode, encode_state)`` for a network with ``input_dim`` inputs (``net.cfg.input_dim``);
    raises ``ValueError`` for a size no encoding produces."""
    version = version_for_dim(int(input_dim))
    enc, enc_state = _ENCODERS[version]
    return version, enc, enc_state
