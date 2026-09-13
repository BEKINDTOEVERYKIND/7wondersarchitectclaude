"""State -> fixed-size feature vector (mover-centric) and legal-action mask.

Layout (all values roughly in [0, 1]):

* ``me`` block, ``opp`` block            – wonder, stages, next stage, tableau, science, tokens, military
* ``deck_own``, ``deck_opp``, ``deck_center`` – size, visible top card, unseen-multiset histogram
* ``global`` block                         – progress tokens, conflict, cat, decision context

``ENTITY_SLICES`` exposes the block boundaries for the attention trunk.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from ..engine.actions import Actions as A
from ..engine.cards import BLUE, GREEN, GREY, KINDS, NUM_KINDS, RED, YELLOW, KIND_OF_SYMBOL, COIN_KIND, KIND_OF_RESOURCE
from ..engine.state import (CENTRAL, D_HALI_CHOOSE, D_PAY, D_PICK, GameState, N_DECISION, PAY_COIN2_CODE, PAY_COIN_CODE)
from ..engine.tokens import NUM_TOKEN_TYPES, TOKENS
from ..engine.wonders import EFFECTS, NUM_WONDERS, WONDERS

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

PLAYER_FEATS = (NUM_WONDERS + 6 + (3 + 2 + 1 + len(EFFECTS)) + 1 + NUM_KINDS * 2 + 4 + 2 + 3 + 2
                + NUM_TOKEN_TYPES * 2 + 3 + 1 + 1)
DECK_FEATS = 3 + KIND_FEATS + NUM_KINDS + 6
PICK_REASONS = ["main", "token", "alexandria", "ephesus", "hali", "olympia"]
GLOBAL_FEATS = (NUM_TOKEN_TYPES * 2 + 1) + (4 + 1) + 3 + 2 + 2 + 6 + (len(PICK_REASONS) + 1) + (1 + 7) + NUM_KINDS + 1

ME_SLICE = (0, PLAYER_FEATS)
OPP_SLICE = (PLAYER_FEATS, 2 * PLAYER_FEATS)
DECK_OWN_SLICE = (2 * PLAYER_FEATS, 2 * PLAYER_FEATS + DECK_FEATS)
DECK_OPP_SLICE = (DECK_OWN_SLICE[1], DECK_OWN_SLICE[1] + DECK_FEATS)
DECK_CENTER_SLICE = (DECK_OPP_SLICE[1], DECK_OPP_SLICE[1] + DECK_FEATS)
GLOBAL_SLICE = (DECK_CENTER_SLICE[1], DECK_CENTER_SLICE[1] + GLOBAL_FEATS)
FEATURE_SIZE = GLOBAL_SLICE[1]
ENTITY_SLICES: List[Tuple[int, int]] = [ME_SLICE, OPP_SLICE, DECK_OWN_SLICE, DECK_OPP_SLICE, DECK_CENTER_SLICE, GLOBAL_SLICE]


def _player_block(out: np.ndarray, off: int, s: GameState, p: int, is_mover: bool) -> None:
    w = WONDERS[s.wonder[p]]
    out[off + s.wonder[p]] = 1.0
    o = off + NUM_WONDERS
    st = s.stages[p]
    out[o + st] = 1.0
    o += 6
    if st < 5:
        stage = w.stages[st]
        out[o + min(stage.cost, 4) - 2] = 1.0
        out[o + 3 + stage.kind] = 1.0
        out[o + 5] = stage.vp / 10.0
        out[o + 6 + _EFFECT_INDEX[stage.effect]] = 1.0
    o += 3 + 2 + 1 + len(EFFECTS)
    out[o] = (5 - st) / 5.0
    o += 1
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
    out[o + 1] = 1.0 if s.knows_central(1 - mover) else 0.0
    o += 2
    out[o] = min(s.turn, 80) / 80.0
    out[o + 1] = 1.0 if mover == 0 else 0.0
    o += 2
    if s.node_type == N_DECISION:
        out[o + s.dkind] = 1.0
    o += 6
    if s.node_type == N_DECISION and s.dkind == D_PICK:
        reason, optional, _ = s.dctx
        r = reason.split(":")[0]
        out[o + (PICK_REASONS.index(r) if r in PICK_REASONS else 0)] = 1.0
        out[o + len(PICK_REASONS)] = 1.0 if optional else 0.0
    o += len(PICK_REASONS) + 1
    if s.node_type == N_DECISION and s.dkind == D_PAY:
        chosen = s.dctx
        val = 0
        for c in chosen:
            out[o + 1 + c] += 1.0
            val += 2 if c == PAY_COIN2_CODE else 1
        out[o] = val / 4.0
    o += 1 + 7
    if s.node_type == N_DECISION and s.dkind == D_HALI_CHOOSE:
        _, revealed = s.dctx
        for k in revealed:
            out[o + k] += 0.25
    o += NUM_KINDS
    out[o] = 1.0 if s.wonder_done else 0.0


def encode_state(s: GameState, player: int = None) -> np.ndarray:
    """Encode ``s`` from the perspective of ``player`` (default: the player to move)."""
    mover = s.mover if player is None else player
    out = np.zeros(FEATURE_SIZE, dtype=np.float32)
    _player_block(out, ME_SLICE[0], s, mover, True)
    _player_block(out, OPP_SLICE[0], s, 1 - mover, False)
    _deck_block(out, DECK_OWN_SLICE[0], s, mover, mover)
    _deck_block(out, DECK_OPP_SLICE[0], s, 1 - mover, mover)
    _deck_block(out, DECK_CENTER_SLICE[0], s, CENTRAL, mover)
    _global_block(out, GLOBAL_SLICE[0], s, mover)
    return out


def legal_mask(s: GameState) -> np.ndarray:
    m = np.zeros(A.NUM, dtype=bool)
    for a in s.legal_actions():
        m[a] = True
    return m


def encode(s: GameState) -> Tuple[np.ndarray, np.ndarray]:
    return encode_state(s), legal_mask(s)
