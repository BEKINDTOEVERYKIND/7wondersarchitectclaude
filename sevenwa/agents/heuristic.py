"""Hand-crafted expert policy for 2-player *7 Wonders: Architects*.

The policy is a one-ply **value-scoring** rule: every legal action of the current engine
decision is given a score in (approximate) victory-point units and the best one is played.
It is deliberately cheap (no copies of the state, no search) so it can serve as the playout
policy of rollout-MCTS and as a prior for the tree search without a network.

Card values (``_card_values``) for a player combine:

* **Wonder progress** – does the card reduce the number of cards still missing for the
  next stage (identical/different requirement, coins as wilds, Engineering/Economy
  tokens)?  A card that *completes* the stage is worth the stage (VP + tempo + effect);
  the 5th stage ends the game and is scored as a win/loss.  A card that only fits the
  *following* stage (given the cards the current stage will consume) gets a smaller value.
* **Blue points** and the Cat (2 VP, the peek, stealing it from the opponent, Politics).
* **Science** – a potential function over the held symbols scaled by the value of the best
  Progress token currently obtainable (face-up or blind).
* **Military** – the 2-player battle rule (1 token if ahead, 2 if ≥ 2×; 1-vs-0 gives one)
  evaluated before/after the card, weighted by how likely/close the battle is
  (``state.conflict`` vs ``rules.conflict_tokens``); horns accelerate the battle, which is
  good when ahead and bad when behind; hornless shields are permanent.
* **Extra picks** granted by Progress tokens (Urbanism, Crafts, Jewellery, Science,
  Propaganda, Architecture) and by Wonder-stage effects.

For a main pick the score of an option is ``my_value(card) - best_reply`` where
``best_reply`` is the (weighted) value, *for the opponent*, of the best card they will have
access to after my move: ``deny`` × a visible top card, or ``deny_reveal`` × the expected
value over the unseen multiset of a deck whose top I take (and of the blind central card).
This single term implements **denial** (taking the card that completes the opponent's
stage / science set / battle) and also makes the policy avoid revealing good cards
needlessly.  A pick that completes my own 5th stage ends the game, so no reply is charged.

Payment decisions (one card at a time, in the engine's canonical non-decreasing code order)
score each option by the best legal *completion* of the payment it still allows, valued by the
holdings that completion leaves for the next stages (``_score_pay``).  Science-set, token and
Halicarnassus decisions use the same value tables.  All weights live in
:class:`HeuristicParams` so they can be tuned later.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..engine.actions import Actions as A
from ..engine.cards import BLUE, COIN_KIND, KINDS, KIND_OF_RESOURCE, KIND_OF_SYMBOL, NUM_KINDS, RED
from ..engine.state import (CENTRAL, D_HALI_CHOOSE, D_HALI_DECK, D_PAY, D_PICK, D_SCIENCE, D_TOKEN, GameState)
from ..engine.tokens import (K_CULTURE, K_ECONOMY, K_ENGINEERING, K_EXTRA_ON_BUILD, K_EXTRA_ON_GREEN,
                             K_EXTRA_ON_HORN, K_EXTRA_ON_RESOURCE, K_SHIELDS, K_VP_CAT, K_VP_MIL, K_VP_PROG,
                             K_VP_WONDER, NUM_TOKEN_TYPES, TOKENS)
from ..engine.wonders import (E_ANY_DECK, E_CENTRAL, E_LEFT_RIGHT, E_LOOK5, E_SHIELD, E_TOKEN, IDENTICAL, WONDERS)

# --------------------------------------------------------------------------- static tables
_KVP = [k.vp for k in KINDS]
_KHORN = [k.horns for k in KINDS]
_KSHIELD = [k.shields for k in KINDS]
_RES_KIND = [KIND_OF_RESOURCE[r] for r in range(5)]      # resource index -> kind id
_SYM_KIND = [KIND_OF_SYMBOL[s] for s in range(3)]        # symbol index -> kind id
_BLUE_KINDS = [k.id for k in KINDS if k.type == BLUE]
_CAT_KINDS = [k.id for k in KINDS if k.type == BLUE and k.cat]
_RED_KINDS = [k.id for k in KINDS if k.type == RED]
_HORN_KINDS = [k.id for k in KINDS if k.type == RED and k.horns > 0]

_TOK_BY_KIND = {}
for _t in TOKENS:
    _TOK_BY_KIND.setdefault(_t.kind, []).append(_t.id)
_ARCH_IDS = _TOK_BY_KIND.get(K_EXTRA_ON_BUILD, [])
_PROP_IDS = _TOK_BY_KIND.get(K_EXTRA_ON_HORN, [])
_SCI_IDS = _TOK_BY_KIND.get(K_EXTRA_ON_GREEN, [])
_ENG_IDS = _TOK_BY_KIND.get(K_ENGINEERING, [])
_ECON_IDS = _TOK_BY_KIND.get(K_ECONOMY, [])
_TACT_IDS = _TOK_BY_KIND.get(K_SHIELDS, [])
_STRAT_IDS = _TOK_BY_KIND.get(K_VP_MIL, [])
_EDU_IDS = _TOK_BY_KIND.get(K_VP_PROG, [])
_DECOR_IDS = _TOK_BY_KIND.get(K_VP_WONDER, [])
_POL_IDS = _TOK_BY_KIND.get(K_VP_CAT, [])
_RESTOK_FOR_RES: List[List[int]] = [[t.id for t in TOKENS if t.kind == K_EXTRA_ON_RESOURCE and r in t.resources]
                                    for r in range(5)]
_RESTOK_FOR_COIN: List[int] = [t.id for t in TOKENS if t.kind == K_EXTRA_ON_RESOURCE and t.triggers_on_coin]
_STRAT_VP = sum(TOKENS[t].vp for t in _STRAT_IDS)
_EDU_VP = sum(TOKENS[t].vp for t in _EDU_IDS)
_POL_VP = sum(TOKENS[t].vp for t in _POL_IDS)
_DECOR_BONUS = sum(TOKENS[t].vp_complete - TOKENS[t].vp_incomplete for t in _DECOR_IDS)

# per wonder: cumulative stage VP (index i = VP of stages < i) and remaining cost (index i = cost of stages >= i)
_STAGE_VP_CUM: List[List[int]] = []
_REMAIN_COST: List[List[int]] = []
for _w in WONDERS:
    cum = [0]
    for _s in _w.stages:
        cum.append(cum[-1] + _s.vp)
    _STAGE_VP_CUM.append(cum)
    rem = [0] * 6
    for _i in range(4, -1, -1):
        rem[_i] = rem[_i + 1] + _w.stages[_i].cost
    _REMAIN_COST.append(rem)

_NEG = -1e9


# --------------------------------------------------------------------------- parameters
@dataclass
class HeuristicParams:
    """Tunable weights of the expert policy (all in approximate VP units)."""

    # --- wonder construction
    w_stage_vp: float = 1.0          # weight of the printed stage VP
    tempo_bonus: float = 3.0         # extra value of completing any stage (race / engine effects)
    res_unit: float = 1.0            # on-plan resource value = res_unit * stage_value / cost
    res_discount: float = 0.85       # per extra card still missing after this one
    future_stage_factor: float = 0.45  # value of a card that only fits the *following* stage
    coin_flex: float = 0.4           # bonus for coins (wild resource)
    late_stage_penalty: float = 0.4  # multiplier when the stage cannot be finished in the remaining turns
    win_bonus: float = 60.0          # completing the 5th stage while ahead (game ends, we win)
    lose_penalty: float = 30.0       # completing the 5th stage while behind (game ends, we lose)
    extra_card_value: float = 4.0    # value of one extra pick (token / wonder effect)
    # --- blue / cat
    w_blue: float = 1.0
    cat_value: float = 0.8           # expected value of holding the Cat (2 VP, may be stolen)
    cat_steal: float = 0.8           # extra when the Cat is taken from the opponent
    cat_peek_per_turn: float = 0.06  # information value of the peek per remaining turn
    # --- science
    sci_phi1: float = 0.35           # potential of one symbol (fraction of a token)
    sci_phi2: float = 0.62           # potential of two different symbols
    sci_token_factor: float = 0.9    # discount of the best available token (it may be gone later)
    # --- progress tokens
    tok_extra_scale: float = 0.6     # scale of extra-card tokens (uncertainty / time)
    builds_per_turn: float = 0.35    # expected stage completions per remaining turn
    res_tok_freq: float = 0.22       # fraction of future picks triggering a resource token
    green_tok_freq: float = 0.15
    horn_tok_freq: float = 0.12
    tok_eng_per_stage: float = 0.9
    tok_econ_per_stage: float = 0.6
    tok_immediate_build: float = 0.5  # fraction of the build value when a token enables a build right now
    tactics_future: float = 1.0
    future_tokens: float = 0.5       # expected further tokens (Education)
    future_cat_cards: float = 0.5    # expected further cat cards (Politics)
    p_complete_leader: float = 0.6   # P(wonder complete at the end) when leading the race (Decor)
    p_complete_trailer: float = 0.3
    culture_second_bonus: float = 2.0
    deny_token: float = 0.35
    # --- military
    pb_base: float = 0.2             # P(battle happens) = base + per_turn*turns_left + per_conflict*conflict
    pb_per_turn: float = 0.04
    pb_per_conflict: float = 0.2
    pb_max: float = 0.9
    horn_accel: float = 0.12         # extra battle probability per horn
    mil_lookahead: int = 3           # a shield may be the first of k needed to change the battle outcome
    mil_step_discount: float = 0.8   # per additional shield still needed
    future_battle_weight: float = 0.5  # weight of battles after the next one (permanent shields only)
    perm_shield_bonus: float = 0.4   # hornless shields survive battles
    # --- denial / lookahead
    deny: float = 0.55               # weight of the opponent's best reply (visible cards)
    deny_reveal: float = 0.3         # weight of the expected value of the card my pick reveals to them
    pick_rate: float = 0.7           # useful resource cards per turn (game-length estimate)
    max_turns: float = 16.0
    # --- payment
    pay_coin_penalty: float = 0.3    # keep wild coins when possible
    pay_econ_bonus: float = 0.8      # spending a doubled coin saves a card
    # --- randomness
    epsilon: float = 0.0             # epsilon-greedy noise in heuristic_action
    tie_noise: float = 1e-3          # random tie-breaking noise
    prior_temperature: float = 1.5   # softmax temperature (VP units) of heuristic_prior


_DEFAULT = HeuristicParams()


# --------------------------------------------------------------------------- per-player view
class _View:
    __slots__ = ("p", "grey", "coins", "toks", "used", "eng", "econ", "n_tokens", "sci", "shields", "perm_shields", "mil",
                 "score", "cat_cards", "stage_idx", "cost", "vp", "effect", "mode", "max_grey", "deficit",
                 "left_grey", "left_coins", "left_max", "next_cost", "next_mode", "next_deficit", "cards_needed",
                 "strat", "edu", "has_decor", "has_pol")


def _maxval(grey: List[int], coins: int, econ: bool, mode: int) -> int:
    """Maximum resource value payable for one stage (mirrors the engine's ``_max_value``)."""
    cap = coins + (1 if (econ and coins >= 1) else 0)
    if mode == 0:
        return max(grey) + cap
    if mode == 1:
        return (grey[0] > 0) + (grey[1] > 0) + (grey[2] > 0) + (grey[3] > 0) + (grey[4] > 0) + cap
    return grey[0] + grey[1] + grey[2] + grey[3] + grey[4] + cap


def _leftover(grey: List[int], coins: int, cost: int, mode: int) -> Tuple[List[int], int]:
    """Holdings after a hypothetical payment of the current stage.

    Assumes the cheapest-to-replace cards are spent: for an identical stage the largest
    stack, for a different stage one card of each of the *smallest* stacks (keeping big
    stacks for the following identical stage), coins last.  Missing cards are assumed to
    arrive later and are ignored.
    """
    lg = list(grey)
    need = cost
    if mode == 0:
        b = 0
        for r in range(1, 5):
            if lg[r] > lg[b]:
                b = r
        u = lg[b] if lg[b] < need else need
        lg[b] -= u
        need -= u
    elif mode == 1:
        order = sorted((c, r) for r, c in enumerate(lg) if c > 0)
        for _c, r in order:
            if need == 0:
                break
            lg[r] -= 1
            need -= 1
    else:
        order = sorted((c, r) for r, c in enumerate(lg) if c > 0)
        for _c, r in order:
            if need == 0:
                break
            u = lg[r] if lg[r] < need else need
            lg[r] -= u
            need -= u
    lc = coins
    u = lc if lc < need else need
    lc -= u
    return lg, lc


def _make_view(s: GameState, p: int) -> _View:
    v = _View()
    v.p = p
    cards = s.cards[p]
    grey = [cards[_RES_KIND[0]], cards[_RES_KIND[1]], cards[_RES_KIND[2]], cards[_RES_KIND[3]], cards[_RES_KIND[4]]]
    v.grey = grey
    v.coins = cards[COIN_KIND]
    toks = s.tokens[p]
    v.toks = toks
    is_mover = p == s.mover
    v.used = s.tokens_used if is_mover else 0
    v.eng = any(toks[t] for t in _ENG_IDS)
    v.econ = any(toks[t] for t in _ECON_IDS) and not (is_mover and s.econ_used)
    n_tok = 0
    for c in toks:
        n_tok += c
    v.n_tokens = n_tok
    v.strat = _STRAT_VP if any(toks[t] for t in _STRAT_IDS) else 0
    v.edu = _EDU_VP if any(toks[t] for t in _EDU_IDS) else 0
    v.has_decor = any(toks[t] for t in _DECOR_IDS)
    v.has_pol = any(toks[t] for t in _POL_IDS)
    v.sci = [cards[_SYM_KIND[0]], cards[_SYM_KIND[1]], cards[_SYM_KIND[2]]]
    perm = s.wonder_shields[p]
    for t in _TACT_IDS:
        perm += toks[t] * TOKENS[t].shields
    sh = perm
    for k in _RED_KINDS:
        c = cards[k]
        if c:
            sh += c * _KSHIELD[k]
            if _KHORN[k] == 0:
                perm += c * _KSHIELD[k]
    v.shields = sh
    v.perm_shields = perm
    v.mil = s.mil_tokens[p]
    cat_cards = 0
    for k in _CAT_KINDS:
        cat_cards += cards[k]
    v.cat_cards = cat_cards
    # score (same bookkeeping as GameState.score_of, inlined for speed)
    w = s.wonder[p]
    si = s.stages[p]
    score = _STAGE_VP_CUM[w][si]
    for k in _BLUE_KINDS:
        score += cards[k] * _KVP[k]
    score += v.mil * s.rules.military_token_vp
    if n_tok:
        score += s.token_vp(p)
    if s.cat == p:
        score += s.rules.cat_vp
    v.score = score
    v.stage_idx = si
    if si < 5:
        st = WONDERS[w].stages[si]
        cost = st.cost
        v.cost = cost
        v.vp = st.vp
        v.effect = st.effect
        mode = 2 if v.eng else (0 if st.kind == IDENTICAL else 1)
        v.mode = mode
        v.max_grey = max(grey)
        mv = _maxval(grey, v.coins, v.econ, mode)
        v.deficit = cost - mv if mv < cost else 0
        lg, lc = _leftover(grey, v.coins, cost, mode)
        v.left_grey = lg
        v.left_coins = lc
        v.left_max = max(lg)
        if si + 1 < 5:
            nst = WONDERS[w].stages[si + 1]
            nmode = 2 if v.eng else (0 if nst.kind == IDENTICAL else 1)
            v.next_cost = nst.cost
            v.next_mode = nmode
            nmv = _maxval(lg, lc, v.econ, nmode)
            v.next_deficit = nst.cost - nmv if nmv < nst.cost else 0
        else:
            v.next_cost = 0
            v.next_mode = -1
            v.next_deficit = 0
        v.cards_needed = v.deficit + _REMAIN_COST[w][si + 1]
    else:
        v.cost = 0
        v.vp = 0
        v.effect = ""
        v.mode = 2
        v.max_grey = max(grey)
        v.deficit = 0
        v.left_grey = grey
        v.left_coins = v.coins
        v.left_max = v.max_grey
        v.next_cost = 0
        v.next_mode = -1
        v.next_deficit = 0
        v.cards_needed = 0
    return v


# --------------------------------------------------------------------------- military helpers
def _mil_tokens(a: int, b: int, rules) -> int:
    """Military tokens won by a player with ``a`` shields against ``b`` (2-player rule)."""
    if a > b:
        if a >= 2 * b and not (rules.double_vs_zero_requires_two and b == 0 and a < 2):
            return 2
        return 1
    return 0


def _mil_diff(v: _View, o: _View, a: int, b: int, rules) -> float:
    """VP difference (mine - theirs) produced by a battle at ``a`` vs ``b`` shields."""
    tm = _mil_tokens(a, b, rules)
    to = _mil_tokens(b, a, rules)
    return rules.military_token_vp * (tm - to) + v.strat * tm - o.strat * to


def _mil_gain(v: _View, o: _View, a: int, b: int, rules, kmax: int, disc: float) -> float:
    """Per-shield value of adding shields at ``a`` vs ``b``.

    Because of the 2-player "at least twice as many" rule a single shield often changes
    nothing while two or three would; the gain is therefore the best of
    ``(outcome(a + k) - outcome(a)) / k`` over ``k = 1..kmax`` (later shields discounted).
    """
    base = _mil_diff(v, o, a, b, rules)
    best = _mil_diff(v, o, a + 1, b, rules) - base
    f = disc
    for k in range(2, kmax + 1):
        g = (_mil_diff(v, o, a + k, b, rules) - base) / k * f
        if g > best:
            best = g
        f *= disc
    return best


# --------------------------------------------------------------------------- science potential
def _phi(c0: int, c1: int, c2: int, T: float, phi1: float, phi2: float) -> float:
    """Value of a science holding: completed sets are worth ``T`` each, partial sets a fraction."""
    # pairs first
    sets = (c0 >> 1) + (c1 >> 1) + (c2 >> 1)
    r0, r1, r2 = c0 & 1, c1 & 1, c2 & 1
    if r0 and r1 and r2:
        sets += 1
        n = 0
    else:
        n = r0 + r1 + r2
    best = sets * T + (phi1 if n == 1 else (phi2 if n == 2 else 0.0))
    if c0 and c1 and c2:  # triple first
        d0, d1, d2 = c0 - 1, c1 - 1, c2 - 1
        sets = 1 + (d0 >> 1) + (d1 >> 1) + (d2 >> 1)
        r0, r1, r2 = d0 & 1, d1 & 1, d2 & 1
        if r0 and r1 and r2:
            sets += 1
            n = 0
        else:
            n = r0 + r1 + r2
        alt = sets * T + (phi1 if n == 1 else (phi2 if n == 2 else 0.0))
        if alt > best:
            best = alt
    return best


# --------------------------------------------------------------------------- evaluation context
class _Ctx:
    """Everything the scorers need, computed lazily once per decision."""

    __slots__ = ("s", "P", "views", "tl", "pb", "nb", "sci_time", "_tokvals", "_blind", "_T", "_tok_ready",
                 "_cardvals", "_build", "_ends")

    def __init__(self, s: GameState, P: HeuristicParams):
        self.s = s
        self.P = P
        v0 = _make_view(s, 0)
        v1 = _make_view(s, 1)
        self.views = (v0, v1)
        rate = P.pick_rate
        tl = min(v0.cards_needed, v1.cards_needed) / rate
        cards_left = (s.deck_size[0] + s.deck_size[1] + s.deck_size[2] + 1) * 0.5
        if cards_left < tl:
            tl = cards_left
        if tl < 1.0:
            tl = 1.0
        elif tl > P.max_turns:
            tl = P.max_turns
        self.tl = tl
        # probability that a battle will still be fought
        if s.battle_pending:
            pb = 1.0
        else:
            horns_left = 0
            for d in range(3):
                un = s.unseen[d]
                for k in _HORN_KINDS:
                    horns_left += un[k]
                t = s.deck_top[d]
                if t >= 0 and _KHORN[t] > 0:
                    horns_left += 1
            if horns_left == 0:
                pb = 0.0
            else:
                pb = P.pb_base + P.pb_per_turn * tl + P.pb_per_conflict * s.conflict
                if pb > P.pb_max:
                    pb = P.pb_max
        self.pb = pb
        # weight of the battles *after* the next one (only permanent shields count there)
        nb = tl / 6.0
        if nb > 2.0:
            nb = 2.0
        self.nb = P.future_battle_weight * nb if pb > 0.0 else 0.0
        st = (tl + 1.0) / 5.0
        self.sci_time = 1.0 if st > 1.0 else st
        self._tokvals: List[Optional[List[float]]] = [None, None]
        self._blind: List[float] = [0.0, 0.0]
        self._T: List[float] = [0.0, 0.0]
        self._tok_ready = [False, False]
        self._cardvals: List[Optional[List[float]]] = [None, None]
        self._build: List[Optional[Tuple[float, float]]] = [None, None]
        self._ends: List[Optional[List[bool]]] = [None, None]

    # ---- stage construction ------------------------------------------------------------
    def build_value(self, p: int) -> Tuple[float, float]:
        """``(plain, total)`` value of completing player ``p``'s current stage right now.

        ``plain`` = stage VP + tempo + stage effect (+ Architecture pick); ``total`` adds the
        win bonus / loss penalty when it is the 5th stage (the game ends at once).
        """
        b = self._build[p]
        if b is not None:
            return b
        s, P = self.s, self.P
        v = self.views[p]
        o = self.views[1 - p]
        if v.stage_idx >= 5:
            b = (0.0, 0.0)
            self._build[p] = b
            return b
        extra = P.extra_card_value
        eff = v.effect
        if eff == E_SHIELD:
            effv = self.perm_shield_value(p)
        elif eff == E_ANY_DECK:
            effv = extra
        elif eff == E_CENTRAL:
            effv = extra * 0.85
        elif eff == E_LEFT_RIGHT:
            effv = extra * 1.6
        elif eff == E_LOOK5:
            effv = extra * 1.3
        elif eff == E_TOKEN:
            effv = self.best_token(p)
        else:
            effv = 0.0
        plain = P.w_stage_vp * v.vp + P.tempo_bonus + effv
        for t in _ARCH_IDS:
            if v.toks[t] and not (v.used >> t) & 1:
                plain += extra
        total = plain
        if v.stage_idx == 4:
            final = v.score + v.vp + (_DECOR_BONUS if v.has_decor else 0)
            if final > o.score or (final == o.score and s.rules.tiebreak_stages):
                total += P.win_bonus
            else:
                total -= P.lose_penalty
        b = (plain, total)
        self._build[p] = b
        return b

    def ends_game(self, p: int) -> Optional[List[bool]]:
        """Per kind: does taking that card complete player ``p``'s 5th stage (None if not at stage 5)."""
        self.card_values(p)
        return self._ends[p]

    # ---- progress tokens ---------------------------------------------------------------
    def token_values(self, p: int) -> List[float]:
        tv = self._tokvals[p]
        if tv is not None:
            return tv
        s, P = self.s, self.P
        v = self.views[p]
        o = self.views[1 - p]
        rules = s.rules
        tl = self.tl
        pb = self.pb
        extra = P.extra_card_value * P.tok_extra_scale
        stages_left = 5 - v.stage_idx
        fb = P.builds_per_turn * tl
        if fb > stages_left:
            fb = float(stages_left)
        edu = float(v.edu)
        sm, so = v.shields, o.shields
        out = [0.0] * NUM_TOKEN_TYPES
        for t in TOKENS:
            kind = t.kind
            if kind == K_EXTRA_ON_BUILD:
                val = extra * fb
            elif kind == K_EXTRA_ON_RESOURCE:
                val = extra * P.res_tok_freq * tl
            elif kind == K_EXTRA_ON_GREEN:
                freq = P.green_tok_freq + (0.05 if (v.sci[0] or v.sci[1] or v.sci[2]) else 0.0)
                val = extra * freq * tl
            elif kind == K_EXTRA_ON_HORN:
                val = extra * P.horn_tok_freq * tl
            elif kind == K_ENGINEERING:
                val = P.tok_eng_per_stage * fb
            elif kind == K_ECONOMY:
                val = P.tok_econ_per_stage * fb + 0.3 * v.coins
            elif kind == K_SHIELDS:
                val = self.perm_shield_value(p, t.shields) + P.tactics_future
            elif kind == K_VP_MIL:
                val = t.vp * (v.mil + pb * _mil_tokens(sm, so, rules))
            elif kind == K_VP_PROG:
                val = t.vp * (v.n_tokens + 1 + P.future_tokens)
            elif kind == K_VP_WONDER:
                pc = P.p_complete_leader if v.cards_needed <= o.cards_needed else P.p_complete_trailer
                val = t.vp_incomplete + (t.vp_complete - t.vp_incomplete) * pc
            elif kind == K_VP_CAT:
                val = t.vp * (v.cat_cards + P.future_cat_cards)
            elif kind == K_CULTURE:
                if v.toks[t.id] >= 1:
                    val = float(t.vp_both - t.vp_one)
                else:
                    val = float(t.vp_one)
                    if o.toks[t.id] == 0 and (s.prog_unseen[t.id] > 0 or t.id in s.faceup):
                        val += P.culture_second_bonus
            else:  # pragma: no cover - unknown token kind
                val = 1.0
            out[t.id] = val + edu
        self._tokvals[p] = out
        # blind expectation and best obtainable token
        blind = 0.0
        if s.prog_stack > 0:
            un = s.prog_unseen
            acc = 0.0
            for t in range(NUM_TOKEN_TYPES):
                c = un[t]
                if c:
                    acc += out[t] * c
            blind = acc / s.prog_stack
        self._blind[p] = blind
        best = blind if s.prog_stack > 0 else 0.0
        for t in s.faceup:
            if out[t] > best:
                best = out[t]
        self._T[p] = best * P.sci_token_factor
        self._tok_ready[p] = True
        # a token that makes the current stage affordable right now triggers the mandatory build
        # (applied after T/blind are published so build_value -> best_token cannot recurse)
        if v.stage_idx < 5 and v.deficit > 0:
            imm = None
            if not v.eng and _maxval(v.grey, v.coins, v.econ, 2) >= v.cost:
                imm = P.tok_immediate_build * self.build_value(p)[1]
                for t in _ENG_IDS:
                    out[t] += imm
            if not v.econ and v.coins >= 1 and v.deficit == 1:
                if imm is None:
                    imm = P.tok_immediate_build * self.build_value(p)[1]
                for t in _ECON_IDS:
                    out[t] += imm
        return out

    def perm_shield_value(self, p: int, n: int = 1) -> float:
        """Value of ``n`` permanent shields (hornless red, Rhodes stage, Tactics) for player ``p``."""
        s, P = self.s, self.P
        v = self.views[p]
        o = self.views[1 - p]
        rules = s.rules
        pb = 1.0 if s.battle_pending else self.pb
        if n == 1:
            near = _mil_gain(v, o, v.shields, o.shields, rules, P.mil_lookahead, P.mil_step_discount)
            far = _mil_gain(v, o, v.perm_shields, o.perm_shields, rules, P.mil_lookahead, P.mil_step_discount)
        else:
            near = _mil_diff(v, o, v.shields + n, o.shields, rules) - _mil_diff(v, o, v.shields, o.shields, rules)
            far = (_mil_diff(v, o, v.perm_shields + n, o.perm_shields, rules)
                   - _mil_diff(v, o, v.perm_shields, o.perm_shields, rules))
        return pb * near + self.nb * far + P.perm_shield_bonus * n

    def blind_token(self, p: int) -> float:
        if not self._tok_ready[p]:
            self.token_values(p)
        return self._blind[p]

    def best_token(self, p: int) -> float:
        if not self._tok_ready[p]:
            self.token_values(p)
        return self._T[p]

    # ---- cards ---------------------------------------------------------------------------
    def card_values(self, p: int) -> List[float]:
        cv = self._cardvals[p]
        if cv is not None:
            return cv
        s, P = self.s, self.P
        v = self.views[p]
        o = self.views[1 - p]
        rules = s.rules
        T = self.best_token(p)
        tl = self.tl
        extra = P.extra_card_value
        toks = v.toks
        used = v.used
        vals = [0.0] * NUM_KINDS

        # ---- resources ------------------------------------------------------------------
        if v.stage_idx < 5:
            cost = v.cost
            mode = v.mode
            deficit = v.deficit
            plain, build = self.build_value(p)
            ends: Optional[List[bool]] = [False] * NUM_KINDS if v.stage_idx == 4 else None
            U1 = P.res_unit * plain / cost
            disc = P.res_discount
            late = P.pick_rate * tl
            nmode = v.next_mode
            nd = v.next_deficit
            lg = v.left_grey
            lmax = v.left_max
            for r in range(5):
                k = _RES_KIND[r]
                g = v.grey[r]
                if mode == 0:
                    d = g + 1 > v.max_grey
                elif mode == 1:
                    d = g == 0
                else:
                    d = True
                val = 0.0
                if d and deficit > 0:
                    if deficit <= 1:
                        val = build
                        if ends is not None:
                            ends[k] = True
                    else:
                        da = deficit - 1
                        val = U1 * disc ** (da - 1)
                        if da > late:
                            val *= P.late_stage_penalty
                elif nmode >= 0 and nd > 0:
                    if nmode == 0:
                        d2 = lg[r] + 1 > lmax
                    elif nmode == 1:
                        d2 = lg[r] == 0
                    else:
                        d2 = True
                    if d2:
                        tot = deficit + nd - 1
                        val = U1 * P.future_stage_factor * disc ** tot
                        if tot > late:
                            val *= P.late_stage_penalty
                for t in _RESTOK_FOR_RES[r]:
                    if toks[t] and not (used >> t) & 1:
                        val += extra
                vals[k] = val
            # coins (wild; a fresh Economy coin counts double)
            dc = 2 if (v.econ and v.coins == 0) else 1
            if deficit > 0:
                if deficit <= dc:
                    val = build
                    if ends is not None:
                        ends[COIN_KIND] = True
                else:
                    da = deficit - dc
                    val = U1 * dc * disc ** (da - 1)
                    if da > late:
                        val *= P.late_stage_penalty
                val += P.coin_flex
            else:
                val = P.coin_flex
            for t in _RESTOK_FOR_COIN:
                if toks[t] and not (used >> t) & 1:
                    val += extra
            vals[COIN_KIND] = val
            self._ends[p] = ends

        # ---- blue / cat -----------------------------------------------------------------
        if s.cat == p:
            cat_bonus = 0.0
        else:
            cat_bonus = P.cat_value + P.cat_peek_per_turn * tl
            if s.cat == 1 - p:
                cat_bonus += P.cat_steal
        pol = _POL_VP if v.has_pol else 0
        for k in _BLUE_KINDS:
            val = P.w_blue * _KVP[k]
            if KINDS[k].cat:
                val += cat_bonus + pol
            vals[k] = val

        # ---- green ----------------------------------------------------------------------
        if T > 0.0:
            st = self.sci_time
            phi1 = P.sci_phi1 * T * st
            phi2 = P.sci_phi2 * T * st
            c0, c1, c2 = v.sci
            base = _phi(c0, c1, c2, T, phi1, phi2)
            sci_extra = 0.0
            for t in _SCI_IDS:
                if toks[t] and not (used >> t) & 1:
                    sci_extra += extra
            vals[_SYM_KIND[0]] = _phi(c0 + 1, c1, c2, T, phi1, phi2) - base + sci_extra
            vals[_SYM_KIND[1]] = _phi(c0, c1 + 1, c2, T, phi1, phi2) - base + sci_extra
            vals[_SYM_KIND[2]] = _phi(c0, c1, c2 + 1, T, phi1, phi2) - base + sci_extra

        # ---- red ------------------------------------------------------------------------
        sm, so = v.shields, o.shields
        pb = self.pb
        pending = s.battle_pending
        base_o = _mil_diff(v, o, sm, so, rules)
        ntok = rules.conflict_tokens
        conflict = s.conflict
        prop = 0.0
        for t in _PROP_IDS:
            if toks[t] and not (used >> t) & 1:
                prop += extra
        gain_near = _mil_gain(v, o, sm, so, rules, P.mil_lookahead, P.mil_step_discount)
        perm_val = self.perm_shield_value(p) if pb > 0.0 else P.perm_shield_bonus
        for k in _RED_KINDS:
            h = _KHORN[k]
            if h == 0:
                vals[k] = perm_val
                continue
            o1 = _mil_diff(v, o, sm + _KSHIELD[k], so, rules)
            if pending:
                val = o1 - base_o
            elif conflict + h >= ntok:
                # the battle is fought at the end of this turn: certain outcome instead of pb-weighted
                val = o1 - pb * base_o
            else:
                pb2 = pb + h * P.horn_accel
                if pb2 > P.pb_max:
                    pb2 = P.pb_max
                val = pb * gain_near + (pb2 - pb) * o1
            vals[k] = val + prop
        self._cardvals[p] = vals
        return vals


def _knows_central(s: GameState, p: int) -> bool:
    """Does ``p`` know the central top card (Cat peek)?  Works with the bitmask engine API
    (``knows_central``) and with the older single-player ``central_known_to == p`` field."""
    f = getattr(s, "knows_central", None)
    if f is not None:
        return bool(f(p))
    return s.deck_top[CENTRAL] >= 0 and s.central_known_to == p


def _expect(vals: List[float], counts: Sequence[int]) -> Optional[float]:
    """Mean value of a card drawn uniformly from ``counts`` (None if empty)."""
    tot = 0
    acc = 0.0
    for i in range(NUM_KINDS):
        c = counts[i]
        if c:
            acc += vals[i] * c
            tot += c
    return acc / tot if tot else None


def _comb_ratio(a: int, n_total: int, n: int) -> float:
    """C(a, n) / C(n_total, n)."""
    if a < n:
        return 0.0
    r = 1.0
    for i in range(n):
        r *= (a - i) / (n_total - i)
    return r


def _expected_max(vals: List[float], top: int, unseen: Sequence[int], n: int) -> float:
    """Expected value of the best of: the visible ``top`` plus ``n`` draws from ``unseen``."""
    vtop = vals[top] if top >= 0 else _NEG
    N = 0
    for c in unseen:
        N += c
    if n > N:
        n = N
    if n <= 0:
        return vtop if top >= 0 else 0.0
    items = sorted(((vals[k], c) for k, c in enumerate(unseen) if c), reverse=True)
    e = 0.0
    prev = 0.0
    m = 0
    for val, cnt in items:
        if val <= vtop:
            break
        m += cnt
        p_ge = 1.0 - _comb_ratio(N - m, N, n)
        e += val * (p_ge - prev)
        prev = p_ge
    if vtop > _NEG:
        e += vtop * (1.0 - prev)
    return e


# --------------------------------------------------------------------------- scorers per decision kind
def _score_pick(ctx: _Ctx, legal: Sequence[int]) -> List[float]:
    s, P = ctx.s, ctx.P
    m = s.mover
    om = 1 - m
    vme = ctx.card_values(m)
    vop = ctx.card_values(om)
    ends = ctx.ends_game(m)           # per kind: my 5th stage completes -> the opponent never replies
    deny = P.deny
    deny_r = P.deny_reveal
    size = s.deck_size
    top = s.deck_top
    # my value of each source and the (weighted) opponent's value of the cards they would see
    my_src = [0.0, 0.0, 0.0]
    cont = [1.0, 1.0, 1.0]            # probability that the game continues after I take from that source
    opp_top = [_NEG, _NEG, _NEG]      # deny * opponent's value of the current top of each deck
    opp_next = [_NEG, _NEG, _NEG]     # deny_reveal * opponent's expected value of the deck after I take its top
    for d in (0, 1):
        if size[d] > 0:
            t = top[d]
            if t >= 0:
                my_src[d] = vme[t]
                opp_top[d] = deny * vop[t]
                if ends is not None and ends[t]:
                    cont[d] = 0.0
            else:  # defensive: unknown top -> expectation
                e = _expect(vme, s.unseen[d])
                my_src[d] = e if e is not None else 0.0
                e = _expect(vop, s.unseen[d])
                opp_top[d] = deny_r * e if e is not None else _NEG
            if size[d] > 1:
                e = _expect(vop, s.unseen[d])
                opp_next[d] = deny_r * e if e is not None else _NEG
    if size[CENTRAL] > 0:
        tc = top[CENTRAL]
        if tc >= 0 and _knows_central(s, m):
            my_src[CENTRAL] = vme[tc]
            opp_top[CENTRAL] = deny * vop[tc]
            if ends is not None and ends[tc]:
                cont[CENTRAL] = 0.0
            e = _expect(vop, s.unseen[CENTRAL])
            opp_next[CENTRAL] = deny_r * e if e is not None else _NEG
        else:
            un = s.unseen[CENTRAL]
            if tc >= 0:  # sampled for the opponent: treat as unseen for me
                un = list(un)
                un[tc] += 1
                opp_top[CENTRAL] = deny * vop[tc]
                e = _expect(vop, s.unseen[CENTRAL])
                opp_next[CENTRAL] = deny_r * e if e is not None else _NEG
            else:
                e = _expect(vop, un)
                opp_top[CENTRAL] = deny_r * e if e is not None else _NEG
                opp_next[CENTRAL] = opp_top[CENTRAL]
            e = _expect(vme, un)
            my_src[CENTRAL] = e if e is not None else 0.0
            if ends is not None:
                tot = 0
                n_end = 0
                for k in range(NUM_KINDS):
                    c = un[k]
                    if c:
                        tot += c
                        if ends[k]:
                            n_end += c
                if tot:
                    cont[CENTRAL] = 1.0 - n_end / tot
    out = []
    for a in legal:
        if a == A.SKIP:
            my = 0.0
            reply = max(opp_top[0], opp_top[1], opp_top[CENTRAL])
            c = 1.0
        else:
            d = m if a == A.PICK_LEFT else (om if a == A.PICK_RIGHT else CENTRAL)
            my = my_src[d]
            c = cont[d]
            reply = opp_next[d]
            for x in (0, 1, CENTRAL):
                if x != d and opp_top[x] > reply:
                    reply = opp_top[x]
        if reply <= _NEG:
            reply = 0.0
        out.append(my - c * reply)
    return out


def _score_pay(ctx: _Ctx, legal: Sequence[int]) -> List[float]:
    """Score each "pay one card" option by the best *completion* of the payment it allows.

    The engine pays in canonical (non-decreasing code) order -- wood < stone < clay < papyrus <
    glass < coin < coin x2 -- so after code ``x`` only codes ``>= x`` may follow and an option
    cannot be judged by the holdings left after that single card (a stone paid first for a
    "2 different" stage can only be followed by clay/papyrus/glass/coins, never by wood).  Every
    legal completion of the current payment is therefore enumerated (a stage costs at most 4
    cards, and the last stage returns early, so this is a handful of sequences) with the same
    rule as ``GameState._pay_options``: codes in non-decreasing order, identical / different /
    any (Engineering) resources, the coin supply, the Economy doubled coin at most once and only
    as the final card, and ``rules.coins_free_choice``.  The holdings a completion leaves are
    valued against the next stage (cards it still covers, coins as wilds, capped at its cost)
    and, with the ``future_stage_factor`` weight, against the stage after that (after the next
    stage is paid the cheapest way, as in ``_leftover``); spending a coin costs
    ``pay_coin_penalty`` (wilds are kept when possible), the doubled coin earns ``pay_econ_bonus``
    (it saves a card), and a small tie-breaker prefers spending from large stacks.  An option's
    score is the best score over its completions.
    """
    s, P = ctx.s, ctx.P
    m = s.mover
    v = ctx.views[m]
    si = v.stage_idx
    if si + 1 >= 5:  # the 5th stage ends the game: leftovers are worthless
        return [P.pay_econ_bonus if a == A.PAY_COIN2 else 0.0 for a in legal]
    chosen = s.dctx or ()
    g = list(v.grey)
    c = v.coins
    econ = v.econ
    value = 0
    ident = -1  # resource already used by this payment (identical stages: the only one allowed)
    for code in chosen:
        if code < 5:
            g[code] -= 1
            value += 1
            if ident < 0:
                ident = code
        elif code == 5:
            c -= 1
            value += 1
        else:
            c -= 1
            value += 2
            econ = False
    need = v.cost - value
    mode = v.mode
    last = chosen[-1] if chosen else -1
    min_code = last if last >= 0 else 0
    if mode == 0:
        if ident >= 0:
            lo = ident if ident >= min_code else 5  # a coin already followed the grey: no more greys
            hi = ident + 1
        else:
            lo, hi = min_code, 5
    elif mode == 1:
        lo = min_code + 1 if 0 <= last < 5 else min_code  # a resource cannot be repeated
        hi = 5
    else:
        lo, hi = min_code, 5
    w = s.wonder[m]
    nst = WONDERS[w].stages[si + 1]
    nmode = 2 if v.eng else (0 if nst.kind == IDENTICAL else 1)
    ncost = nst.cost
    if si + 2 < 5:
        st2 = WONDERS[w].stages[si + 2]
        mode2 = 2 if v.eng else (0 if st2.kind == IDENTICAL else 1)
        cost2 = st2.cost
    else:
        mode2 = -1
        cost2 = 0
    w2 = P.future_stage_factor
    coin_pen = P.pay_coin_penalty
    econ_bonus = P.pay_econ_bonus
    free = s.rules.coins_free_choice
    best = [_NEG] * 7  # best completion score per first code

    def leaf(cs: int, econ_left: bool, acc: float) -> float:
        """Value of the holdings ``g`` (greys, mutated in place) / ``cs`` coins for the coming stages."""
        mv = _maxval(g, cs, econ_left, nmode)
        if mv > ncost:
            mv = ncost
        val = mv + acc
        if cost2:
            lg2, lc2 = _leftover(g, cs, ncost, nmode)
            mv2 = _maxval(lg2, lc2, econ_left, mode2)
            if mv2 > cost2:
                mv2 = cost2
            val += w2 * mv2
        return val

    def rec(first: int, glo: int, ghi: int, cs: int, need: int, econ_ok: bool, acc: float) -> bool:
        """Enumerate the completions from this node (greys ``glo..ghi-1`` allowed next, then coins);
        returns whether any completion exists (the engine only offers steps that keep one reachable)."""
        found = False
        for r in range(glo, ghi):
            n = g[r]
            if n <= 0:
                continue
            g[r] = n - 1
            acc2 = acc + 0.02 * n
            f = first if first >= 0 else r
            if need == 1:
                sc = leaf(cs, econ_ok, acc2)
                if sc > best[f]:
                    best[f] = sc
                found = True
            elif mode == 0:
                found = rec(f, r, r + 1, cs, need - 1, econ_ok, acc2) or found
            elif mode == 1:
                found = rec(f, r + 1, 5, cs, need - 1, econ_ok, acc2) or found
            else:
                found = rec(f, r, 5, cs, need - 1, econ_ok, acc2) or found
            g[r] = n
        if cs >= 1 and (free or not found):
            f = first if first >= 0 else 5
            acc2 = acc - coin_pen
            if need == 1:
                sc = leaf(cs - 1, econ_ok, acc2)
                if sc > best[f]:
                    best[f] = sc
                found = True
            else:
                found = rec(f, 5, 5, cs - 1, need - 1, econ_ok, acc2) or found  # only coins may follow a coin
            if econ_ok and need == 2:  # the doubled coin is the highest code: always the final card
                f = first if first >= 0 else 6
                sc = leaf(cs - 1, False, acc - coin_pen + econ_bonus)
                if sc > best[f]:
                    best[f] = sc
                found = True
        return found

    rec(-1, lo, hi, c, need, econ, 0.0)
    out = []
    for a in legal:
        if a < A.PAY_COIN:
            code = a - A.PAY_BASE
        else:
            code = 5 if a == A.PAY_COIN else 6
        sc = best[code]
        if sc <= _NEG:  # defensive: an offered step the enumeration did not reach -> single-card view
            if code < 5:
                g[code] -= 1
                sc = leaf(c, econ, 0.02 * (g[code] + 1))
                g[code] += 1
            else:
                sc = leaf(c - 1, econ and code != 6, -coin_pen + (econ_bonus if code == 6 else 0.0))
        out.append(sc)
    return out


def _score_science(ctx: _Ctx, legal: Sequence[int]) -> List[float]:
    s, P = ctx.s, ctx.P
    m = s.mover
    v = ctx.views[m]
    T = ctx.best_token(m)
    st = ctx.sci_time
    phi1 = P.sci_phi1 * T * st
    phi2 = P.sci_phi2 * T * st
    c0, c1, c2 = v.sci
    out = []
    for a in legal:
        if a == A.SCI_TRIPLE:
            out.append(_phi(c0 - 1, c1 - 1, c2 - 1, T, phi1, phi2))
        else:
            sym = a - A.SCI_PAIR_BASE
            out.append(_phi(c0 - (2 if sym == 0 else 0), c1 - (2 if sym == 1 else 0), c2 - (2 if sym == 2 else 0),
                            T, phi1, phi2))
    return out


def _score_token(ctx: _Ctx, legal: Sequence[int]) -> List[float]:
    s, P = ctx.s, ctx.P
    m = s.mover
    om = 1 - m
    mine = ctx.token_values(m)
    theirs = ctx.token_values(om)
    blind_me = ctx.blind_token(m)
    blind_opp = ctx.blind_token(om) if s.prog_stack > 0 else _NEG
    faceup = s.faceup
    deny = P.deny_token
    out = []
    for a in legal:
        if a == A.TOKEN_BLIND:
            my = blind_me
            reply = _NEG
            for u in faceup:
                if theirs[u] > reply:
                    reply = theirs[u]
        else:
            t = a - A.TOKEN_BASE
            my = mine[t]
            reply = blind_opp
            skipped = False
            for u in faceup:
                if u == t and not skipped:
                    skipped = True
                    continue
                if theirs[u] > reply:
                    reply = theirs[u]
        if reply <= _NEG:
            reply = 0.0
        out.append(my - deny * reply)
    return out


def _score_hali_deck(ctx: _Ctx, legal: Sequence[int]) -> List[float]:
    s = ctx.s
    m = s.mover
    vme = ctx.card_values(m)
    out = []
    for a in legal:
        if a == A.SKIP:
            out.append(0.0)
            continue
        d = m if a == A.PICK_LEFT else 1 - m
        n = s.deck_size[d] - 1
        if n > 4:
            n = 4
        out.append(_expected_max(vme, s.deck_top[d], s.unseen[d], n))
    return out


def _score_hali_choose(ctx: _Ctx, legal: Sequence[int]) -> List[float]:
    vme = ctx.card_values(ctx.s.mover)
    return [vme[a - A.HALI_BASE] for a in legal]


# --------------------------------------------------------------------------- public API
def score_actions(state: GameState, params: Optional[HeuristicParams] = None) -> Tuple[Sequence[int], List[float]]:
    """Return ``(legal_actions, scores)`` for the decision node ``state`` (VP-like units)."""
    P = params or _DEFAULT
    legal = state.legal_actions()
    if not legal:
        raise ValueError("score_actions called on a non-decision node")
    if len(legal) == 1:
        return legal, [0.0]
    ctx = _Ctx(state, P)
    dk = state.dkind
    if dk == D_PICK:
        scores = _score_pick(ctx, legal)
    elif dk == D_PAY:
        scores = _score_pay(ctx, legal)
    elif dk == D_SCIENCE:
        scores = _score_science(ctx, legal)
    elif dk == D_TOKEN:
        scores = _score_token(ctx, legal)
    elif dk == D_HALI_DECK:
        scores = _score_hali_deck(ctx, legal)
    elif dk == D_HALI_CHOOSE:
        scores = _score_hali_choose(ctx, legal)
    else:  # pragma: no cover - unknown decision kind: stay legal
        scores = [0.0] * len(legal)
    return legal, scores


def heuristic_action(state: GameState, rng: Optional[np.random.Generator] = None,
                     params: Optional[HeuristicParams] = None) -> int:
    """Pick a legal action for the player to move (greedy on the heuristic scores).

    ``rng`` is used only for tie-breaking noise and the optional epsilon-greedy exploration
    (``params.epsilon``); with ``rng=None`` the choice is deterministic.
    """
    P = params or _DEFAULT
    legal = state.legal_actions()
    n = len(legal)
    if n == 1:
        return int(legal[0])
    if rng is not None and P.epsilon > 0.0 and rng.random() < P.epsilon:
        return int(legal[int(rng.integers(n))])
    _, scores = score_actions(state, P)
    if rng is not None and P.tie_noise > 0.0:
        noise = rng.random(n)
        best = 0
        bs = scores[0] + P.tie_noise * noise[0]
        for i in range(1, n):
            x = scores[i] + P.tie_noise * noise[i]
            if x > bs:
                bs = x
                best = i
        return int(legal[best])
    best = 0
    bs = scores[0]
    for i in range(1, n):
        if scores[i] > bs:
            bs = scores[i]
            best = i
    return int(legal[best])


def heuristic_prior(state: GameState, params: Optional[HeuristicParams] = None,
                    temperature: Optional[float] = None) -> np.ndarray:
    """Softmax of the heuristic scores over the legal actions, shape ``(Actions.NUM,)``.

    Illegal actions get probability 0.  ``temperature`` (VP units) defaults to
    ``params.prior_temperature``; a smaller value makes the prior sharper.
    """
    P = params or _DEFAULT
    out = np.zeros(A.NUM, dtype=np.float32)
    legal = state.legal_actions()
    if not legal:
        return out
    if len(legal) == 1:
        out[int(legal[0])] = 1.0
        return out
    _, scores = score_actions(state, P)
    temp = P.prior_temperature if temperature is None else temperature
    if temp <= 0.0:
        temp = 1e-6
    sc = np.asarray(scores, dtype=np.float64) / temp
    sc -= sc.max()
    w = np.exp(sc)
    w /= w.sum()
    for a, p in zip(legal, w):
        out[int(a)] = p
    return out


class HeuristicAgent:
    """Agent wrapper around :func:`heuristic_action` with optional epsilon-greedy noise."""

    def __init__(self, params: Optional[HeuristicParams] = None, epsilon: float = 0.0,
                 rng: Optional[np.random.Generator] = None, seed: Optional[int] = None, name: Optional[str] = None):
        base = params or _DEFAULT
        if epsilon != base.epsilon:
            from dataclasses import replace
            base = replace(base, epsilon=epsilon)
        self.params = base
        self.rng = rng or np.random.default_rng(seed)
        self.name = name or ("heuristic" if base.epsilon <= 0 else f"heuristic(eps={base.epsilon:g})")

    def new_game(self) -> None:
        pass

    def select_action(self, state: GameState) -> int:
        return heuristic_action(state, self.rng, self.params)


__all__ = ["HeuristicParams", "HeuristicAgent", "heuristic_action", "heuristic_prior", "score_actions"]
