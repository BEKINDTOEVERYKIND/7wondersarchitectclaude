"""Belief-state engine for 2-player *7 Wonders: Architects*.

The state never stores deck orders.  Each deck is ``(visible top kind, multiset of unseen
kinds)``; revealing a card is a **chance node** whose distribution is derived from the unseen
multiset.  The real game (:mod:`sevenwa.engine.env`) resolves chance nodes with the true
shuffled order; the search samples them.

Turn processing is driven by a small work queue (``self.queue``).  ``_run`` pops items and
resolves them until it reaches a *decision* (player must choose), a *chance* event, or the end
of the game.  Decisions with a single legal action and chance events with a single outcome are
resolved automatically so the search tree only contains real choices.

Canonical order of resolution within a turn (the rulebook lets the player choose any order;
fixing an order keeps the tree small and, because every mandatory action is still taken, does
not change what a player can achieve):

1. (Cat holder) peek at the central deck.
2. Main pick: left (own) deck, right (opponent's) deck or central deck.
3. Card placed -> immediate effects (Cat pawn, horns) -> Progress-token extra picks that the
   card triggers -> mandatory science sets -> Progress token choice -> mandatory construction
   (with payment choices) -> stage effect -> Architecture extra pick -> repeat until nothing is
   pending.  Science is resolved before construction so that a token gained this turn
   (Engineering, Economy, Architecture...) already applies to this turn's build; doing science
   first never prevents a build (green cards are not building resources), so this order weakly
   dominates the alternative.  (A token gained mid-turn never fires for a card placed earlier in
   the same turn; this is the only way the fixed order differs from a fully free ordering.
   Olympia's two cards both land before the mandatory-construction check runs.)
4. End of turn: battle (if triggered), game end (if a Wonder is complete), next player.

Payments are sequences of "pay one card" decisions (grey resource, coin, or a coin worth 2 with
Economy).  Codes must be chosen in non-decreasing order (wood < stone < clay < papyrus < glass <
coin < coin×2), which removes permutations of the same multiset from the tree: a payment
decision only appears when genuinely different sets of cards could be spent.

Deck indices are absolute: 0 = player 0's Wonder deck, 1 = player 1's Wonder deck, 2 = central.
The deck to a player's *left* is their own deck (rulebook: "make a deck between the player to
your left and yourself"); the deck to their *right* is the opponent's.  Actions are expressed
relative to the mover (``PICK_LEFT`` = own deck).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..game import CHANCE
from .actions import Actions as A
from .cards import (BLUE, COIN_KIND, GREEN, GREY, KINDS, KIND_OF_RESOURCE, KIND_OF_SYMBOL, NUM_KINDS, RED,
                    RESOURCES, SYMBOLS, YELLOW, central_deck_counts, wonder_deck_counts)
from .rules import RulesConfig
from .tokens import (K_CULTURE, K_ECONOMY, K_ENGINEERING, K_EXTRA_ON_BUILD, K_EXTRA_ON_GREEN, K_EXTRA_ON_HORN,
                     K_EXTRA_ON_RESOURCE, K_SHIELDS, K_VP_CAT, K_VP_MIL, K_VP_PROG, K_VP_WONDER, NUM_TOKEN_TYPES,
                     TOKENS)
from .wonders import (ALL_BUILT, DIFFERENT, E_ANY_DECK, E_CENTRAL, E_LEFT_RIGHT, E_LOOK5, E_NONE, E_SHIELD, E_TOKEN,
                      IDENTICAL, NUM_STAGES, WONDERS, popcount)

CENTRAL = 2
NUM_DECKS = 3

# node types
N_DECISION, N_CHANCE, N_TERMINAL = 0, 1, 2

# decision kinds
D_PICK, D_PAY, D_SCIENCE, D_TOKEN, D_HALI_DECK, D_HALI_CHOOSE, D_STAGE = range(7)
DECISION_NAMES = ["pick", "pay", "science", "token", "hali_deck", "hali_choose", "stage"]
# chance kinds
C_REVEAL, C_DRAW_CENTRAL, C_PEEK, C_TOKEN_REVEAL, C_TOKEN_BLIND, C_HALI_REVEAL = range(6)
CHANCE_NAMES = ["reveal", "draw_central", "peek", "token_reveal", "token_blind", "hali_reveal"]

# payment codes: 0..4 grey resources, 5 coin, 6 coin x2
PAY_COIN_CODE, PAY_COIN2_CODE = 5, 6

# token type ids by kind (resolved once)
_TOK_BY_KIND: Dict[str, List[int]] = {}
for _t in TOKENS:
    _TOK_BY_KIND.setdefault(_t.kind, []).append(_t.id)
_ARCHITECTURE = _TOK_BY_KIND.get(K_EXTRA_ON_BUILD, [])
_PROPAGANDA = _TOK_BY_KIND.get(K_EXTRA_ON_HORN, [])
_SCIENCE_TOK = _TOK_BY_KIND.get(K_EXTRA_ON_GREEN, [])
_RESOURCE_TOKS = _TOK_BY_KIND.get(K_EXTRA_ON_RESOURCE, [])
_ENGINEERING = _TOK_BY_KIND.get(K_ENGINEERING, [])
_ECONOMY = _TOK_BY_KIND.get(K_ECONOMY, [])
_SHIELD_TOKS = _TOK_BY_KIND.get(K_SHIELDS, [])
_RED_HORN_KINDS = [k.id for k in KINDS if k.type == RED and k.horns > 0]
_CAT_KINDS = [k.id for k in KINDS if k.type == BLUE and k.cat]

_ALL_SOURCES = (0, 1, 2)


class GameState:
    """Immutable-by-convention belief state (``apply_*`` return modified copies)."""

    __slots__ = (
        "rules", "wonder", "built", "cards", "tokens", "wonder_shields", "mil_tokens",
        "deck_top", "unseen", "deck_size", "central_known_to", "discard",
        "faceup", "prog_unseen", "prog_stack", "conflict", "cat",
        "mover", "turn", "tokens_used", "econ_used", "battle_pending", "wonder_done", "game_over",
        "queue", "node_type", "dkind", "dctx", "ckind", "cctx", "_legal", "_outcomes",
        "history_len", "observer", "hali_event", "tokens_new",
    )

    # ------------------------------------------------------------------ construction
    def __init__(self):
        pass

    @staticmethod
    def initial(wonders: Tuple[int, int], rules: Optional[RulesConfig] = None) -> "GameState":
        s = GameState()
        s.rules = rules or RulesConfig()
        s.wonder = (int(wonders[0]), int(wonders[1]))
        s.built = [0, 0]  # per player: bitmask of constructed stages (bit i = stage i in cost order)
        s.cards = [[0] * NUM_KINDS, [0] * NUM_KINDS]
        s.tokens = [[0] * NUM_TOKEN_TYPES, [0] * NUM_TOKEN_TYPES]
        s.wonder_shields = [0, 0]
        s.mil_tokens = [0, 0]
        s.deck_top = [-1, -1, -1]
        s.unseen = [wonder_deck_counts(WONDERS[s.wonder[0]].name), wonder_deck_counts(WONDERS[s.wonder[1]].name),
                    central_deck_counts()]
        s.deck_size = [sum(s.unseen[0]), sum(s.unseen[1]), sum(s.unseen[2])]
        s.central_known_to = 0  # bitmask of players who know the central top card
        s.observer = -1  # player whose belief this state represents (-1: public/omniscient environment view)
        s.hali_event = None  # (deck, window size, kept kind) when the last transition resolved a Halicarnassus choice
        s.discard = [0] * NUM_KINDS
        s.faceup = []
        s.prog_unseen = [t.copies for t in TOKENS]
        s.prog_stack = sum(s.prog_unseen)
        s.conflict = 0
        s.cat = -1
        s.mover = 0
        s.turn = 0
        s.tokens_used = 0
        s.tokens_new = 0
        s.econ_used = False
        s.battle_pending = False
        s.wonder_done = False
        s.game_over = False
        s.queue = (("reveal", 0), ("reveal", 1)) + (("token_reveal",),) * s.rules.faceup_progress_tokens + (("turn_start",),)
        s.node_type = N_CHANCE
        s.dkind = -1
        s.dctx = None
        s.ckind = -1
        s.cctx = None
        s._legal = None
        s._outcomes = None
        s.history_len = 0
        s._run()
        return s

    def _copy(self) -> "GameState":
        s = GameState()
        s.rules = self.rules
        s.wonder = self.wonder
        s.built = list(self.built)
        s.cards = [list(self.cards[0]), list(self.cards[1])]
        s.tokens = [list(self.tokens[0]), list(self.tokens[1])]
        s.wonder_shields = list(self.wonder_shields)
        s.mil_tokens = list(self.mil_tokens)
        s.deck_top = list(self.deck_top)
        s.unseen = [list(self.unseen[0]), list(self.unseen[1]), list(self.unseen[2])]
        s.deck_size = list(self.deck_size)
        s.central_known_to = self.central_known_to
        s.discard = list(self.discard)
        s.faceup = list(self.faceup)
        s.prog_unseen = list(self.prog_unseen)
        s.prog_stack = self.prog_stack
        s.conflict = self.conflict
        s.cat = self.cat
        s.mover = self.mover
        s.turn = self.turn
        s.tokens_used = self.tokens_used
        s.tokens_new = self.tokens_new
        s.econ_used = self.econ_used
        s.battle_pending = self.battle_pending
        s.wonder_done = self.wonder_done
        s.game_over = self.game_over
        s.queue = self.queue
        s.node_type = self.node_type
        s.dkind = self.dkind
        s.dctx = self.dctx
        s.ckind = self.ckind
        s.cctx = self.cctx
        s._legal = None
        s._outcomes = None
        s.history_len = self.history_len
        s.observer = self.observer
        s.hali_event = None
        return s

    # ------------------------------------------------------------------ GameState protocol
    def to_move(self) -> int:
        if self.node_type == N_DECISION:
            return self.mover
        if self.node_type == N_CHANCE:
            return CHANCE
        return -2

    def is_terminal(self) -> bool:
        return self.node_type == N_TERMINAL

    def is_chance(self) -> bool:
        return self.node_type == N_CHANCE

    def legal_actions(self) -> Sequence[int]:
        if self.node_type != N_DECISION:
            return ()
        return self._legal

    def apply_action(self, action: int) -> "GameState":
        if self.node_type != N_DECISION:
            raise ValueError("not a decision node")
        if action not in self._legal:
            raise ValueError(f"illegal action {A.name(action)} at {self.describe_node()}")
        s = self._copy()
        s.history_len += 1
        s._apply_decision(action)
        s._run()
        return s

    def chance_outcomes(self) -> Sequence[Tuple[int, float]]:
        if self.node_type != N_CHANCE:
            return ()
        return self._outcomes

    def apply_chance(self, outcome: int) -> "GameState":
        if self.node_type != N_CHANCE:
            raise ValueError("not a chance node")
        s = self._copy()
        s.history_len += 1
        s._apply_chance(outcome)
        s._run()
        return s

    def returns(self) -> Tuple[float, float]:
        if not self.game_over:
            return (0.0, 0.0)
        s0, s1 = self.scores()
        if s0 != s1:
            return (1.0, -1.0) if s0 > s1 else (-1.0, 1.0)
        n0, n1 = popcount(self.built[0]), popcount(self.built[1])
        if self.rules.tiebreak_stages and n0 != n1:
            return (1.0, -1.0) if n0 > n1 else (-1.0, 1.0)
        return (0.0, 0.0)

    def score_diff(self) -> float:
        s0, s1 = self.scores()
        return float(s0 - s1)

    def key(self) -> bytes:
        return repr((
            self.wonder, self.built, self.cards, self.tokens, self.wonder_shields, self.mil_tokens,
            self.deck_top, self.unseen, self.deck_size, self.central_known_to, self.faceup, self.prog_unseen,
            self.conflict, self.cat, self.mover, self.tokens_used, self.econ_used, self.battle_pending,
            self.wonder_done, self.game_over, self.queue, self.node_type, self.dkind, self.dctx, self.ckind, self.cctx,
            self.observer, self.tokens_new,
        )).encode()

    @staticmethod
    def num_actions() -> int:
        return A.NUM

    # ------------------------------------------------------------------ scoring
    def scores(self) -> Tuple[int, int]:
        return (self.score_of(0), self.score_of(1))

    def score_of(self, p: int) -> int:
        total = WONDERS[self.wonder[p]].vp_of_built(self.built[p])
        cards = self.cards[p]
        for k in KINDS:
            if k.type == BLUE and cards[k.id]:
                total += cards[k.id] * k.vp
        total += self.mil_tokens[p] * self.rules.military_token_vp
        total += self.token_vp(p)
        if self.cat == p:
            total += self.rules.cat_vp
        return total

    def token_vp(self, p: int) -> int:
        toks = self.tokens[p]
        total = 0
        n_tokens = sum(toks)
        for t in TOKENS:
            c = toks[t.id]
            if not c:
                continue
            if t.kind == K_VP_MIL:
                total += t.vp * self.mil_tokens[p]
            elif t.kind == K_VP_PROG:
                total += t.vp * n_tokens
            elif t.kind == K_VP_WONDER:
                total += t.vp_complete if self.built[p] == ALL_BUILT else t.vp_incomplete
            elif t.kind == K_VP_CAT:
                total += t.vp * sum(self.cards[p][k] for k in _CAT_KINDS)
            elif t.kind == K_CULTURE:
                total += t.vp_both if c >= 2 else t.vp_one
        return total

    def shields_of(self, p: int) -> int:
        cards = self.cards[p]
        total = self.wonder_shields[p]
        for k in KINDS:
            if k.type == RED and cards[k.id]:
                total += cards[k.id] * k.shields
        for t in _SHIELD_TOKS:
            total += self.tokens[p][t] * TOKENS[t].shields
        return total

    def num_stages(self, p: int) -> int:
        """Number of constructed Wonder stages of player ``p``."""
        return popcount(self.built[p])

    def available_stages(self, p: int) -> Tuple[int, ...]:
        """Stage indexes player ``p`` may construct next (unbuilt, prerequisites built)."""
        return WONDERS[self.wonder[p]].available(self.built[p])

    def wonder_complete(self, p: int) -> bool:
        return self.built[p] == ALL_BUILT

    def science_counts(self, p: int) -> List[int]:
        return [self.cards[p][KIND_OF_SYMBOL[s]] for s in range(len(SYMBOLS))]

    def knows_central(self, p: int) -> bool:
        """Does player ``p`` know the identity of the central deck's top card (in this belief state)?"""
        return p >= 0 and self.deck_top[CENTRAL] >= 0 and bool((self.central_known_to >> p) & 1)

    def has_peeked(self, p: int) -> bool:
        """Public fact: player ``p`` has looked at the current central top card (even if this belief
        state does not know which card it is)."""
        return p >= 0 and bool((self.central_known_to >> p) & 1)

    def has_token(self, p: int, tid: int) -> bool:
        return self.tokens[p][tid] > 0

    def _token_active(self, p: int, t: int) -> bool:
        """Owned, not yet used this turn, and (unless allowed) not gained this very turn."""
        if not self.tokens[p][t] or (self.tokens_used >> t) & 1:
            return False
        if not self.rules.token_usable_same_turn and (self.tokens_new >> t) & 1:
            return False
        return True

    def tokens_remaining(self) -> int:
        return len(self.faceup) + self.prog_stack

    # ------------------------------------------------------------------ helpers
    def _deck_of(self, action: int) -> int:
        if action == A.PICK_LEFT:
            return self.mover
        if action == A.PICK_RIGHT:
            return 1 - self.mover
        return CENTRAL

    def _action_of_deck(self, d: int) -> int:
        if d == CENTRAL:
            return A.PICK_CENTER
        return A.PICK_LEFT if d == self.mover else A.PICK_RIGHT

    def _push(self, *items) -> None:
        self.queue = tuple(items) + self.queue

    def _set_decision(self, kind: int, ctx, legal: List[int]) -> bool:
        """Return True if the run loop must stop (a real choice); auto-resolves single options."""
        if len(legal) == 1:
            self.dkind, self.dctx = kind, ctx
            self._apply_decision(legal[0])
            return False
        self.node_type = N_DECISION
        self.dkind, self.dctx = kind, ctx
        self._legal = legal
        return True

    def _set_chance(self, kind: int, ctx, counts: Sequence[int]) -> bool:
        total = sum(counts)
        outs = [(i, c / total) for i, c in enumerate(counts) if c]
        if len(outs) == 1:
            self.ckind, self.cctx = kind, ctx
            self._apply_chance(outs[0][0])
            return False
        self.node_type = N_CHANCE
        self.ckind, self.cctx = kind, ctx
        self._outcomes = outs
        return True

    # ------------------------------------------------------------------ main loop
    def _run(self) -> None:
        while True:
            if self.game_over:
                self.node_type = N_TERMINAL
                self._legal = ()
                self._outcomes = ()
                return
            if not self.queue:
                raise RuntimeError("empty queue")  # pragma: no cover
            item = self.queue[0]
            self.queue = self.queue[1:]
            if self._handle(item):
                return

    def _handle(self, item) -> bool:
        op = item[0]
        m = self.mover
        if op == "turn_start":
            self.tokens_used = 0
            self.tokens_new = 0
            self.econ_used = False
            if all(n == 0 for n in self.deck_size):
                # No card can ever be drawn again: the game cannot progress, so it ends here
                # (``end_when_no_cards`` only controls whether the game *also* ends as soon as the
                # player to move has no card to draw while other decks are still empty—which is the
                # same condition in a 2-player game, so both settings coincide).
                self.game_over = True
                return False
            self.queue = (("pick", "main", False, _ALL_SOURCES), ("end_turn",)) + self.queue
            if self.cat == m and self.deck_size[CENTRAL] > 0 and not self.knows_central(m):
                if self.deck_top[CENTRAL] >= 0:
                    # already known to the other player (who must be the observer): the peek reveals that card
                    self.central_known_to |= 1 << m
                    return False
                self.central_known_to &= ~(1 << m)  # (re)peek: the sampled card becomes known to the mover
                return self._set_chance(C_PEEK, None, self.unseen[CENTRAL])
            return False
        if op == "pick":
            _, reason, optional, sources = item
            if reason.startswith("token:") and (self.tokens_used >> int(reason[6:])) & 1:
                return False  # once per turn: an earlier trigger of this token already took its card
            avail = [d for d in sources if self.deck_size[d] > 0]
            if not avail:
                return False
            if (not self.rules.cat_peek_main_draw_only and reason != "main" and self.cat == m and CENTRAL in avail
                    and not self.knows_central(m)):
                # house rule: the Cat holder may also peek before an extra draw (same logic as ``turn_start``)
                if self.deck_top[CENTRAL] >= 0:
                    # a card is already on top (peeked by the other player before the Cat changed hands): the
                    # peek reveals that very card -- no chance node, and no second card is sampled
                    self.central_known_to |= 1 << m
                else:
                    self.central_known_to &= ~(1 << m)
                    self._push(item)
                    return self._set_chance(C_PEEK, None, self.unseen[CENTRAL])
            legal = [self._action_of_deck(d) for d in avail]
            if optional:
                legal.append(A.SKIP)
            return self._set_decision(D_PICK, (reason, optional, tuple(avail)), legal)
        if op == "take":
            _, d, reason = item
            return self._take(d, reason)
        if op == "reveal":
            d = item[1]
            if self.deck_size[d] > 0 and self.deck_top[d] < 0:
                return self._set_chance(C_REVEAL, d, self.unseen[d])
            return False
        if op == "token_reveal":
            if self.prog_stack > 0 and len(self.faceup) < self.rules.faceup_progress_tokens:
                return self._set_chance(C_TOKEN_REVEAL, None, self.prog_unseen)
            return False
        if op == "token_blind":
            if self.prog_stack > 0:
                return self._set_chance(C_TOKEN_BLIND, None, self.prog_unseen)
            return False
        if op == "placed":
            _, k, d, reason = item
            self._placed(k, d, reason)
            return False
        if op == "check":
            return self._check()
        if op == "pay":
            return self._pay_decision(item[1], item[2])
        if op == "token_choice":
            return self._token_choice(item[1])
        if op == "hali_deck":
            avail = [d for d in (m, 1 - m) if self.deck_size[d] > 0]
            if not avail:
                return False
            legal = [self._action_of_deck(d) for d in avail]
            if self.rules.wonder_effect_optional:
                legal.append(A.SKIP)
            return self._set_decision(D_HALI_DECK, tuple(avail), legal)
        if op == "hali_reveal":
            _, d, remaining, revealed = item
            if remaining > 0 and sum(self.unseen[d]) > 0:
                return self._set_chance(C_HALI_REVEAL, (d, remaining, revealed), self.unseen[d])
            return self._hali_choose(d, revealed)
        if op == "end_turn":
            self._end_turn()
            return False
        raise RuntimeError(f"unknown queue item {item}")  # pragma: no cover

    # ------------------------------------------------------------------ handlers
    def _take(self, d: int, reason: str) -> bool:
        """Take the top card of deck ``d`` for the mover."""
        if self.deck_size[d] == 0:
            return False
        if d != CENTRAL:
            k = self.deck_top[d]
            assert k >= 0, "wonder deck top must be visible"
            self.deck_size[d] -= 1
            self.deck_top[d] = -1
            self._push(("reveal", d), ("placed", k, d, reason))
            return False
        # central deck: deterministic if the mover or the observer knows the top card
        if self.deck_top[CENTRAL] >= 0 and (self.knows_central(self.mover) or self.knows_central(self.observer)):
            k = self.deck_top[CENTRAL]
            self.deck_top[CENTRAL] = -1
            self.central_known_to = 0
            self.deck_size[CENTRAL] -= 1
            self._push(("placed", k, d, reason))
            return False
        if self.deck_top[CENTRAL] >= 0:  # an in-tree sample known only to the other player: forget it
            self.unseen[CENTRAL][self.deck_top[CENTRAL]] += 1
            self.deck_top[CENTRAL] = -1
        # Blind draw: a new, unseen card comes on top, so nobody knows it.  This also drops the public
        # "the other player has peeked" bit that an observation keeps while hiding the card (env.observe).
        self.central_known_to = 0
        return self._set_chance(C_DRAW_CENTRAL, reason, self.unseen[CENTRAL])

    def _placed(self, k: int, d: int, reason: str) -> None:
        m = self.mover
        kind = KINDS[k]
        self.cards[m][k] += 1
        if reason.startswith("token:"):
            self.tokens_used |= 1 << int(reason[6:])
        if kind.type == BLUE and kind.cat:
            self.cat = m
        if kind.type == RED and kind.horns > 0:
            room = self.rules.conflict_tokens - self.conflict
            if room > 0:
                self.conflict += min(kind.horns, room)
                if self.conflict >= self.rules.conflict_tokens:
                    self.battle_pending = True
        # everything below is pushed to the FRONT, so push in reverse order of execution
        if reason != "olympia_nocheck":  # Olympia: the check runs once both cards have landed
            self._push(("check",))
        trig: List[int] = []
        if kind.type == GREY:
            for t in _RESOURCE_TOKS:
                if kind.resource in TOKENS[t].resources:
                    trig.append(t)
        elif kind.type == YELLOW:
            for t in _RESOURCE_TOKS:
                if TOKENS[t].triggers_on_coin:
                    trig.append(t)
        elif kind.type == GREEN:
            trig.extend(_SCIENCE_TOK)
        elif kind.type == RED and kind.horns > 0:
            trig.extend(_PROPAGANDA)
        for t in trig:
            if self._token_active(m, t):
                self._push(("pick", f"token:{t}", self.rules.extra_card_optional, _ALL_SOURCES))

    # ---- mandatory construction & science
    def _check(self) -> bool:
        """Mandatory actions: science sets first (a new token may apply to this turn's build), then construction."""
        m = self.mover
        sci = self._science_options()
        if sci:
            return self._set_decision(D_SCIENCE, None, sci)
        opts = [i for i in self.available_stages(m) if self._affordable(i)]
        if not opts:
            return False
        # several affordable stages (Rhodes' two foundations, Ephesus' three middle stages...): the player
        # chooses which one to construct; a single option is auto-resolved by ``_set_decision``
        return self._set_decision(D_STAGE, tuple(opts), [A.STAGE_BASE + i for i in opts])

    def _science_options(self) -> List[int]:
        if self.tokens_remaining() == 0:
            return []
        counts = self.science_counts(self.mover)
        opts = [A.SCI_PAIR_BASE + s for s in range(len(SYMBOLS)) if counts[s] >= 2]
        if all(c >= 1 for c in counts):
            opts.append(A.SCI_TRIPLE)
        return opts

    # ---- payment
    def _stage_at(self, i: int):
        return WONDERS[self.wonder[self.mover]].stages[i]

    def _pay_state(self, chosen: Tuple[int, ...]):
        m = self.mover
        cards = self.cards[m]
        grey_avail = [cards[KIND_OF_RESOURCE[r]] for r in range(5)]
        coins_avail = cards[COIN_KIND]
        value = 0
        econ_used = self.econ_used
        used_res = []
        for c in chosen:
            if c < 5:
                grey_avail[c] -= 1
                value += 1
                used_res.append(c)
            elif c == PAY_COIN_CODE:
                coins_avail -= 1
                value += 1
            else:
                coins_avail -= 1
                value += 2
                econ_used = True
        return grey_avail, coins_avail, value, econ_used, used_res

    def _mode(self, i: int) -> int:
        """0 identical, 1 different, 2 any (Engineering) for stage ``i`` of the mover's Wonder."""
        if any(self.tokens[self.mover][t] for t in _ENGINEERING):
            return 2
        return 0 if self._stage_at(i).kind == IDENTICAL else 1

    def _max_value(self, grey_avail, coins_avail, econ_used, used_res, mode: int, min_code: int = 0) -> int:
        """Largest resource value still reachable, paying in canonical order (codes >= ``min_code``).

        Payments are sequences of codes; requiring non-decreasing codes removes permutations of the
        same multiset (which would otherwise create redundant decision branches) while keeping every
        distinct multiset reachable.
        """
        econ = (not econ_used) and coins_avail >= 1 and any(self.tokens[self.mover][t] for t in _ECONOMY)
        coin_cap = (coins_avail if min_code <= PAY_COIN_CODE else 0) + (1 if (econ and min_code <= PAY_COIN2_CODE) else 0)
        if mode == 2:
            return sum(grey_avail[r] for r in range(min_code, 5)) + coin_cap
        if mode == 0:
            if used_res:
                r = used_res[0]
                return (grey_avail[r] if r >= min_code else 0) + coin_cap
            return max([grey_avail[r] for r in range(min_code, 5)] + [0]) + coin_cap
        return sum(1 for r in range(min_code, 5) if grey_avail[r] > 0 and r not in used_res) + coin_cap

    def _affordable(self, i: int) -> bool:
        grey_avail, coins_avail, value, econ_used, used_res = self._pay_state(())
        # With ``economy_forces_build`` False the doubled coin does not count towards the *mandatory*
        # construction check (BGG reading of "you can use"); it may still be spent once building.
        econ_used = econ_used or not self.rules.economy_forces_build
        return self._max_value(grey_avail, coins_avail, econ_used, used_res, self._mode(i)) >= self._stage_at(i).cost

    def _pay_options(self, i: int, chosen: Tuple[int, ...]) -> List[int]:
        cost = self._stage_at(i).cost
        mode = self._mode(i)
        grey_avail, coins_avail, value, econ_used, used_res = self._pay_state(chosen)
        need = cost - value
        min_code = chosen[-1] if chosen else 0
        opts: List[int] = []
        grey_possible = False
        for r in range(min_code, 5):
            if grey_avail[r] <= 0:
                continue
            if mode == 0 and used_res and used_res[0] != r:
                continue
            if mode == 1 and r in used_res:
                continue
            ga = list(grey_avail)
            ga[r] -= 1
            if 1 + self._max_value(ga, coins_avail, econ_used, used_res + [r], mode, r) >= need:
                opts.append(A.PAY_BASE + r)
                grey_possible = True
        if coins_avail >= 1 and (self.rules.coins_free_choice or not grey_possible):
            if min_code <= PAY_COIN_CODE and 1 + self._max_value(grey_avail, coins_avail - 1, econ_used, used_res, mode, PAY_COIN_CODE) >= need:
                opts.append(A.PAY_COIN)
            econ_ok = (not econ_used) and any(self.tokens[self.mover][t] for t in _ECONOMY)
            if econ_ok and need >= 2 and 2 + self._max_value(grey_avail, coins_avail - 1, True, used_res, mode, PAY_COIN2_CODE) >= need:
                opts.append(A.PAY_COIN2)
        return opts

    def _pay_decision(self, i: int, chosen: Tuple[int, ...]) -> bool:
        opts = self._pay_options(i, chosen)
        if not opts:  # every offered step keeps a completion reachable, so this is an engine bug
            raise RuntimeError(f"payment dead end: chosen={chosen} stage={self._stage_at(i)} cards={self.cards[self.mover]}")
        return self._set_decision(D_PAY, (i, chosen), opts)

    def _build(self, i: int, chosen: Tuple[int, ...]) -> None:
        m = self.mover
        for c in chosen:
            k = KIND_OF_RESOURCE[c] if c < 5 else COIN_KIND
            self.cards[m][k] -= 1
            self.discard[k] += 1
            if c == PAY_COIN2_CODE:
                self.econ_used = True
        stage = self._stage_at(i)
        assert i in self.available_stages(m), (i, self.built[m])
        self.built[m] |= 1 << i
        if self.rules.economy_once_per_build:
            self.econ_used = False  # the doubled coin may be used again for another stage this turn
        if self.built[m] == ALL_BUILT:
            self.wonder_done = True
        # push in reverse execution order: check <- architecture <- effect
        self._push(("check",))
        for t in _ARCHITECTURE:
            if self._token_active(m, t):
                self._push(("pick", f"token:{t}", self.rules.extra_card_optional, _ALL_SOURCES))
        eff = stage.effect
        opt = self.rules.wonder_effect_optional
        if eff == E_SHIELD:
            self.wonder_shields[m] += 1
        elif eff == E_TOKEN:
            self._push(("token_choice", "babylon"))
        elif eff == E_ANY_DECK:
            self._push(("pick", "alexandria", opt, _ALL_SOURCES))
        elif eff == E_CENTRAL:
            self._push(("pick", "ephesus", opt, (CENTRAL,)))
        elif eff == E_LEFT_RIGHT:
            # both cards land before the mandatory-construction check ("take the top card of the decks
            # to your left AND right"), so the player may pay with either card
            decks = [d for d in (m, 1 - m) if self.deck_size[d] > 0]
            items = [("take", d, "olympia_nocheck" if i < len(decks) - 1 else "olympia") for i, d in enumerate(decks)]
            if items:
                self._push(*items)
        elif eff == E_LOOK5:
            self._push(("hali_deck",))

    # ---- progress tokens
    def _token_choice(self, reason: str) -> bool:
        legal = sorted({A.TOKEN_BASE + t for t in self.faceup})
        if self.prog_stack > 0:
            legal.append(A.TOKEN_BLIND)
        if not legal:
            return False
        return self._set_decision(D_TOKEN, reason, legal)

    # ---- Halicarnassus
    def _hali_choose(self, d: int, revealed: Tuple[int, ...]) -> bool:
        legal = sorted({A.HALI_BASE + k for k in revealed})
        return self._set_decision(D_HALI_CHOOSE, (d, revealed), legal)

    # ---- end of turn
    def _end_turn(self) -> None:
        if self.battle_pending:
            self._resolve_battle()
        if self.wonder_done:
            self.game_over = True
            return
        self.mover = 1 - self.mover
        self.turn += 1
        self.queue = (("turn_start",),) + self.queue

    def _resolve_battle(self) -> None:
        s = [self.shields_of(0), self.shields_of(1)]
        for p in (0, 1):
            o = 1 - p
            if s[p] > s[o]:
                two = s[p] >= 2 * s[o]
                if self.rules.double_vs_zero_requires_two and s[o] == 0 and s[p] < 2:
                    two = False
                self.mil_tokens[p] += 2 if two else 1
        for p in (0, 1):
            for k in _RED_HORN_KINDS:
                c = self.cards[p][k]
                if c:
                    self.cards[p][k] = 0
                    self.discard[k] += c
        self.conflict = 0
        self.battle_pending = False

    # ------------------------------------------------------------------ applying choices
    def _apply_decision(self, action: int) -> None:
        kind = self.dkind
        ctx = self.dctx
        self.node_type = -1
        self._legal = None
        if kind == D_PICK:
            reason, optional, avail = ctx
            if action == A.SKIP:
                return
            d = self._deck_of(action)
            self._push(("take", d, reason))
        elif kind == D_STAGE:
            self._push(("pay", action - A.STAGE_BASE, ()))
        elif kind == D_PAY:
            i, chosen = ctx
            chosen = tuple(chosen)
            if A.PAY_BASE <= action < A.PAY_BASE + 5:
                code = action - A.PAY_BASE
            elif action == A.PAY_COIN:
                code = PAY_COIN_CODE
            else:
                code = PAY_COIN2_CODE
            chosen = chosen + (code,)
            _, _, value, _, _ = self._pay_state(chosen)
            if value >= self._stage_at(i).cost:
                self._build(i, chosen)
            else:
                self._push(("pay", i, chosen))
        elif kind == D_SCIENCE:
            m = self.mover
            if action == A.SCI_TRIPLE:
                for s in range(len(SYMBOLS)):
                    k = KIND_OF_SYMBOL[s]
                    self.cards[m][k] -= 1
                    self.discard[k] += 1
            else:
                k = KIND_OF_SYMBOL[action - A.SCI_PAIR_BASE]
                self.cards[m][k] -= 2
                self.discard[k] += 2
            self._push(("token_choice", "science"), ("check",))
        elif kind == D_TOKEN:
            m = self.mover
            if action == A.TOKEN_BLIND:
                self._push(("token_blind",))
            else:
                t = action - A.TOKEN_BASE
                self.faceup.remove(t)
                self.tokens[m][t] += 1
                self.tokens_new |= 1 << t
                self._push(("token_reveal",))
        elif kind == D_HALI_DECK:
            if action == A.SKIP:
                return
            d = self._deck_of(action)
            top = self.deck_top[d]
            self.deck_top[d] = -1
            n = min(5, self.deck_size[d])
            self._push(("hali_reveal", d, n - 1, (top,)))
        elif kind == D_HALI_CHOOSE:
            d, revealed = ctx
            k = action - A.HALI_BASE
            rest = list(revealed)
            rest.remove(k)
            for x in rest:
                self.unseen[d][x] += 1
            self.deck_size[d] -= 1
            self.hali_event = (d, len(revealed), k)
            self._push(("reveal", d), ("placed", k, d, "hali"))
        else:  # pragma: no cover
            raise RuntimeError("bad decision kind")

    def _apply_chance(self, outcome: int) -> None:
        kind = self.ckind
        ctx = self.cctx
        self.node_type = -1
        self._outcomes = None
        if kind == C_REVEAL:
            d = ctx
            self.unseen[d][outcome] -= 1
            self.deck_top[d] = outcome
        elif kind == C_DRAW_CENTRAL:
            self.unseen[CENTRAL][outcome] -= 1
            self.deck_size[CENTRAL] -= 1
            self._push(("placed", outcome, CENTRAL, ctx))
        elif kind == C_PEEK:
            self.unseen[CENTRAL][outcome] -= 1
            self.deck_top[CENTRAL] = outcome
            self.central_known_to |= 1 << self.mover
        elif kind == C_TOKEN_REVEAL:
            self.prog_unseen[outcome] -= 1
            self.prog_stack -= 1
            self.faceup.append(outcome)
        elif kind == C_TOKEN_BLIND:
            self.prog_unseen[outcome] -= 1
            self.prog_stack -= 1
            self.tokens[self.mover][outcome] += 1
            self.tokens_new |= 1 << outcome
        elif kind == C_HALI_REVEAL:
            d, remaining, revealed = ctx
            self.unseen[d][outcome] -= 1
            self._push(("hali_reveal", d, remaining - 1, revealed + (outcome,)))
        else:  # pragma: no cover
            raise RuntimeError("bad chance kind")

    # ------------------------------------------------------------------ debugging / display
    def describe_node(self) -> str:
        if self.node_type == N_DECISION:
            return f"decision[{DECISION_NAMES[self.dkind]}] ctx={self.dctx} legal={[A.name(a) for a in self._legal]}"
        if self.node_type == N_CHANCE:
            return f"chance[{CHANCE_NAMES[self.ckind]}] ctx={self.cctx} outcomes={len(self._outcomes)}"
        return "terminal"

    def describe(self) -> str:
        from .cards import describe_counts
        lines = [f"turn {self.turn} mover P{self.mover} conflict {self.conflict}/{self.rules.conflict_tokens} cat {self.cat} "
                 f"faceup tokens {[TOKENS[t].name for t in self.faceup]} stack {self.prog_stack}"]
        for p in (0, 1):
            w = WONDERS[self.wonder[p]]
            built = [w.stages[i].label() for i in range(NUM_STAGES) if (self.built[p] >> i) & 1]
            lines.append(f"  P{p} {w.name} stages {self.num_stages(p)}/5 {built} score {self.score_of(p)} shields {self.shields_of(p)} "
                         f"mil {self.mil_tokens[p]} tokens {[TOKENS[i].name for i, c in enumerate(self.tokens[p]) for _ in range(c)]}")
            lines.append(f"     cards: {describe_counts(self.cards[p])}")
        for d in range(3):
            top = KINDS[self.deck_top[d]].name if self.deck_top[d] >= 0 else "?"
            lines.append(f"  deck{d} size {self.deck_size[d]} top {top}" + (f" (known to {[p for p in (0, 1) if self.knows_central(p)]})" if d == CENTRAL and self.central_known_to else ""))
        lines.append("  node: " + self.describe_node())
        return "\n".join(lines)


def new_game(wonders: Tuple[int, int] = (0, 1), rules: Optional[RulesConfig] = None) -> GameState:
    return GameState.initial(wonders, rules)
