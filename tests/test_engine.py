"""Rules-fidelity tests for the belief-state engine (``sevenwa.engine.state``).

Organisation (one class per rule area):

* data tables, initial state, action space;
* card / token conservation and node sanity over random games;
* picks and chance draws, the Cat pawn;
* mandatory construction and payment (coins, Engineering, Economy);
* Wonder stage effects; science sets and Progress tokens (acquisition and use);
* military (horns, battles, 2-player token rule); end of game and scoring.

See ``conftest.py`` for the state-building helpers (``first_pick``, ``edit``/``run``,
``take_card``, ``check`` ...).  Tests marked ``xfail(strict=True)`` document engine behaviour
that deviates from the rules / design contract; they must start passing (and be un-marked) once
the engine is fixed.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Dict, List

import pytest

from sevenwa.engine import Actions as A
from sevenwa.engine import GameState, RulesConfig
from sevenwa.engine.cards import NUM_KINDS, central_deck_counts, wonder_deck_counts
from sevenwa.engine.state import (C_DRAW_CENTRAL, C_HALI_REVEAL, C_PEEK, C_REVEAL, C_TOKEN_BLIND, C_TOKEN_REVEAL, CENTRAL,
                                  D_HALI_CHOOSE, D_HALI_DECK, D_PAY, D_PICK, D_SCIENCE, D_STAGE, D_TOKEN)
from sevenwa.engine.tokens import TOKENS, TOTAL_TOKEN_COPIES
from sevenwa.engine.wonders import (E_ANY_DECK, E_CENTRAL, E_LEFT_RIGHT, E_LOOK5, E_NONE, E_SHIELD, E_TOKEN, WONDERS,
                                    WONDER_BY_NAME)
from sevenwa.game import CHANCE

from conftest import (DEFAULT_FACEUP, GREEN_KINDS, K, T, W, assert_invariants, assert_node_sane, check, drain_deck, edit, first_n,
                      exhaust_tokens, first_pick, give, give_many, give_token, in_transit, initial_composition,
                      is_main_pick_of, is_pick, most_likely, pay_through, payment_leaves, random_game, run, set_deck,
                      set_faceup, set_top, settle, take_card)


# =====================================================================================
# Data tables
# =====================================================================================
class TestData:
    def test_deck_compositions(self):
        assert NUM_KINDS == 14
        for w in WONDERS:
            assert sum(wonder_deck_counts(w.name)) == 25, w.name
        assert sum(central_deck_counts()) == 60
        comp = initial_composition(GameState.initial((W.Giza, W.Rhodes)))
        assert sum(comp) == 110

    def test_wonder_stage_tables(self):
        totals = {"Giza": 30, "Rhodes": 26, "Alexandria": 25, "Halicarnassus": 24, "Olympia": 22, "Ephesus": 22,
                  "Babylon": 20}
        effects = {"Giza": (E_NONE, 0), "Rhodes": (E_SHIELD, 2), "Alexandria": (E_ANY_DECK, 2),
                   "Halicarnassus": (E_LOOK5, 2), "Olympia": (E_LEFT_RIGHT, 2), "Ephesus": (E_CENTRAL, 3),
                   "Babylon": (E_TOKEN, 2)}
        assert len(WONDERS) == 7
        for w in WONDERS:
            assert len(w.stages) == 5
            assert w.total_vp == totals[w.name]
            assert all(2 <= st.cost <= 4 for st in w.stages)
            eff, n = effects[w.name]
            assert sum(1 for st in w.stages if st.effect == eff and eff != E_NONE) == n
            assert all(st.effect in (E_NONE, eff) for st in w.stages)
        # exact boards (user-supplied component photographs): VP per stage in cost order, effect stages, prerequisites
        boards = {
            "Alexandria": ([4, 3, 6, 5, 7], [1, 3], [(), (0,), (1,), (2,), (3,)]),
            "Babylon": ([3, 0, 5, 5, 7], [1, 3], [(), (0,), (1,), (2,), (2,)]),
            "Ephesus": ([3, 3, 4, 5, 7], [1, 2, 3], [(), (0,), (0,), (0,), (1, 2, 3)]),
            "Giza": ([4, 5, 6, 7, 8], [], [(), (0,), (1,), (2,), (3,)]),
            "Halicarnassus": ([3, 3, 6, 5, 7], [1, 3], [(), (0,), (1,), (1,), (2, 3)]),
            "Olympia": ([3, 2, 5, 5, 7], [1, 3], [(), (0,), (0,), (1, 2), (3,)]),
            "Rhodes": ([4, 4, 5, 6, 7], [0, 3], [(), (), (0, 1), (2,), (3,)]),
        }
        for name, (vps, eff_at, reqs) in boards.items():
            w = WONDER_BY_NAME[name]
            assert [st.vp for st in w.stages] == vps, name
            assert [i for i, st in enumerate(w.stages) if st.effect != E_NONE] == eff_at, name
            assert [st.requires for st in w.stages] == reqs, name
            assert [(st.cost, st.kind) for st in w.stages] == [(2, 1), (2, 0), (3, 1), (3, 0), (4, 1)], name

    def test_wonder_stage_graphs(self):
        """``available`` follows the printed tray diagrams; every Wonder is completable; VP/cost tables agree."""
        rhodes, babylon, ephesus = WONDER_BY_NAME["Rhodes"], WONDER_BY_NAME["Babylon"], WONDER_BY_NAME["Ephesus"]
        assert rhodes.available(0) == (0, 1) and rhodes.available(0b00001) == (1,) and rhodes.available(0b00010) == (0,)
        assert rhodes.available(0b00011) == (2,) and rhodes.available(0b00111) == (3,)
        assert babylon.available(0b00111) == (3, 4) and babylon.available(0b10111) == (3,)
        assert ephesus.available(0b00001) == (1, 2, 3) and ephesus.available(0b01011) == (2,)
        assert ephesus.available(0b01111) == (4,)
        for w in WONDERS:
            assert w.available(0) == ((0, 1) if w.name == "Rhodes" else (0,))
            assert w.available(31) == () and w.vp_of_built(31) == w.total_vp and w.cost_remaining(31) == 0
            assert w.vp_of_built(0) == 0 and w.cost_remaining(0) == 14
            assert sorted(w.plan(0)) == [0, 1, 2, 3, 4]
            for mask in range(32):
                for i in w.available(mask):
                    assert not (mask >> i) & 1 and (mask & w.stages[i].prereq_mask) == w.stages[i].prereq_mask

    def test_progress_tokens(self):
        assert len(TOKENS) == 14
        assert TOTAL_TOKEN_COPIES == 15
        assert TOKENS[T.Culture].copies == 2
        assert all(t.copies == 1 for t in TOKENS if t.id != T.Culture)
        names = {t.name for t in TOKENS}
        assert names == {"Architecture", "Propaganda", "Urbanism", "Crafts", "Jewellery", "Science", "Engineering",
                         "Economy", "Tactics", "Strategy", "Education", "Decor", "Politics", "Culture"}


# =====================================================================================
# Initial state and action space
# =====================================================================================
class TestInitialState:
    def test_initial_is_a_reveal_of_deck0_with_deck_proportions(self):
        s = GameState.initial((W.Giza, W.Rhodes))
        assert s.is_chance() and s.to_move() == CHANCE
        assert s.ckind == C_REVEAL and s.cctx == 0
        comp = wonder_deck_counts("Giza")
        outs = dict(s.chance_outcomes())
        assert set(outs) == {k for k in range(NUM_KINDS) if comp[k]}
        for k, p in outs.items():
            assert abs(p - comp[k] / 25) < 1e-12
        assert abs(sum(outs.values()) - 1) < 1e-12
        assert_invariants(s)

    def test_first_decision_after_setup(self, base):
        assert base.deck_size == [25, 25, 60]
        assert len(base.faceup) == 3 and base.prog_stack == 12
        assert sorted(base.faceup) == sorted(DEFAULT_FACEUP)
        assert base.to_move() == 0 and base.turn == 0
        assert base.dkind == D_PICK and base.dctx == ("main", False, (0, 1, 2))
        assert list(base.legal_actions()) == [A.PICK_LEFT, A.PICK_RIGHT, A.PICK_CENTER]
        assert base.deck_top[0] == K.wood and base.deck_top[1] == K.stone
        assert base.deck_top[CENTRAL] == -1 and base.central_known_to == 0 and base.observer == -1
        assert not base.knows_central(0) and not base.knows_central(1)
        assert base.unseen[0][K.wood] == wonder_deck_counts("Giza")[K.wood] - 1
        assert base.cat == -1 and base.conflict == 0 and not base.battle_pending
        assert base.scores() == (0, 0) and base.returns() == (0.0, 0.0) and base.score_diff() == 0.0
        assert not base.is_terminal()
        assert_invariants(base)

    def test_face_up_token_reveal_probabilities(self):
        s = GameState.initial((W.Giza, W.Rhodes)).apply_chance(K.wood).apply_chance(K.stone)
        assert s.is_chance() and s.ckind == C_TOKEN_REVEAL
        outs = dict(s.chance_outcomes())
        assert len(outs) == 14
        assert abs(outs[T.Culture] - 2 / 15) < 1e-12
        assert all(abs(outs[t] - 1 / 15) < 1e-12 for t in outs if t != T.Culture)
        s = s.apply_chance(T.Strategy)
        outs = dict(s.chance_outcomes())
        assert T.Strategy not in outs and abs(outs[T.Culture] - 2 / 14) < 1e-12
        s = s.apply_chance(T.Culture)
        assert abs(dict(s.chance_outcomes())[T.Culture] - 1 / 13) < 1e-12

    def test_history_len_counts_transitions(self, base):
        assert base.history_len == 5  # 2 deck reveals + 3 token reveals
        assert base.apply_action(A.PICK_LEFT).history_len == 6

    def test_apply_never_mutates_the_source(self, base):
        before = base.key()
        nxt = base.apply_action(A.PICK_CENTER)
        assert base.key() == before and nxt.key() != before
        before2 = nxt.key()
        nxt.apply_chance(K.civ3)
        assert nxt.key() == before2

    def test_wrong_node_type_raises(self, base):
        with pytest.raises(ValueError):
            base.apply_chance(K.wood)
        with pytest.raises(ValueError):
            base.apply_action(A.SKIP)  # not legal at a main pick
        ch = base.apply_action(A.PICK_CENTER)
        with pytest.raises(ValueError):
            ch.apply_action(A.PICK_LEFT)


class TestActions:
    def test_name_covers_every_action_id(self):
        names = [A.name(a) for a in range(A.NUM)]
        assert not any(n.startswith("?") for n in names)
        assert len(set(names)) == A.NUM
        assert A.name(A.NUM).startswith("?") and A.name(-1).startswith("?")

    def test_num_actions_matches(self):
        assert GameState.num_actions() == A.NUM
        assert A.STAGE_BASE == A.HALI_BASE + NUM_KINDS and A.NUM == A.STAGE_BASE + 5
        assert A.TOKEN_BLIND == A.TOKEN_BASE + len(TOKENS)
        assert A.SCI_TRIPLE == A.SCI_PAIR_BASE + 3
        assert A.PAY_COIN == A.PAY_BASE + 5 and A.PAY_COIN2 == A.PAY_COIN + 1


# =====================================================================================
# Invariants over random games
# =====================================================================================
class TestInvariantsOverRandomGames:
    N_GAMES = 200

    def test_conservation_and_node_sanity(self):
        stats: Dict[str, int] = {"battles": 0, "hali": 0, "peeks": 0, "tokens": 0, "builds": 0, "blind": 0,
                                 "pay_decisions": 0, "extra_picks": 0, "no_cards_end": 0, "wonder_end": 0, "olympia": 0}
        comp_cache: Dict = {}

        def on_step(before: GameState, after: GameState) -> None:
            comp = comp_cache.setdefault(before.wonder, initial_composition(before))
            assert_invariants(after, comp)
            assert after.history_len == before.history_len + 1
            if before.battle_pending and not after.battle_pending:
                stats["battles"] += 1
                assert after.conflict == 0
            if after.is_chance():
                if after.ckind == C_HALI_REVEAL:
                    stats["hali"] += 1
                elif after.ckind == C_PEEK:
                    stats["peeks"] += 1
                elif after.ckind == C_TOKEN_BLIND:
                    stats["blind"] += 1
            elif not after.is_terminal():
                if after.dkind == D_PAY:
                    stats["pay_decisions"] += 1
                elif after.dkind == D_PICK and after.dctx[0].startswith("token:"):
                    stats["extra_picks"] += 1
            if after.num_stages(0) + after.num_stages(1) > before.num_stages(0) + before.num_stages(1):
                stats["builds"] += 1
            if sum(map(sum, after.tokens)) > sum(map(sum, before.tokens)):
                stats["tokens"] += 1
            if after.is_terminal():
                if all(n == 0 for n in after.deck_size) and not after.wonder_done:
                    stats["no_cards_end"] += 1
                if after.wonder_done:
                    stats["wonder_end"] += 1

        for seed in range(self.N_GAMES):
            final = random_game(seed, on_step=on_step)
            assert final.is_terminal() and final.game_over
            s0, s1 = final.scores()
            r = final.returns()
            if s0 != s1:
                assert r == ((1.0, -1.0) if s0 > s1 else (-1.0, 1.0))
            else:
                assert r in ((1.0, -1.0), (-1.0, 1.0), (0.0, 0.0))
            assert final.score_diff() == s0 - s1
            if final.wonder_done:
                assert max(final.num_stages(0), final.num_stages(1)) == 5
        # the property tests above are not vacuous: every rule area was exercised.  (A science *choice*
        # -- pair vs triple -- needs the Science token's extra pick to bring a 4th green card in one
        # turn and shows up only once in a few hundred random games; it is covered deterministically
        # by ``TestScience.test_science_token_extra_pick_creates_a_pair_vs_triple_choice``.)
        assert stats["battles"] > 0 and stats["builds"] > 0 and stats["tokens"] > 0
        assert stats["pay_decisions"] > 0 and stats["extra_picks"] > 0
        assert stats["peeks"] > 0 and stats["blind"] > 0 and stats["wonder_end"] > 0

    def test_score_never_decreases_except_for_cat_and_battle_cards(self):
        """Blue cards, stages and tokens are permanent; only the Cat can move away."""
        def on_step(before: GameState, after: GameState) -> None:
            for p in (0, 1):
                lost_cat = before.cat == p and after.cat != p
                assert after.score_of(p) + (before.rules.cat_vp if lost_cat else 0) >= before.score_of(p)

        for seed in range(40):
            random_game(seed, on_step=on_step)


# =====================================================================================
# Picks and chance draws
# =====================================================================================
class TestPicks:
    def test_pick_own_deck_places_top_and_reveals_next(self, base):
        s = base.apply_action(A.PICK_LEFT)
        assert s.is_chance() and s.ckind == C_REVEAL and s.cctx == 0
        outs = dict(s.chance_outcomes())
        assert abs(sum(outs.values()) - 1) < 1e-12
        assert abs(outs[K.wood] - 1 / 24) < 1e-12  # 2 wood in the Giza deck, one was on top
        s = s.apply_chance(K.clay)
        assert s.cards[0][K.wood] == 1
        assert s.deck_size[0] == 24 and s.deck_top[0] == K.clay and s.unseen[0][K.clay] == 0  # Giza's single clay
        assert s.to_move() == 1 and s.turn == 1 and is_main_pick_of(s, 1)
        assert_invariants(s)

    def test_pick_opponent_deck(self, base):
        s = base.apply_action(A.PICK_RIGHT)
        assert s.is_chance() and s.ckind == C_REVEAL and s.cctx == 1
        s = s.apply_chance(K.glass)
        assert s.cards[0][K.stone] == 1 and s.cards[1][K.stone] == 0
        assert s.deck_size[1] == 24 and s.deck_top[1] == K.glass and s.deck_size[0] == 25
        assert is_main_pick_of(s, 1)

    def test_pick_center_is_a_chance_draw_proportional_to_unseen(self, base):
        s = base.apply_action(A.PICK_CENTER)
        assert s.is_chance() and s.ckind == C_DRAW_CENTRAL and s.cctx == "main"
        comp = central_deck_counts()
        outs = dict(s.chance_outcomes())
        assert set(outs) == {k for k in range(NUM_KINDS) if comp[k]}
        for k, p in outs.items():
            assert abs(p - comp[k] / 60) < 1e-12
        assert abs(sum(outs.values()) - 1) < 1e-12
        assert s.deck_size[CENTRAL] == 60  # nothing leaves the deck before the outcome is known
        s = s.apply_chance(K.civ3)
        assert s.cards[0][K.civ3] == 1 and s.deck_size[CENTRAL] == 59
        assert s.unseen[CENTRAL][K.civ3] == comp[K.civ3] - 1
        assert s.deck_top[CENTRAL] == -1 and is_main_pick_of(s, 1)
        assert_invariants(s)

    def test_relative_actions_map_to_absolute_decks_for_player1(self, base):
        s = settle(base.apply_action(A.PICK_LEFT))
        assert s.mover == 1
        s2 = s.apply_action(A.PICK_LEFT)  # P1's own deck is deck 1
        assert s2.is_chance() and s2.ckind == C_REVEAL and s2.cctx == 1
        s3 = s.apply_action(A.PICK_RIGHT)
        assert s3.is_chance() and s3.ckind == C_REVEAL and s3.cctx == 0

    def test_pick_with_a_single_available_deck_is_auto_resolved(self, base):
        c = edit(base)
        drain_deck(c, 1)
        drain_deck(c, CENTRAL)
        s = run(c, ("turn_start",))
        # the only option (own deck) was taken automatically: we are already at the reveal
        assert s.is_chance() and s.ckind == C_REVEAL and s.cctx == 0
        assert s.mover == 0 and K.wood in [it[1] for it in s.queue if it[0] == "placed"]
        s = s.apply_chance(K.clay)
        assert s.cards[0][K.wood] == 1
        # P1's only option (P0's deck) is auto-resolved too: straight to the next reveal of deck 0
        assert s.mover == 1 and s.is_chance() and s.ckind == C_REVEAL and s.cctx == 0
        assert s.deck_size[0] == 23 and K.clay in in_transit(s)

    def test_turn_ends_with_no_cards_available_to_the_mover(self, base):
        """All decks empty is a game end; a *single* empty deck merely removes an option."""
        c = edit(base)
        drain_deck(c, 0)
        s = run(c, ("turn_start",))
        assert is_main_pick_of(s, 0)
        assert list(s.legal_actions()) == [A.PICK_RIGHT, A.PICK_CENTER]


# =====================================================================================
# The Cat pawn
# =====================================================================================
def holder_at_pick(base: GameState) -> GameState:
    """Player 0 holds the Cat at the start of their turn: resolve the peek to ``civ3``."""
    c = edit(base)
    c.cat = 0
    s = run(c, ("turn_start",))
    assert s.is_chance() and s.ckind == C_PEEK
    return s.apply_chance(K.civ3)


class TestCat:
    def test_taking_a_cat_card_moves_the_cat(self, base):
        s = take_card(base, K.civ2cat)
        assert s.cat == 0 and s.cards[0][K.civ2cat] == 1
        assert s.score_of(0) == 2 + base.rules.cat_vp

    def test_taking_a_cat_card_steals_the_cat(self, base):
        c = edit(base)
        c.cat = 1
        give(c, 1, K.civ2cat)
        s = take_card(c, K.civ2cat)
        assert s.cat == 0
        assert s.score_of(1) == 2 and s.score_of(0) == 2 + base.rules.cat_vp

    def test_plain_blue_card_does_not_move_the_cat(self, base):
        c = edit(base)
        c.cat = 1
        assert take_card(c, K.civ3).cat == 1

    def test_holder_peeks_at_turn_start(self, base):
        c = edit(base)
        c.cat = 0
        s = run(c, ("turn_start",))
        assert s.is_chance() and s.ckind == C_PEEK and s.to_move() == CHANCE
        outs = dict(s.chance_outcomes())
        comp = central_deck_counts()
        for k, p in outs.items():
            assert abs(p - comp[k] / 60) < 1e-12
        assert abs(sum(outs.values()) - 1) < 1e-12
        s = s.apply_chance(K.civ3)
        assert is_main_pick_of(s, 0)
        assert s.deck_top[CENTRAL] == K.civ3 and s.central_known_to == 1 << 0
        assert s.knows_central(0) and not s.knows_central(1)
        assert s.deck_size[CENTRAL] == 60 and s.unseen[CENTRAL][K.civ3] == comp[K.civ3] - 1
        assert list(s.legal_actions()) == [A.PICK_LEFT, A.PICK_RIGHT, A.PICK_CENTER]
        assert_invariants(s)

    def test_non_holder_does_not_peek(self, base):
        c = edit(base)
        c.cat = 1
        s = run(c, ("turn_start",))
        assert is_main_pick_of(s, 0) and s.deck_top[CENTRAL] == -1

    def test_holder_takes_the_known_central_card_without_a_chance_node(self, base):
        s = holder_at_pick(base)
        s = s.apply_action(A.PICK_CENTER)
        assert not s.is_chance()
        assert s.cards[0][K.civ3] == 1 and s.deck_size[CENTRAL] == 59
        assert s.deck_top[CENTRAL] == -1 and s.central_known_to == 0
        assert is_main_pick_of(s, 1)
        assert_invariants(s)

    def test_known_card_stays_known_when_holder_picks_elsewhere(self, base):
        s = settle(holder_at_pick(base).apply_action(A.PICK_LEFT))
        assert is_main_pick_of(s, 1)
        assert s.deck_top[CENTRAL] == K.civ3 and s.knows_central(0) and not s.knows_central(1)
        assert s.key() != settle(base.apply_action(A.PICK_LEFT)).key()

    def test_non_holder_central_pick_becomes_a_chance_draw_that_forgets(self, base):
        """Environment view (``observer == -1``): the non-holder's draw is a chance node."""
        s = settle(holder_at_pick(base).apply_action(A.PICK_LEFT))
        assert s.mover == 1 and s.knows_central(0) and s.observer == -1
        s = s.apply_action(A.PICK_CENTER)
        assert s.is_chance() and s.ckind == C_DRAW_CENTRAL
        # the known card was returned to the unseen multiset before sampling
        assert s.deck_top[CENTRAL] == -1 and s.central_known_to == 0
        assert s.unseen[CENTRAL][K.civ3] == central_deck_counts()[K.civ3]
        outs = dict(s.chance_outcomes())
        assert abs(outs[K.civ3] - central_deck_counts()[K.civ3] / 60) < 1e-12
        assert abs(sum(outs.values()) - 1) < 1e-12
        assert_invariants(s)
        s = s.apply_chance(K.civ3)
        assert s.cards[1][K.civ3] == 1 and s.deck_size[CENTRAL] == 59
        assert_invariants(s)

    def test_in_the_holders_belief_the_opponents_central_draw_is_deterministic(self, base):
        """``observer == holder``: the holder knows what the opponent will draw from the center."""
        c = edit(settle(holder_at_pick(base).apply_action(A.PICK_LEFT)))
        c.observer = 0
        s = c.apply_action(A.PICK_CENTER)  # P1 draws the card P0 knows: no draw chance node
        assert s.cards[1][K.civ3] == 1 and s.deck_size[CENTRAL] == 59 and s.observer == 0
        # ... and P0's next turn starts with a fresh peek at the new top card
        assert s.is_chance() and s.ckind == C_PEEK and s.mover == 0
        assert s.deck_top[CENTRAL] == -1 and s.central_known_to == 0
        assert_invariants(s)

    def test_no_second_peek_while_the_known_card_is_still_there(self, base):
        s = settle(holder_at_pick(base).apply_action(A.PICK_LEFT))  # P1 to move, central known to P0
        s = settle(s.apply_action(A.PICK_LEFT))  # P1 takes their own deck
        assert is_main_pick_of(s, 0)  # no C_PEEK node was exposed
        assert s.deck_top[CENTRAL] == K.civ3 and s.central_known_to == 1 << 0

    def test_holder_peeks_again_once_the_known_card_is_gone(self, base):
        s = holder_at_pick(base).apply_action(A.PICK_CENTER)  # P0 took civ3
        s = s.apply_action(A.PICK_LEFT)  # P1's turn: reveal of deck 1, then P0's peek
        assert s.is_chance() and s.ckind == C_REVEAL and s.cctx == 1
        s = s.apply_chance(most_likely(s))
        assert s.is_chance() and s.ckind == C_PEEK and s.mover == 0
        assert s.unseen[CENTRAL][K.civ3] == central_deck_counts()[K.civ3] - 1

    def test_new_holder_learns_the_card_the_old_holder_already_knows(self, base):
        # P0 (holder) knows civ3 is on top of the center and takes wood; a civ3 shows up on deck 0
        s = holder_at_pick(base).apply_action(A.PICK_LEFT).apply_chance(K.civ3)  # P1 to move
        c = edit(s)
        set_top(c, 1, K.civ2cat)
        s = settle(c.apply_action(A.PICK_LEFT))  # P1 takes the cat card from their own deck
        assert s.cat == 1 and is_main_pick_of(s, 0)
        assert s.deck_top[CENTRAL] == K.civ3 and s.knows_central(0) and not s.knows_central(1)
        s = settle(s.apply_action(A.PICK_LEFT))  # P0 takes the civ3 (no build) -> P1's turn as holder
        # the peek reveals the very card P0 saw: no chance node, both players now know it
        assert is_main_pick_of(s, 1)
        assert s.deck_top[CENTRAL] == K.civ3 and s.knows_central(0) and s.knows_central(1)
        assert s.central_known_to == 3
        assert_invariants(s)
        s = s.apply_action(A.PICK_CENTER)
        assert not s.is_chance() and s.cards[1][K.civ3] == 1 and s.central_known_to == 0

    def test_no_peek_when_the_central_deck_is_empty(self, base):
        c = edit(base)
        c.cat = 0
        drain_deck(c, CENTRAL)
        s = run(c, ("turn_start",))
        assert is_main_pick_of(s, 0) and list(s.legal_actions()) == [A.PICK_LEFT, A.PICK_RIGHT]

    def test_peek_with_a_single_kind_left_is_auto_resolved(self, base):
        c = edit(base)
        c.cat = 0
        set_deck(c, CENTRAL, None, [K.civ3, K.civ3])
        s = run(c, ("turn_start",))
        assert is_main_pick_of(s, 0)
        assert s.deck_top[CENTRAL] == K.civ3 and s.knows_central(0) and s.unseen[CENTRAL][K.civ3] == 1

    def test_cat_is_worth_two_points(self, base):
        c = edit(base)
        c.cat = 0
        assert c.score_of(0) == 2 and c.score_of(1) == 0
        c.cat = 1
        assert c.score_of(0) == 0 and c.score_of(1) == 2
        assert edit(first_pick(rules=RulesConfig(cat_vp=5))).score_of(0) == 0
        c5 = edit(first_pick(rules=RulesConfig(cat_vp=5)))
        c5.cat = 0
        assert c5.score_of(0) == 5

    def test_holder_extra_draw_from_center_uses_the_known_card(self, base):
        """The peeked card is physically still on top: an extra pick from the center takes it."""
        c = edit(holder_at_pick(base))
        give_token(c, 0, T.Urbanism)
        set_top(c, 0, K.wood)
        s = settle(c.apply_action(A.PICK_LEFT))  # wood -> Urbanism extra pick
        assert is_pick(s, f"token:{T.Urbanism}", 0)
        s = s.apply_action(A.PICK_CENTER)
        assert not s.is_chance() and s.cards[0][K.civ3] == 1 and s.central_known_to == 0


# =====================================================================================
# Mandatory construction and payment (Giza: 2 diff, 2 same, 3 diff, 3 same, 4 diff)
# =====================================================================================
def giza_with(base: GameState, stage: int, kinds: List[int], tokens: List[int] = ()) -> GameState:
    c = edit(base)
    c.built[0] = first_n(stage)
    give_many(c, 0, kinds)
    for t in tokens:
        give_token(c, 0, t)
    return c


class TestConstruction:
    def test_exact_identical_resources_build_immediately(self, base):
        c = giza_with(base, 1, [K.wood, K.wood])
        before = c.score_of(0)
        s = check(c)
        assert is_main_pick_of(s, 1)  # no payment decision was exposed
        assert s.num_stages(0) == 2
        assert s.cards[0][K.wood] == 0 and s.discard[K.wood] == 2
        assert s.score_of(0) == before + WONDERS[W.Giza].stages[1].vp
        assert_invariants(s)

    def test_exact_different_resources_build(self, base):
        s, taken = pay_through(check(giza_with(base, 0, [K.wood, K.stone])))
        assert s.num_stages(0) == 1 and is_main_pick_of(s, 1)
        assert s.cards[0][K.wood] == 0 and s.cards[0][K.stone] == 0
        assert s.discard[K.wood] == 1 and s.discard[K.stone] == 1
        assert s.score_of(0) == WONDERS[W.Giza].stages[0].vp
        assert_invariants(s)

    def test_exact_different_resources_do_not_expose_a_choice(self, base):
        s = check(giza_with(base, 0, [K.wood, K.stone]))
        assert s.dkind != D_PAY or s.mover != 0

    def test_identical_requirement_not_met_by_different_cards(self, base):
        s = check(giza_with(base, 1, [K.wood, K.stone]))
        assert s.num_stages(0) == 1 and s.cards[0][K.wood] == 1 and s.cards[0][K.stone] == 1
        assert is_main_pick_of(s, 1)

    def test_different_requirement_not_met_by_identical_cards(self, base):
        s = check(giza_with(base, 0, [K.wood, K.wood]))
        assert s.num_stages(0) == 0 and s.cards[0][K.wood] == 2

    def test_three_different_needs_three_distinct_resources(self, base):
        assert check(giza_with(base, 2, [K.wood, K.wood, K.stone])).num_stages(0) == 2
        s, _ = pay_through(check(giza_with(base, 2, [K.wood, K.clay, K.stone])))
        assert s.num_stages(0) == 3 and sum(s.cards[0]) == 0

    def test_surplus_cards_are_kept(self, base):
        s = check(giza_with(base, 1, [K.wood, K.wood, K.wood, K.civ3]))
        assert s.num_stages(0) == 2 and s.cards[0][K.wood] == 1 and s.cards[0][K.civ3] == 1

    def test_coins_are_wild_for_identical(self, base):
        s, _ = pay_through(check(giza_with(base, 1, [K.wood, K.coin])))
        assert s.num_stages(0) == 2 and s.discard[K.wood] == 1 and s.discard[K.coin] == 1

    def test_two_coins_build_identical_without_a_choice(self, base):
        s = check(giza_with(base, 1, [K.coin, K.coin]))
        assert is_main_pick_of(s, 1) and s.num_stages(0) == 2 and s.discard[K.coin] == 2

    def test_coins_are_wild_for_different(self, base):
        s = check(giza_with(base, 0, [K.coin, K.coin]))
        assert s.num_stages(0) == 1 and s.discard[K.coin] == 2
        s, _ = pay_through(check(giza_with(base, 2, [K.wood, K.coin, K.coin])))
        assert s.num_stages(0) == 3 and sum(s.cards[0]) == 0

    def test_mandatory_even_with_a_coin(self, base):
        """The rulebook: coins *must* replace missing resources (BGG 'Mandatory or not?')."""
        s, _ = pay_through(check(giza_with(base, 0, [K.wood, K.coin])))
        assert s.num_stages(0) == 1

    def test_payment_choice_when_several_payments_are_valid(self, base):
        """wood / stone / clay for "2 different" are three multisets.  Payments are sequences of codes
        in canonical (non-decreasing) order -- wood 0 < stone 1 < clay 2 -- so a decision only
        branches between genuinely different multisets and clay can never be the *first* card."""
        s = check(giza_with(base, 0, [K.wood, K.stone, K.clay]))
        assert s.to_move() == 0 and s.dkind == D_PAY and s.dctx[1] == ()
        assert list(s.legal_actions()) == [A.PAY_BASE + 0, A.PAY_BASE + 1]
        assert_node_sane(s)
        s1 = s.apply_action(A.PAY_BASE + 0)
        assert s1.dkind == D_PAY and s1.dctx[1] == (0,)
        assert list(s1.legal_actions()) == [A.PAY_BASE + 1, A.PAY_BASE + 2]
        assert_node_sane(s1)
        s2 = s1.apply_action(A.PAY_BASE + 1)
        assert s2.num_stages(0) == 1 and is_main_pick_of(s2, 1)
        assert s2.cards[0][K.clay] == 1 and s2.discard[K.wood] == 1 and s2.discard[K.stone] == 1
        assert_invariants(s2)
        # stone first: wood (a lower code) may not follow, clay is forced -> auto-resolved, wood is kept
        s3 = s.apply_action(A.PAY_BASE + 1)
        assert s3.num_stages(0) == 1 and is_main_pick_of(s3, 1)
        assert s3.cards[0][K.wood] == 1 and s3.discard[K.stone] == 1 and s3.discard[K.clay] == 1
        assert_invariants(s3)

    def test_choice_between_coin_and_grey(self, base):
        """wood / wood / coin for "2 identical": {wood, wood} or {wood, coin}.  The first wood is forced
        (a coin first could not be followed by a wood), the second card is the real choice."""
        s = check(giza_with(base, 1, [K.wood, K.wood, K.coin]))
        assert s.to_move() == 0 and s.dkind == D_PAY and s.dctx[1] == (0,)
        assert list(s.legal_actions()) == [A.PAY_BASE + 0, A.PAY_COIN]
        assert_node_sane(s)
        kept_coin = s.apply_action(A.PAY_BASE + 0)
        assert kept_coin.num_stages(0) == 2 and kept_coin.cards[0][K.coin] == 1 and kept_coin.cards[0][K.wood] == 0
        kept_wood = s.apply_action(A.PAY_COIN)
        assert kept_wood.num_stages(0) == 2 and kept_wood.cards[0][K.coin] == 0 and kept_wood.cards[0][K.wood] == 1
        assert is_main_pick_of(kept_coin, 1) and is_main_pick_of(kept_wood, 1)

    def test_grey_is_never_offered_after_a_coin(self, base):
        """Coin codes (5, 6) are above every grey code: wood / coin / coin for "2 different" is either
        {wood, coin} or {coin, coin}; once a coin is paid the wood is no longer an option."""
        s = check(giza_with(base, 0, [K.wood, K.coin, K.coin]))
        assert s.dkind == D_PAY and s.dctx[1] == () and list(s.legal_actions()) == [A.PAY_BASE + 0, A.PAY_COIN]
        coins = s.apply_action(A.PAY_COIN)  # the second coin is forced
        assert coins.num_stages(0) == 1 and coins.cards[0][K.wood] == 1 and coins.cards[0][K.coin] == 0
        wood = s.apply_action(A.PAY_BASE + 0)  # a coin is forced (the only kind left)
        assert wood.num_stages(0) == 1 and wood.cards[0][K.wood] == 0 and wood.cards[0][K.coin] == 1
        assert is_main_pick_of(coins, 1) and is_main_pick_of(wood, 1)
        assert_invariants(coins)

    def test_identical_stage_coin_after_grey_never_dead_ends(self, base):
        """3 identical (Giza stage 4): wood x2 + coin has the single payment {wood, wood, coin} and
        wood x3 + coin has {wood x3} / {wood, wood, coin}; every payment branch must build the stage."""
        for kinds in ([K.wood, K.wood, K.coin], [K.wood, K.wood, K.wood, K.coin]):
            s = check(giza_with(base, 3, kinds))
            for leaf in payment_leaves(s):
                assert leaf.num_stages(0) == 4, leaf.describe()

    def test_coins_only_when_no_grey_possible_if_free_choice_disabled(self):
        b = first_pick(rules=RulesConfig(coins_free_choice=False))
        s = check(giza_with(b, 0, [K.wood, K.stone, K.coin]))
        assert s.dkind == D_PAY and A.PAY_COIN not in s.legal_actions()
        s = s.apply_action(A.PAY_BASE + 0)  # stone is then forced (no coin offered)
        assert s.num_stages(0) == 1 and s.cards[0][K.coin] == 1
        s = check(giza_with(b, 0, [K.wood, K.coin]))  # coin needed -> offered
        assert s.num_stages(0) == 1 and s.cards[0][K.coin] == 0

    def test_engineering_ignores_identical_and_different(self, base):
        assert check(giza_with(base, 1, [K.wood, K.stone])).num_stages(0) == 1
        s, _ = pay_through(check(giza_with(base, 1, [K.wood, K.stone], [T.Engineering])))
        assert s.num_stages(0) == 2 and s.discard[K.wood] == 1 and s.discard[K.stone] == 1
        assert check(giza_with(base, 0, [K.wood, K.wood], [T.Engineering])).num_stages(0) == 1
        s, _ = pay_through(check(giza_with(base, 4, [K.wood, K.wood, K.coin, K.stone], [T.Engineering])))
        assert s.num_stages(0) == 5

    def test_economy_one_coin_worth_two(self, base):
        assert check(giza_with(base, 0, [K.coin])).num_stages(0) == 0
        s = check(giza_with(base, 0, [K.coin], [T.Economy]))
        assert s.num_stages(0) == 1 and s.discard[K.coin] == 1 and s.cards[0][K.coin] == 0
        s = check(giza_with(base, 1, [K.coin], [T.Economy]))  # identical too
        assert s.num_stages(0) == 2

    def test_economy_at_most_once_per_build(self, base):
        """Giza's 5th stage (4 different) paid with coins only: one coin may be doubled, never two."""
        assert check(giza_with(base, 4, [K.coin, K.coin], [T.Economy])).num_stages(0) == 4  # 2 + 1 < 4
        # 3 coins: {coin, coin, coin x2} is the only payment -> no decision, the game ends (5th stage)
        s = check(giza_with(base, 4, [K.coin, K.coin, K.coin], [T.Economy]))
        assert s.is_terminal() and s.num_stages(0) == 5 and s.discard[K.coin] == 3 and s.cards[0][K.coin] == 0
        assert s.econ_used
        # 4 coins: {coin x4} or {coin x3, one doubled}; the doubled coin (highest code) is the last card
        c = giza_with(base, 4, [K.coin] * 4, [T.Economy])
        s = check(c)
        assert s.to_move() == 0 and s.dkind == D_PAY and s.dctx[1] == (5, 5)
        assert list(s.legal_actions()) == [A.PAY_COIN, A.PAY_COIN2]
        assert_node_sane(s)
        doubled = s.apply_action(A.PAY_COIN2)
        assert doubled.is_terminal() and doubled.num_stages(0) == 5 and doubled.cards[0][K.coin] == 1 and doubled.econ_used
        plain = s.apply_action(A.PAY_COIN)  # the 4th coin is forced
        assert plain.is_terminal() and plain.num_stages(0) == 5 and plain.cards[0][K.coin] == 0 and not plain.econ_used
        # a second doubled coin is never offered: not after one (nothing follows code 6) nor for 1 missing
        assert A.PAY_COIN2 not in c._pay_options(4, (5, 6)) and c._pay_options(4, (5, 5, 5)) == [A.PAY_COIN]

    def test_economy_offered_only_when_two_more_are_needed(self, base):
        """The doubled coin is the last code of a payment, so it is offered exactly when 2 resources
        are still missing (it would overpay for 1; with 3+ missing nothing could follow it)."""
        c = giza_with(base, 2, [K.wood, K.coin, K.coin], [T.Economy])  # cost 3 different
        s = check(c)
        assert s.dkind == D_PAY and s.dctx[1] == () and list(s.legal_actions()) == [A.PAY_BASE + 0, A.PAY_COIN]
        s = s.apply_action(A.PAY_BASE + 0)  # 2 missing: plain coin or doubled coin
        assert s.dkind == D_PAY and s.dctx[1] == (0,) and list(s.legal_actions()) == [A.PAY_COIN, A.PAY_COIN2]
        assert_node_sane(s)
        two = s.apply_action(A.PAY_COIN2)
        assert two.num_stages(0) == 3 and two.discard[K.coin] == 1 and two.discard[K.wood] == 1 and two.cards[0][K.coin] == 1
        one = s.apply_action(A.PAY_COIN)  # 1 missing: a 2-coin payment would overpay, the plain coin is forced
        assert one.num_stages(0) == 3 and one.discard[K.coin] == 2 and one.discard[K.wood] == 1 and one.cards[0][K.coin] == 0
        assert c._pay_options(2, (0, 5)) == [A.PAY_COIN]
        # coin first: the wood may not follow a coin, the doubled coin completes the payment
        coin_first = check(c).apply_action(A.PAY_COIN)
        assert coin_first.num_stages(0) == 3 and coin_first.cards[0][K.wood] == 1 and coin_first.cards[0][K.coin] == 0

    def test_economy_is_once_per_turn_not_once_per_build(self, base):
        """Two builds in one turn: the doubled coin is available only once (rulebook: each token once
        per turn) under the default ``RulesConfig.economy_once_per_build = False``."""
        s = check(giza_with(base, 0, [K.coin, K.coin], [T.Economy]))
        assert s.dkind == D_PAY and set(s.legal_actions()) == {A.PAY_COIN, A.PAY_COIN2}
        s = s.apply_action(A.PAY_COIN2)
        assert s.num_stages(0) == 1 and s.cards[0][K.coin] == 1
        assert is_main_pick_of(s, 1)  # stage 2 (2 identical) is NOT built: 1 coin, Economy spent

    def test_economy_is_reset_at_the_next_turn(self, base):
        # the Architecture extra pick keeps the turn open after the build so that econ_used can be observed
        s = check(giza_with(base, 0, [K.coin], [T.Economy, T.Architecture]))
        assert s.num_stages(0) == 1 and s.cards[0][K.coin] == 0 and s.econ_used
        assert is_pick(s, f"token:{T.Architecture}", 0)
        s = s.apply_action(A.SKIP)
        assert is_main_pick_of(s, 1) and not s.econ_used  # per turn: cleared for the opponent's turn
        s = settle(s.apply_action(A.PICK_LEFT))  # P1's turn passes
        assert is_main_pick_of(s, 0) and not s.econ_used
        c = edit(s)
        set_top(c, 0, K.coin)
        s = settle(c.apply_action(A.PICK_LEFT))  # a single coin builds stage 2 (2 identical) again
        assert s.num_stages(0) == 2 and s.cards[0][K.coin] == 0 and s.econ_used

    def test_several_stages_in_one_turn(self, base):
        s = check(giza_with(base, 0, [K.wood, K.stone, K.clay, K.clay]))
        assert s.dkind == D_PAY and list(s.legal_actions()) == [A.PAY_BASE + 0, A.PAY_BASE + 1]  # clay cannot come first
        s = s.apply_action(A.PAY_BASE + 0)
        assert s.dkind == D_PAY and list(s.legal_actions()) == [A.PAY_BASE + 1, A.PAY_BASE + 2]
        s = s.apply_action(A.PAY_BASE + 1)  # stage 1 with wood + stone, then clay x2 builds stage 2 (2 identical) at once
        assert s.num_stages(0) == 2 and sum(s.cards[0]) == 0 and is_main_pick_of(s, 1)
        assert s.score_of(0) == WONDERS[W.Giza].stages[0].vp + WONDERS[W.Giza].stages[1].vp

    def test_build_triggered_by_a_taken_card(self, base):
        c = edit(base)
        give(c, 0, K.wood)
        s = take_card(c, K.coin)  # wood + coin is the only payment: built at once, no decision
        assert s.num_stages(0) == 1 and s.cards[0][K.wood] == 0 and s.cards[0][K.coin] == 0 and is_main_pick_of(s, 1)
        assert s.discard[K.wood] == 1 and s.discard[K.coin] == 1
        c = edit(base)
        give_many(c, 0, [K.wood, K.stone])
        s = take_card(c, K.clay)  # three greys for "2 different": the payment choice is exposed at once
        assert s.to_move() == 0 and s.dkind == D_PAY and s.dctx[1] == ()
        s, _ = pay_through(s)
        assert s.num_stages(0) == 1 and sum(s.cards[0]) == 1 and is_main_pick_of(s, 1)

    def test_no_build_after_the_fifth_stage(self, base):
        c = giza_with(base, 5, [K.stone, K.stone, K.stone, K.stone])
        s = check(c)
        assert s.num_stages(0) == 5 and s.cards[0][K.stone] == 4 and is_main_pick_of(s, 1)

    def test_stage_vp_bookkeeping(self, base):
        c = edit(base)
        for n in range(6):
            c.built[0] = first_n(n)
            assert c.score_of(0) == sum(st.vp for st in WONDERS[W.Giza].stages[:n])


# =====================================================================================
# Wonder stage effects
# =====================================================================================
def wonder_at_stage2(w: int, base_kw: dict = None, rules: RulesConfig = None) -> GameState:
    """Player 0 plays Wonder ``w`` with stage 1 built and the 2 identical cards for stage 2."""
    b = first_pick(w, W.Giza, rules=rules, **(base_kw or {}))
    c = edit(b)
    c.built[0] = first_n(1)
    give_many(c, 0, [K.stone, K.stone])
    return c


class TestStageEffects:
    def test_rhodes_shields_are_permanent(self):
        # Rhodes' first shield is on the 2-different foundation (S1), the second on the 3-identical stage (S4)
        b = first_pick(W.Rhodes, W.Giza)
        c = edit(b)
        give_many(c, 0, [K.wood, K.stone])
        s = check(c)
        assert s.built[0] == 0b00001 and s.wonder_shields[0] == 1 and s.shields_of(0) == 1
        c = edit(s)
        c.mover, c.battle_pending, c.conflict = 0, True, 3
        s2 = run(c, ("end_turn",))
        assert s2.mil_tokens == [1, 0] and s2.wonder_shields[0] == 1 and s2.shields_of(0) == 1
        c = edit(s)
        c.built[0] = first_n(2)  # the 2-identical foundation gives no shield ...
        c.wonder_shields[0] = 1
        give_many(c, 0, [K.clay, K.clay, K.clay])
        s3 = check(c, p=0)  # ... S3 (3 different) is not affordable with 3 clay; nothing happens
        assert s3.built[0] == first_n(2) and s3.wonder_shields[0] == 1
        c = edit(s)
        c.built[0] = first_n(3)
        c.wonder_shields[0] = 1
        give_many(c, 0, [K.clay, K.clay, K.clay])
        s4 = check(c, p=0)
        assert s4.built[0] == first_n(4) and s4.wonder_shields[0] == 2 and s4.shields_of(0) == 2

    def test_rhodes_two_foundations_are_a_choice(self):
        """Rhodes may start with either foundation; with cards for both the player chooses (D_STAGE)."""
        b = first_pick(W.Rhodes, W.Giza)
        c = edit(b)
        give_many(c, 0, [K.wood, K.wood, K.stone])
        s = check(c)
        assert s.to_move() == 0 and s.dkind == D_STAGE and s.dctx == (0, 1)
        assert list(s.legal_actions()) == [A.STAGE_BASE, A.STAGE_BASE + 1]
        s1 = settle(s.apply_action(A.STAGE_BASE))       # 2 different: wood + stone
        assert s1.built[0] == 0b00001 and s1.wonder_shields[0] == 1 and s1.cards[0][K.wood] == 1
        assert is_main_pick_of(s1, 1)                     # the leftover wood cannot pay the 2-identical stage
        s2 = settle(s.apply_action(A.STAGE_BASE + 1))   # 2 identical: wood + wood
        assert s2.built[0] == 0b00010 and s2.wonder_shields[0] == 0 and s2.cards[0][K.stone] == 1
        assert is_main_pick_of(s2, 1)
        # with cards for both foundations the mandatory check repeats and both are built in one turn
        c = edit(b)
        give_many(c, 0, [K.wood, K.wood, K.stone, K.clay])
        s = check(c)
        assert s.dkind == D_STAGE
        s3 = settle(s.apply_action(A.STAGE_BASE + 1))
        assert s3.built[0] == 0b00011 and s3.num_stages(0) == 2 and s3.wonder_shields[0] == 1 and sum(s3.cards[0]) == 0

    def test_stage_prerequisites_gate_construction(self):
        """Rhodes S3 needs both foundations; Babylon may build its 4-different stage before its 3-identical one;
        Ephesus opens three stages after its foundation; Olympia's S4 needs both S2 and S3."""
        b = first_pick(W.Rhodes, W.Giza)
        c = edit(b)
        c.built[0] = 0b00001
        give_many(c, 0, [K.wood, K.stone, K.clay])  # would pay S3 (3 different) but S2 is not built
        s = check(c)
        assert s.built[0] == 0b00001 and is_main_pick_of(s, 1)
        c = edit(b)
        c.built[0] = 0b00011
        give_many(c, 0, [K.wood, K.stone, K.clay])
        s = check(c)
        assert s.built[0] == 0b00111 and s.score_of(0) == 4 + 4 + 5
        # Babylon: S1..S3 built, 4 different cards -> S5 is available and built (S4 is not affordable)
        b = first_pick(W.Babylon, W.Giza)
        c = edit(b)
        c.built[0] = 0b00111
        give_many(c, 0, [K.wood, K.stone, K.clay, K.glass])
        s = check(c)
        assert s.built[0] == 0b10111 and not s.is_terminal() and s.score_of(0) == 3 + 0 + 5 + 7
        # Ephesus: after S1 three stages are available; 3 identical cards build S4 (effect: central card)
        b = first_pick(W.Ephesus, W.Giza)
        c = edit(b)
        c.built[0] = 0b00001
        assert c.available_stages(0) == (1, 2, 3)
        give_many(c, 0, [K.papyrus, K.papyrus, K.papyrus])
        s = check(c)
        assert s.dkind == D_STAGE and s.dctx == (1, 3)  # 2 identical (S2) or 3 identical (S4)
        s = settle(s.apply_action(A.STAGE_BASE + 3))
        assert (s.built[0] >> 3) & 1 and s.cards[0][K.papyrus] == 0
        # Olympia: S4 (3 identical, effect) requires S2 and S3
        b = first_pick(W.Olympia, W.Giza)
        c = edit(b)
        c.built[0] = 0b00011
        give_many(c, 0, [K.stone, K.stone, K.stone])
        s = check(c)
        assert s.built[0] == 0b00011 and is_main_pick_of(s, 1)
        c = edit(b)
        c.built[0] = 0b00111
        give_many(c, 0, [K.stone, K.stone, K.stone])
        s = check(c)
        assert s.built[0] == 0b01111 and s.score_of(0) == 3 + 2 + 5 + 5

    def test_babylon_stage_grants_a_token_choice(self):
        s = check(wonder_at_stage2(W.Babylon))
        assert s.to_move() == 0 and s.dkind == D_TOKEN and s.dctx == "babylon"
        assert list(s.legal_actions()) == sorted(A.TOKEN_BASE + t for t in DEFAULT_FACEUP) + [A.TOKEN_BLIND]
        s = s.apply_action(A.TOKEN_BASE + T.Strategy)
        assert s.tokens[0][T.Strategy] == 1 and T.Strategy not in s.faceup
        assert s.is_chance() and s.ckind == C_TOKEN_REVEAL  # refill
        s = s.apply_chance(T.Tactics)
        assert sorted(s.faceup) == sorted([T.Education, T.Decor, T.Tactics]) and s.prog_stack == 11
        assert is_main_pick_of(s, 1)
        assert_invariants(s)

    def test_babylon_blind_choice(self):
        s = check(wonder_at_stage2(W.Babylon)).apply_action(A.TOKEN_BLIND)
        assert s.is_chance() and s.ckind == C_TOKEN_BLIND
        s = s.apply_chance(T.Culture)
        assert s.tokens[0][T.Culture] == 1 and len(s.faceup) == 3 and s.prog_stack == 11

    def test_babylon_with_no_tokens_left(self):
        c = wonder_at_stage2(W.Babylon)
        exhaust_tokens(c)
        s = check(c)
        assert s.num_stages(0) == 2 and sum(s.tokens[0]) == 0 and is_main_pick_of(s, 1)

    def test_alexandria_picks_from_any_deck(self):
        s = check(wonder_at_stage2(W.Alexandria))
        assert is_pick(s, "alexandria", 0) and s.dctx == ("alexandria", False, (0, 1, 2))
        assert list(s.legal_actions()) == [A.PICK_LEFT, A.PICK_RIGHT, A.PICK_CENTER]
        s2 = settle(s.apply_action(A.PICK_RIGHT))
        assert s2.cards[0][K.stone] == 1 and s2.deck_size[1] == 24  # the opponent's visible top
        s3 = s.apply_action(A.PICK_CENTER)
        assert s3.is_chance() and s3.ckind == C_DRAW_CENTRAL and s3.cctx == "alexandria"
        assert_invariants(s3)

    def test_alexandria_effect_optional_rule(self):
        s = check(wonder_at_stage2(W.Alexandria, rules=RulesConfig(wonder_effect_optional=True)))
        assert is_pick(s, "alexandria", 0) and A.SKIP in s.legal_actions()
        s = s.apply_action(A.SKIP)
        assert is_main_pick_of(s, 1) and sum(s.cards[0]) == 0

    def test_ephesus_draws_the_central_card(self):
        c = edit(first_pick(W.Ephesus, W.Giza))  # S1 built; S2 (2 identical) draws the central card
        c.built[0] = first_n(1)
        give_many(c, 0, [K.stone, K.stone])
        s, _ = pay_through(check(c))
        assert s.num_stages(0) == 2
        assert s.is_chance() and s.ckind == C_DRAW_CENTRAL and s.cctx == "ephesus"
        before = s.deck_size[CENTRAL]
        s = s.apply_chance(K.civ3)
        assert s.cards[0][K.civ3] == 1 and s.deck_size[CENTRAL] == before - 1 and is_main_pick_of(s, 1)
        assert_invariants(s)

    def test_ephesus_with_empty_central_deck(self):
        c = edit(first_pick(W.Ephesus, W.Giza))
        give_many(c, 0, [K.wood, K.stone], frm=0)
        drain_deck(c, CENTRAL)
        s, _ = pay_through(check(c))
        assert s.num_stages(0) == 1 and sum(s.cards[0]) == 0 and is_main_pick_of(s, 1)

    def test_olympia_takes_both_wonder_deck_tops(self):
        c = wonder_at_stage2(W.Olympia, {"tops": (K.clay, K.papyrus)})
        s = check(c)
        assert s.num_stages(0) == 2
        assert s.is_chance() and s.ckind == C_REVEAL and s.cctx == 0  # own deck first
        s = s.apply_chance(K.glass)
        assert s.cards[0][K.clay] == 1
        assert s.is_chance() and s.ckind == C_REVEAL and s.cctx == 1
        s = s.apply_chance(K.glass)
        assert s.cards[0][K.papyrus] == 1 and s.deck_size[0] == 24 and s.deck_size[1] == 24
        assert is_main_pick_of(s, 1)
        assert_invariants(s)

    def test_olympia_skips_an_empty_deck(self):
        c = wonder_at_stage2(W.Olympia, {"tops": (K.clay, K.papyrus)})
        drain_deck(c, 1)
        s = settle(check(c))
        assert s.cards[0][K.clay] == 1 and s.cards[0][K.papyrus] == 0 and s.deck_size[0] == 24
        c2 = wonder_at_stage2(W.Olympia)
        drain_deck(c2, 0)
        drain_deck(c2, 1)
        s2 = check(c2)
        assert s2.num_stages(0) == 2 and sum(s2.cards[0]) == 0

    def test_olympia_extra_cards_can_trigger_a_build(self):
        c = wonder_at_stage2(W.Olympia, {"tops": (K.wood, K.clay)})
        give(c, 0, K.glass)  # stage 3 needs 3 different: glass + wood + clay from the Olympia cards
        s = settle(check(c))
        s, _ = pay_through(s)
        assert s.num_stages(0) == 3

    # ---- Halicarnassus ------------------------------------------------------------
    def hali(self, deck0_top, deck0_unseen, rules=None) -> GameState:
        c = wonder_at_stage2(W.Halicarnassus, rules=rules)
        set_deck(c, 0, deck0_top, deck0_unseen)
        s = check(c)
        assert s.num_stages(0) == 2
        return s

    def test_halicarnassus_full_flow(self):
        s = self.hali(K.clay, [K.papyrus, K.glass, K.coin, K.civ3, K.tablet, K.gear])
        assert s.to_move() == 0 and s.dkind == D_HALI_DECK and s.dctx == (0, 1)
        assert list(s.legal_actions()) == [A.PICK_LEFT, A.PICK_RIGHT]
        s = s.apply_action(A.PICK_LEFT)
        expected = [(4, (K.clay,)), (3, (K.clay, K.papyrus)), (2, (K.clay, K.papyrus, K.glass)),
                    (1, (K.clay, K.papyrus, K.glass, K.coin))]
        for outcome, (remaining, revealed) in zip((K.papyrus, K.glass, K.coin, K.civ3), expected):
            assert s.is_chance() and s.ckind == C_HALI_REVEAL and s.cctx == (0, remaining, revealed)
            outs = dict(s.chance_outcomes())
            assert set(outs) == {k for k in range(NUM_KINDS) if s.unseen[0][k]}
            assert abs(sum(outs.values()) - 1) < 1e-12
            assert_invariants(s)
            s = s.apply_chance(outcome)
        assert s.to_move() == 0 and s.dkind == D_HALI_CHOOSE
        assert s.dctx == (0, (K.clay, K.papyrus, K.glass, K.coin, K.civ3))
        assert list(s.legal_actions()) == sorted(A.HALI_BASE + k for k in (K.clay, K.papyrus, K.glass, K.coin, K.civ3))
        assert s.deck_top[0] == -1 and s.deck_size[0] == 7
        assert_invariants(s)
        s = s.apply_action(A.HALI_BASE + K.coin)
        assert s.deck_size[0] == 6
        for k in (K.clay, K.papyrus, K.glass, K.civ3, K.tablet, K.gear):
            assert s.unseen[0][k] == 1  # the other 4 revealed cards went back into the deck
        assert s.is_chance() and s.ckind == C_REVEAL and s.cctx == 0  # a new top is revealed
        assert in_transit(s) == [K.coin]  # the kept card is placed right after the reveal
        assert all(abs(p - 1 / 6) < 1e-12 for _, p in s.chance_outcomes())
        assert_invariants(s)
        s = s.apply_chance(K.tablet)
        assert s.cards[0][K.coin] == 1 and s.deck_top[0] == K.tablet and is_main_pick_of(s, 1)
        assert_invariants(s)

    def test_halicarnassus_short_deck_reveals_fewer_cards(self):
        s = self.hali(K.clay, [K.papyrus, K.glass]).apply_action(A.PICK_LEFT)
        assert s.is_chance() and s.ckind == C_HALI_REVEAL and s.cctx == (0, 2, (K.clay,))
        s = s.apply_chance(K.papyrus)  # the last reveal is forced (only glass left) -> auto
        assert s.dkind == D_HALI_CHOOSE and s.dctx == (0, (K.clay, K.papyrus, K.glass))
        assert len(s.legal_actions()) == 3
        s = s.apply_action(A.HALI_BASE + K.glass)
        assert in_transit(s) == [K.glass] and s.deck_size[0] == 2
        assert s.is_chance() and s.ckind == C_REVEAL and sorted(dict(s.chance_outcomes())) == [K.clay, K.papyrus]
        s = s.apply_chance(K.clay)
        assert s.cards[0][K.glass] == 1 and s.deck_top[0] == K.clay and s.unseen[0][K.papyrus] == 1

    def test_halicarnassus_single_card_deck_has_no_reveal(self):
        s = self.hali(K.clay, [])
        assert s.dkind == D_HALI_DECK
        s = s.apply_action(A.PICK_LEFT)
        assert not (s.is_chance() and s.ckind == C_HALI_REVEAL)
        assert s.cards[0][K.clay] == 1 and s.deck_size[0] == 0 and s.deck_top[0] == -1
        assert is_main_pick_of(s, 1)
        assert_invariants(s)

    def test_halicarnassus_identical_cards_auto_resolve(self):
        s = self.hali(K.wood, [K.wood] * 4).apply_action(A.PICK_LEFT)
        # every reveal and the choice have a single outcome: all auto-resolved, no node exposed
        assert s.cards[0][K.wood] == 1 and s.deck_size[0] == 4 and s.deck_top[0] == K.wood
        assert s.unseen[0][K.wood] == 3 and is_main_pick_of(s, 1)
        assert_invariants(s)

    def test_halicarnassus_with_one_deck_available_skips_the_deck_choice(self):
        c = wonder_at_stage2(W.Halicarnassus)
        drain_deck(c, 1)
        s = check(c)
        assert s.is_chance() and s.ckind == C_HALI_REVEAL and s.cctx[0] == 0 and s.cctx[1] == 4

    def test_halicarnassus_with_both_decks_empty(self):
        c = wonder_at_stage2(W.Halicarnassus)
        drain_deck(c, 0)
        drain_deck(c, 1)
        s = check(c)
        # no effect; P1's main pick has a single option (the center) and is auto-resolved into a draw
        assert s.num_stages(0) == 2 and s.mover == 1 and s.is_chance() and s.ckind == C_DRAW_CENTRAL

    def test_halicarnassus_optional_rule(self):
        s = self.hali(K.clay, [K.papyrus], rules=RulesConfig(wonder_effect_optional=True))
        assert s.dkind == D_HALI_DECK and A.SKIP in s.legal_actions()
        assert is_main_pick_of(s.apply_action(A.SKIP), 1)

    def test_halicarnassus_opponent_deck(self):
        s = self.hali(K.clay, [K.papyrus]).apply_action(A.PICK_RIGHT)
        assert s.is_chance() and s.ckind == C_HALI_REVEAL and s.cctx[0] == 1 and s.cctx[2] == (K.stone,)

    def test_halicarnassus_choice_is_recorded_for_the_environment(self):
        """``hali_event`` = (deck, window size, kept kind) on the state produced by the transition that
        resolved the choice -- also when the choice was auto-resolved -- and on that state only."""
        s = self.hali(K.clay, [K.papyrus, K.glass])
        assert s.hali_event is None
        s = s.apply_action(A.PICK_LEFT).apply_chance(K.papyrus)  # the last reveal (glass) is forced
        assert s.dkind == D_HALI_CHOOSE and s.hali_event is None
        s = s.apply_action(A.HALI_BASE + K.glass)
        assert s.hali_event == (0, 3, K.glass)
        assert s.is_chance() and s.ckind == C_REVEAL
        assert s.apply_chance(K.clay).hali_event is None  # cleared by the next transition
        # five identical cards: reveals and choice are auto-resolved inside one transition
        auto = self.hali(K.wood, [K.wood] * 4).apply_action(A.PICK_LEFT)
        assert auto.hali_event == (0, 5, K.wood) and is_main_pick_of(auto, 1)


# =====================================================================================
# Science sets and Progress-token acquisition
# =====================================================================================
def with_science(base: GameState, kinds: List[int]) -> GameState:
    c = edit(base)
    give_many(c, 0, kinds)
    return c


class TestScience:
    def test_two_identical_symbols_force_a_token(self, base):
        s = check(with_science(base, [K.tablet, K.tablet]))
        assert s.to_move() == 0 and s.dkind == D_TOKEN and s.dctx == "science"
        assert s.cards[0][K.tablet] == 0 and s.discard[K.tablet] == 2  # used cards are discarded
        assert list(s.legal_actions()) == sorted(A.TOKEN_BASE + t for t in DEFAULT_FACEUP) + [A.TOKEN_BLIND]
        assert_invariants(s)

    def test_three_different_symbols_force_a_token(self, base):
        s = check(with_science(base, [K.tablet, K.gear, K.compass]))
        assert s.dkind == D_TOKEN and all(s.cards[0][k] == 0 for k in GREEN_KINDS)
        assert all(s.discard[k] == 1 for k in GREEN_KINDS)

    def test_incomplete_sets_are_kept(self, base):
        s = check(with_science(base, [K.tablet, K.gear]))
        assert is_main_pick_of(s, 1) and s.cards[0][K.tablet] == 1 and s.cards[0][K.gear] == 1

    def test_choice_between_pair_and_triple(self, base):
        s = check(with_science(base, [K.tablet, K.tablet, K.gear, K.compass]))
        assert s.to_move() == 0 and s.dkind == D_SCIENCE
        assert list(s.legal_actions()) == [A.SCI_PAIR_BASE + 0, A.SCI_TRIPLE]
        pair = s.apply_action(A.SCI_PAIR_BASE + 0)
        assert pair.dkind == D_TOKEN and [pair.cards[0][k] for k in GREEN_KINDS] == [0, 1, 1]
        triple = s.apply_action(A.SCI_TRIPLE)
        assert triple.dkind == D_TOKEN and [triple.cards[0][k] for k in GREEN_KINDS] == [1, 0, 0]

    def test_science_token_extra_pick_creates_a_pair_vs_triple_choice(self, base):
        """Through the public API a pair-vs-triple choice needs a 4th green card in one turn, which
        only the Science token's extra pick provides: tablet + gear in hand, a tablet is taken, the
        extra pick brings the compass (the science check runs after the extra pick)."""
        c = edit(base)
        give_token(c, 0, T.Science)
        give_many(c, 0, [K.tablet, K.gear])
        set_top(c, 1, K.compass)
        s = take_card(c, K.tablet)
        assert is_pick(s, f"token:{T.Science}", 0) and s.cards[0][K.tablet] == 2
        s = settle(s.apply_action(A.PICK_RIGHT))
        assert s.to_move() == 0 and s.dkind == D_SCIENCE and (s.tokens_used >> T.Science) & 1
        assert list(s.legal_actions()) == [A.SCI_PAIR_BASE + 0, A.SCI_TRIPLE]
        assert_node_sane(s)
        pair = s.apply_action(A.SCI_PAIR_BASE + 0)
        assert pair.dkind == D_TOKEN and [pair.cards[0][k] for k in GREEN_KINDS] == [0, 1, 1]
        triple = s.apply_action(A.SCI_TRIPLE)
        assert triple.dkind == D_TOKEN and [triple.cards[0][k] for k in GREEN_KINDS] == [1, 0, 0]
        assert_invariants(pair)
        assert_invariants(triple)

    def test_choice_between_two_pairs(self, base):
        s = check(with_science(base, [K.gear, K.gear, K.compass, K.compass]))
        assert s.dkind == D_SCIENCE and list(s.legal_actions()) == [A.SCI_PAIR_BASE + 1, A.SCI_PAIR_BASE + 2]
        s = settle(s.apply_action(A.SCI_PAIR_BASE + 2).apply_action(A.TOKEN_BLIND))
        assert s.dkind == D_TOKEN and s.cards[0][K.compass] == 0 and s.cards[0][K.gear] == 0  # second set

    def test_face_up_token_is_refilled_from_the_stack(self, base):
        s = check(with_science(base, [K.tablet, K.tablet])).apply_action(A.TOKEN_BASE + T.Strategy)
        assert s.tokens[0][T.Strategy] == 1 and T.Strategy not in s.faceup and len(s.faceup) == 2
        assert s.is_chance() and s.ckind == C_TOKEN_REVEAL
        outs = dict(s.chance_outcomes())
        assert len(outs) == 11 and abs(outs[T.Culture] - 2 / 12) < 1e-12 and T.Strategy not in outs
        assert abs(sum(outs.values()) - 1) < 1e-12
        s = s.apply_chance(T.Urbanism)
        assert sorted(s.faceup) == sorted([T.Education, T.Decor, T.Urbanism]) and s.prog_stack == 11
        assert is_main_pick_of(s, 1)
        assert_invariants(s)

    def test_blind_token_is_a_chance_draw(self, base):
        s = check(with_science(base, [K.tablet, K.tablet])).apply_action(A.TOKEN_BLIND)
        assert s.is_chance() and s.ckind == C_TOKEN_BLIND
        outs = dict(s.chance_outcomes())
        assert abs(outs[T.Culture] - 2 / 12) < 1e-12 and all(abs(p - 1 / 12) < 1e-12 for t, p in outs.items() if t != T.Culture)
        s = s.apply_chance(T.Tactics)
        assert s.tokens[0][T.Tactics] == 1 and sorted(s.faceup) == sorted(DEFAULT_FACEUP) and s.prog_stack == 11
        assert is_main_pick_of(s, 1)
        assert_invariants(s)

    def test_no_forced_discard_when_no_tokens_remain(self, base):
        c = with_science(base, [K.tablet, K.tablet])
        exhaust_tokens(c)
        s = check(c)
        assert is_main_pick_of(s, 1) and s.cards[0][K.tablet] == 2 and s.discard[K.tablet] == 0
        assert_invariants(s)

    def test_last_token_without_a_stack_is_taken_automatically(self, base):
        c = with_science(base, [K.tablet, K.tablet])
        exhaust_tokens(c)
        c.tokens[1][T.Strategy] -= 1
        c.faceup = [T.Strategy]
        s = check(c)
        assert s.tokens[0][T.Strategy] == 1 and s.faceup == [] and s.cards[0][K.tablet] == 0
        assert is_main_pick_of(s, 1)

    def test_blind_only_when_the_face_up_row_is_empty(self, base):
        c = with_science(base, [K.tablet, K.tablet])
        set_faceup(c, [])
        s = check(c)
        assert s.is_chance() and s.ckind == C_TOKEN_BLIND  # the single option was auto-resolved

    def test_two_sets_in_one_turn(self, base):
        s = check(with_science(base, [K.tablet] * 4))
        assert s.dkind == D_TOKEN
        s = settle(s.apply_action(A.TOKEN_BASE + T.Strategy), lambda st: T.Tactics)
        assert s.dkind == D_TOKEN and s.cards[0][K.tablet] == 0
        s = settle(s.apply_action(A.TOKEN_BASE + T.Tactics), lambda st: T.Culture)
        assert s.tokens[0][T.Strategy] == 1 and s.tokens[0][T.Tactics] == 1 and is_main_pick_of(s, 1)
        assert_invariants(s)

    def test_science_set_completed_by_the_card_just_taken(self, base):
        c = with_science(base, [K.gear, K.compass])
        s = take_card(c, K.tablet)
        assert s.dkind == D_TOKEN and all(s.cards[0][k] == 0 for k in GREEN_KINDS)

    def test_token_taken_by_science_can_change_the_build_check(self, base):
        """Engineering acquired in the same turn immediately unlocks a build (check re-run)."""
        c = with_science(base, [K.tablet, K.tablet, K.wood, K.stone])
        c.built[0] = first_n(1)  # 2 identical: not affordable with wood + stone ...
        set_faceup(c, [T.Engineering, T.Strategy, T.Decor])
        s = check(c)
        assert s.dkind == D_TOKEN and s.num_stages(0) == 1
        s, _ = pay_through(s.apply_action(A.TOKEN_BASE + T.Engineering).apply_chance(T.Tactics))
        assert s.num_stages(0) == 2  # ... until Engineering arrives

    def test_player_may_take_a_token_before_building(self, base):
        c = with_science(base, [K.tablet, K.tablet, K.wood, K.wood])
        c.built[0] = first_n(1)
        set_faceup(c, [T.Architecture, T.Strategy, T.Decor])
        s = check(c)
        # rulebook: science first -> Architecture -> build -> extra pick offered this turn
        assert s.dkind == D_TOKEN and s.num_stages(0) == 1
        s = s.apply_action(A.TOKEN_BASE + T.Architecture).apply_chance(T.Tactics)
        assert s.num_stages(0) == 2 and is_pick(s, f"token:{T.Architecture}", 0)


# =====================================================================================
# Progress tokens: extra picks and passive / end-game effects
# =====================================================================================
class TestProgressTokens:
    def offered(self, base: GameState, token: int, kind: int, rules: RulesConfig = None) -> GameState:
        b = base if rules is None else first_pick(rules=rules)
        c = edit(b)
        give_token(c, 0, token)
        return take_card(c, kind)

    @pytest.mark.parametrize("token,kind,expect", [
        (T.Urbanism, K.wood, True), (T.Urbanism, K.clay, True), (T.Urbanism, K.stone, False), (T.Urbanism, K.coin, False),
        (T.Crafts, K.papyrus, True), (T.Crafts, K.glass, True), (T.Crafts, K.wood, False),
        (T.Jewellery, K.stone, True), (T.Jewellery, K.coin, True), (T.Jewellery, K.glass, False),
        (T.Science, K.tablet, True), (T.Science, K.gear, True), (T.Science, K.compass, True), (T.Science, K.civ3, False),
        (T.Propaganda, K.shield_h1, True), (T.Propaganda, K.shield_h2, True), (T.Propaganda, K.shield, False),
        (T.Tactics, K.wood, False), (T.Engineering, K.stone, False),
    ])
    def test_extra_pick_triggers(self, base, token, kind, expect):
        s = self.offered(base, token, kind)
        if expect:
            assert is_pick(s, f"token:{token}", 0) and s.dctx == (f"token:{token}", True, (0, 1, 2))
            assert list(s.legal_actions()) == [A.PICK_LEFT, A.PICK_RIGHT, A.PICK_CENTER, A.SKIP]
        else:
            assert not is_pick(s, f"token:{token}")

    def test_extra_pick_is_offered_only_to_the_token_owner(self, base):
        c = edit(base)
        give_token(c, 1, T.Urbanism)
        assert not is_pick(take_card(c, K.wood), f"token:{T.Urbanism}")

    def test_extra_pick_skip(self, base):
        s = self.offered(base, T.Urbanism, K.wood).apply_action(A.SKIP)
        assert is_main_pick_of(s, 1) and sum(s.cards[0]) == 1

    def test_extra_pick_takes_a_card_and_marks_the_token_used(self, base):
        c = edit(base)
        give_token(c, 0, T.Urbanism)
        give_token(c, 0, T.Architecture)  # its extra pick keeps the turn open after the build (used-bits observable)
        s = take_card(c, K.wood)
        assert is_pick(s, f"token:{T.Urbanism}", 0)
        s = s.apply_action(A.PICK_RIGHT)  # deck 1 top is a stone
        assert s.is_chance() and s.ckind == C_REVEAL and s.cctx == 1
        s = s.apply_chance(K.clay)
        # wood + stone is exactly the "2 different" first stage: the mandatory build is auto-resolved
        assert s.num_stages(0) == 1 and s.cards[0][K.wood] == 0 and s.cards[0][K.stone] == 0
        assert s.discard[K.wood] == 1 and s.discard[K.stone] == 1
        assert is_pick(s, f"token:{T.Architecture}", 0)  # offered by the build, still this turn
        assert (s.tokens_used >> T.Urbanism) & 1 and not (s.tokens_used >> T.Architecture) & 1
        assert_invariants(s)

    def test_extra_pick_from_own_deck_with_a_second_trigger_is_not_repeated(self, base):
        c = edit(base)
        c.built[0] = first_n(1)  # stage 2 needs 2 identical: wood + clay will not build
        give_token(c, 0, T.Urbanism)
        set_top(c, 0, K.clay)  # an Urbanism card on top of the own deck
        s = take_card(c, K.wood)
        assert is_pick(s, f"token:{T.Urbanism}", 0)
        s = s.apply_action(A.PICK_LEFT).apply_chance(K.civ3)  # takes the clay, reveals civ3
        assert s.cards[0][K.clay] == 1 and s.cards[0][K.wood] == 1
        assert is_main_pick_of(s, 1)  # clay would trigger Urbanism again, but it was used this turn

    def test_extra_pick_mandatory_when_rule_disabled(self, base):
        s = self.offered(base, T.Urbanism, K.wood, rules=RulesConfig(extra_card_optional=False))
        assert is_pick(s, f"token:{T.Urbanism}", 0) and s.dctx[1] is False
        assert list(s.legal_actions()) == [A.PICK_LEFT, A.PICK_RIGHT, A.PICK_CENTER]

    def test_extra_pick_from_the_center_is_a_chance_draw(self, base):
        s = self.offered(base, T.Jewellery, K.coin).apply_action(A.PICK_CENTER)
        assert s.is_chance() and s.ckind == C_DRAW_CENTRAL and s.cctx == f"token:{T.Jewellery}"
        s = s.apply_chance(K.civ3)
        assert s.cards[0][K.civ3] == 1 and s.cards[0][K.coin] == 1 and is_main_pick_of(s, 1)

    def test_different_tokens_chain_in_one_turn(self, base):
        c = edit(base)
        give_token(c, 0, T.Urbanism)
        give_token(c, 0, T.Jewellery)
        s = take_card(c, K.wood)  # Urbanism
        assert is_pick(s, f"token:{T.Urbanism}", 0)
        s = s.apply_action(A.PICK_RIGHT).apply_chance(K.glass)  # the stone on deck 1 -> Jewellery
        assert is_pick(s, f"token:{T.Jewellery}", 0)
        s = s.apply_action(A.PICK_RIGHT).apply_chance(K.coin)  # glass: no token for it
        assert s.cards[0][K.glass] == 1 and s.to_move() == 0 and s.dkind == D_PAY  # wood/stone/glass build

    def test_tokens_are_usable_again_next_turn(self, base):
        c = edit(base)
        give_token(c, 0, T.Urbanism)
        set_top(c, 0, K.wood)
        s = take_card(c, K.wood).apply_action(A.SKIP)  # used? no: declined
        s = settle(s.apply_action(A.PICK_LEFT))  # P1's turn
        assert is_main_pick_of(s, 0) and s.tokens_used == 0
        s = s.apply_action(A.PICK_LEFT).apply_chance(K.civ3)
        assert is_pick(s, f"token:{T.Urbanism}", 0)

    def test_architecture_offers_an_extra_pick_after_the_stage_effect(self):
        c = wonder_at_stage2(W.Babylon)
        give_token(c, 0, T.Architecture)
        s = check(c)
        assert s.dkind == D_TOKEN and s.dctx == "babylon"  # stage effect first
        s = s.apply_action(A.TOKEN_BLIND).apply_chance(T.Tactics)
        assert is_pick(s, f"token:{T.Architecture}", 0)
        assert A.SKIP in s.legal_actions()

    def test_architecture_once_per_turn_but_declining_does_not_consume_it(self, base):
        c = giza_with(base, 0, [K.wood, K.stone, K.clay, K.clay], [T.Architecture])
        s = check(c).apply_action(A.PAY_BASE + 0).apply_action(A.PAY_BASE + 1)
        assert s.num_stages(0) == 1 and is_pick(s, f"token:{T.Architecture}", 0)
        declined = s.apply_action(A.SKIP)  # stage 2 (clay x2) follows: offered again
        assert declined.num_stages(0) == 2 and is_pick(declined, f"token:{T.Architecture}", 0)
        used = s.apply_action(A.PICK_CENTER).apply_chance(K.civ3)
        assert used.num_stages(0) == 2 and is_main_pick_of(used, 1)  # not offered a second time

    def test_tactics_adds_two_shields(self, base):
        c = edit(base)
        give_token(c, 0, T.Tactics)
        assert c.shields_of(0) == 2 and c.shields_of(1) == 0
        give(c, 0, K.shield)
        assert c.shields_of(0) == 3

    def test_engineering_and_economy_do_not_score(self, base):
        c = edit(base)
        give_token(c, 0, T.Engineering)
        give_token(c, 0, T.Economy)
        give_token(c, 0, T.Urbanism)
        assert c.score_of(0) == 0

    def test_end_game_vp_tokens(self, base):
        c = edit(base)
        give_token(c, 0, T.Strategy)
        assert c.score_of(0) == 0
        c.mil_tokens[0] = 2
        assert c.score_of(0) == 2 * base.rules.military_token_vp + 2
        c.mil_tokens[0] = 0
        give_token(c, 0, T.Education)  # 2 tokens now
        assert c.score_of(0) == 2 * 2
        give_token(c, 0, T.Decor)  # 3 tokens
        assert c.score_of(0) == 2 * 3 + 4
        c.built[0] = first_n(5)
        assert c.score_of(0) == 30 + 2 * 3 + 6
        c.built[0] = first_n(0)
        give_token(c, 0, T.Politics)  # 4 tokens
        assert c.score_of(0) == 2 * 4 + 4
        give_many(c, 0, [K.civ2cat, K.civ2cat, K.civ3])
        c.cat = 1
        assert c.score_of(0) == 2 * 4 + 4 + 2 + 2 + 2 + 3
        c.cards[0] = [0] * NUM_KINDS
        c.tokens[0] = [0] * len(TOKENS)
        give_token(c, 0, T.Culture)
        assert c.score_of(0) == 4
        give_token(c, 0, T.Culture)
        assert c.score_of(0) == 12


# =====================================================================================
# Military
# =====================================================================================
def battle(base: GameState, shields0: int, shields1: int, rules: RulesConfig = None) -> GameState:
    """Resolve a pending battle with the given (permanent) shield counts; returns the state after."""
    c = edit(base if rules is None else first_pick(rules=rules))
    c.wonder_shields = [shields0, shields1]
    c.conflict, c.battle_pending = c.rules.conflict_tokens, True
    return run(c, ("end_turn",))


class TestMilitary:
    def test_horns_flip_conflict_tokens(self, base):
        s = take_card(base, K.shield_h1)
        assert s.conflict == 1 and not s.battle_pending and s.mil_tokens == [0, 0]
        s = take_card(base, K.shield_h2)
        assert s.conflict == 2 and not s.battle_pending
        assert take_card(base, K.shield).conflict == 0

    def test_third_horn_triggers_a_battle_resolved_at_end_of_turn(self, base):
        c = edit(base)
        c.conflict = 2
        give_many(c, 0, [K.wood, K.stone, K.clay])  # keeps the turn open with a pay decision
        s = take_card(c, K.shield_h1)
        assert s.to_move() == 0 and s.dkind == D_PAY
        assert s.battle_pending and s.conflict == 3 and s.mil_tokens == [0, 0]
        assert s.cards[0][K.shield_h1] == 1  # mid-turn: nothing resolved yet
        s, _ = pay_through(s)
        assert is_main_pick_of(s, 1)
        assert not s.battle_pending and s.conflict == 0
        assert s.mil_tokens == [1, 0]  # 1 shield vs 0
        assert s.cards[0][K.shield_h1] == 0 and s.discard[K.shield_h1] == 1
        assert s.score_of(0) == base.rules.military_token_vp + WONDERS[W.Giza].stages[0].vp
        assert_invariants(s)

    def test_overflow_is_capped_at_the_number_of_conflict_tokens(self, base):
        c = edit(base)
        c.conflict = 2
        give_many(c, 0, [K.tablet, K.tablet])  # keep the turn open (token choice)
        s = take_card(c, K.shield_h2)
        assert s.dkind == D_TOKEN and s.conflict == 3 and s.battle_pending

    def test_extra_horns_after_the_battle_is_triggered_are_ignored(self, base):
        c = edit(base)
        c.conflict, c.battle_pending = 3, True
        give_token(c, 0, T.Propaganda)
        s = take_card(c, K.shield_h2)
        assert is_pick(s, f"token:{T.Propaganda}", 0) and s.conflict == 3

    @pytest.mark.parametrize("s0,s1,expect", [
        (0, 0, (0, 0)), (1, 1, (0, 0)), (2, 2, (0, 0)),
        (1, 0, (1, 0)), (2, 0, (2, 0)), (5, 0, (2, 0)),
        (2, 1, (2, 0)), (3, 2, (1, 0)), (4, 2, (2, 0)), (5, 3, (1, 0)),
        (0, 1, (0, 1)), (1, 3, (0, 2)), (2, 3, (0, 1)),
    ])
    def test_two_player_battle_rule(self, base, s0, s1, expect):
        s = battle(base, s0, s1)
        assert tuple(s.mil_tokens) == expect
        assert s.conflict == 0 and not s.battle_pending
        assert s.score_of(0) == expect[0] * base.rules.military_token_vp + s.token_vp(0)

    def test_double_vs_zero_rule_option(self, base):
        assert battle(base, 1, 0).mil_tokens == [1, 0]
        assert battle(base, 1, 0, rules=RulesConfig(double_vs_zero_requires_two=False)).mil_tokens == [2, 0]
        assert battle(base, 0, 1, rules=RulesConfig(double_vs_zero_requires_two=False)).mil_tokens == [0, 2]

    def test_horned_cards_are_discarded_after_the_battle_hornless_kept(self, base):
        c = edit(base)
        give_many(c, 0, [K.shield, K.shield_h1, K.shield_h2])
        give_many(c, 1, [K.shield_h1])
        give_token(c, 1, T.Tactics)
        c.conflict, c.battle_pending = 3, True
        assert c.shields_of(0) == 3 and c.shields_of(1) == 3
        s = run(c, ("end_turn",))
        assert s.mil_tokens == [0, 0]  # tie
        assert s.cards[0][K.shield] == 1 and s.cards[0][K.shield_h1] == 0 and s.cards[0][K.shield_h2] == 0
        assert s.cards[1][K.shield_h1] == 0
        assert s.discard[K.shield_h1] == 2 and s.discard[K.shield_h2] == 1 and s.discard[K.shield] == 0
        assert s.shields_of(0) == 1 and s.shields_of(1) == 2  # Tactics persists
        assert_invariants(s)

    def test_shields_from_all_sources_count(self, base):
        c = edit(base)
        give_many(c, 0, [K.shield, K.shield_h2])
        give_token(c, 0, T.Tactics)
        c.wonder_shields[0] = 1
        assert c.shields_of(0) == 5
        give_many(c, 1, [K.shield_h1, K.shield_h1])
        c.conflict, c.battle_pending = 3, True
        s = run(c, ("end_turn",))
        assert s.mil_tokens == [2, 0]  # 5 >= 2 x 2

    def test_military_token_value(self, base):
        c = edit(base)
        c.mil_tokens = [2, 1]
        assert c.scores() == (6, 3)
        c5 = edit(first_pick(rules=RulesConfig(military_token_vp=5)))
        c5.mil_tokens = [1, 0]
        assert c5.scores() == (5, 0)

    def test_battle_resolves_after_the_rest_of_the_turn(self):
        """A stage built later in the same turn (Rhodes shield) counts in the battle."""
        c = edit(first_pick(W.Rhodes, W.Giza))
        give_many(c, 0, [K.wood, K.stone])  # Rhodes S1 (2 different) gives the shield
        c.conflict = 2
        s = take_card(c, K.shield_h1)  # triggers the battle, then the mandatory build gives +1 shield
        assert s.num_stages(0) == 1 and s.mil_tokens == [2, 0]  # 2 shields vs 0 -> two tokens
        assert s.cards[0][K.shield_h1] == 0 and s.wonder_shields[0] == 1

    def test_battle_triggered_by_the_opponent_still_compares_both(self, base):
        c = edit(base)
        give_many(c, 0, [K.shield, K.shield])
        c.conflict = 2
        c.mover = 1
        s = take_card(c, K.shield_h1, p=1)  # P1 triggers the battle with 1 shield vs P0's 2
        assert s.mil_tokens == [2, 0] and s.cards[1][K.shield_h1] == 0 and s.cards[0][K.shield] == 2

    def test_conflict_tokens_rule_option(self):
        b = first_pick(rules=RulesConfig(conflict_tokens=2))
        s = take_card(b, K.shield_h2)
        assert s.mil_tokens == [1, 0] and s.conflict == 0  # 2 horns fill both tokens -> battle


# =====================================================================================
# End of game and scoring
# =====================================================================================
class TestGameEnd:
    def final_stage(self, base: GameState, extra=lambda c: None) -> GameState:
        """Giza's 5th stage (4 different) paid out of 5 different greys: the payment is a real decision,
        so the game must not end before it is resolved."""
        c = giza_with(base, 4, [K.wood, K.stone, K.clay, K.papyrus, K.glass])
        extra(c)
        s = check(c)
        assert s.to_move() == 0 and s.dkind == D_PAY and not s.is_terminal()
        s, taken = pay_through(s)
        assert len(taken) == 4 and s.num_stages(0) == 5  # wood, stone, clay, papyrus (first option each time)
        return s

    def test_fifth_stage_ends_the_game_at_the_end_of_the_turn(self, base):
        s = self.final_stage(base)
        assert s.is_terminal() and s.game_over and s.wonder_done and s.to_move() == -2
        assert s.num_stages(0) == 5 and s.scores() == (30, 0) and s.returns() == (1.0, -1.0)
        assert s.score_diff() == 30 and s.legal_actions() == () and s.chance_outcomes() == ()
        assert s.mover == 0  # the opponent gets no further turn
        with pytest.raises(ValueError):
            s.apply_action(A.PICK_LEFT)
        assert_invariants(s)

    def test_battle_in_the_final_turn_still_resolves(self, base):
        def extra(c):
            give(c, 0, K.shield)
            c.conflict, c.battle_pending = 3, True
        s = self.final_stage(base, extra)
        assert s.is_terminal() and s.mil_tokens == [1, 0] and s.conflict == 0
        assert s.scores() == (33, 0)

    def test_rest_of_the_final_turn_is_completed(self, base):
        """The 5th stage does not end the game on the spot: the Architecture pick it grants, the science
        set completed by the picked card and the token choice are all resolved first."""
        def extra(c):
            give_token(c, 0, T.Architecture)
            give(c, 0, K.tablet)
            set_top(c, 0, K.tablet)
        s = self.final_stage(base, extra)
        assert not s.is_terminal() and s.num_stages(0) == 5 and is_pick(s, f"token:{T.Architecture}", 0)
        s = settle(s.apply_action(A.PICK_LEFT))  # the tablet completes a pair
        assert not s.is_terminal() and s.dkind == D_TOKEN and s.dctx == "science"
        s = settle(s.apply_action(A.TOKEN_BASE + T.Decor))
        assert s.is_terminal() and s.scores() == (36, 0)  # 30 + Decor (6: Wonder complete)

    def test_science_is_resolved_before_the_final_stage(self, base):
        """Mandatory science sets come before the construction of the same check (the token gained
        may apply to that build), also in the turn that completes the Wonder."""
        c = giza_with(base, 4, [K.wood, K.stone, K.clay, K.papyrus, K.tablet, K.tablet])
        s = check(c)
        assert s.to_move() == 0 and s.dkind == D_TOKEN and s.dctx == "science" and s.num_stages(0) == 4
        s = settle(s.apply_action(A.TOKEN_BASE + T.Decor))  # then the (exact) payment is auto-resolved
        assert s.is_terminal() and s.num_stages(0) == 5 and s.scores() == (36, 0)

    def test_loser_completing_the_wonder_still_ends_the_game(self, base):
        s = self.final_stage(base, lambda c: c.mil_tokens.__setitem__(1, 11))
        assert s.is_terminal() and s.returns() == (-1.0, 1.0) and s.scores() == (30, 33)

    def test_game_ends_when_all_decks_are_empty(self, base):
        c = edit(base)
        for d in range(3):
            drain_deck(c, d)
        c.cards[0][K.civ3], c.discard[K.civ3] = 1, c.discard[K.civ3] - 1  # one blue card for P0
        s = run(c, ("turn_start",))
        assert s.is_terminal() and not s.wonder_done and s.scores() == (3, 0) and s.returns() == (1.0, -1.0)
        assert_invariants(s)

    def test_all_decks_empty_ends_the_game_even_with_the_rule_disabled(self):
        """No card can ever be drawn again: the game ends regardless of ``end_when_no_cards``."""
        c = edit(first_pick(rules=RulesConfig(end_when_no_cards=False)))
        for d in range(3):
            drain_deck(c, d)
        s = run(c, ("turn_start",))
        assert s.is_terminal() and s.game_over and not s.wonder_done
        assert_invariants(s)

    def test_decks_emptying_mid_turn_ends_the_game_at_the_next_turn_start(self, base):
        c = edit(base)
        drain_deck(c, 1)
        drain_deck(c, CENTRAL)
        set_deck(c, 0, K.civ3, [])
        s = run(c, ("turn_start",))  # the single option is auto-taken and the game is over
        assert s.is_terminal() and s.cards[0][K.civ3] == 1 and s.turn == 1

    def test_tie_break_by_stages_then_draw(self, base):
        def ended(rules=None):
            c = edit(base if rules is None else first_pick(rules=rules))
            for d in range(3):
                drain_deck(c, d)
            c.built[0] = first_n(1)  # Giza stage 1: 4 VP
            c.cards[1][K.civ2cat], c.discard[K.civ2cat] = 1, c.discard[K.civ2cat] - 1
            c.cat = 1  # 2 + 2 = 4 VP
            return c
        s = run(ended(), ("turn_start",))
        assert s.is_terminal() and s.scores() == (4, 4)
        assert s.returns() == (1.0, -1.0)  # more stages
        s = run(ended(RulesConfig(tiebreak_stages=False)), ("turn_start",))
        assert s.scores() == (4, 4) and s.returns() == (0.0, 0.0)
        c = ended()
        c.built = [0, 0]
        c.cards[0][K.civ3], c.discard[K.civ3] = 1, c.discard[K.civ3] - 1
        c.cards[1][K.civ2cat] = 0
        c.discard[K.civ2cat] += 1
        c.cards[1][K.civ3], c.discard[K.civ3] = 1, c.discard[K.civ3] - 1
        c.cat = -1
        s = run(c, ("turn_start",))
        assert s.scores() == (3, 3) and s.built == [0, 0] and s.returns() == (0.0, 0.0)

    def test_returns_are_zero_before_the_end(self, base):
        c = giza_with(base, 4, [K.civ3])
        assert c.returns() == (0.0, 0.0) and c.score_diff() == 25.0

    def test_no_cards_rule_disabled_does_not_hang(self):
        code = r"""
import sys
sys.path.insert(0, %r)
from conftest import *
c = edit(first_pick(rules=RulesConfig(end_when_no_cards=False)))
for d in range(3):
    drain_deck(c, d)
s = run(c, ("turn_start",))
print("node", s.describe_node())
""" % (str(__import__("os").path.dirname(__file__)),)
        try:
            proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=6)
        except subprocess.TimeoutExpired:
            pytest.fail("engine hung with all decks empty and end_when_no_cards=False")
        assert proc.returncode == 0, proc.stderr
