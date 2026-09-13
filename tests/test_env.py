"""Tests for :class:`sevenwa.engine.env.Environment` (the real game with hidden information).

* random games complete, every true chance outcome is possible under the belief (the environment
  raises otherwise), the physical decks / token stack always match the belief state;
* games are deterministic given the seed;
* ``observe(p)`` never leaks the central top card to a player who has not seen it;
* the physical model of the Halicarnassus effect (window removal + reshuffle).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

from sevenwa.engine import Actions as A
from sevenwa.engine import RulesConfig
from sevenwa.engine.cards import NUM_KINDS
from sevenwa.engine.env import Environment, make_env
from sevenwa.engine.state import CENTRAL, D_HALI_CHOOSE, D_HALI_DECK

from conftest import (K, W, assert_invariants, check, edit, give_many, initial_composition, is_main_pick_of,
                      physical_matches_belief, random_env_game, set_deck, sync_env_to_state)


class TestRandomGames:
    N_GAMES = 300

    def test_random_games_complete_and_stay_consistent(self):
        stats: Dict[str, int] = {"known_central": 0, "hidden_obs": 0, "both_know": 0, "hali_choices": 0, "terminal": 0,
                                 "wonder_end": 0, "no_cards_end": 0}
        comps: Dict = {}

        def on_step(env: Environment, action: int) -> None:
            s = env.state
            comp = comps.setdefault(s.wonder, initial_composition(s))
            assert s.to_move() in (0, 1) and env.to_move() == s.to_move()
            assert_invariants(s, comp)
            physical_matches_belief(env)
            assert s.observer == -1  # the environment keeps the omniscient view
            m = s.mover
            obs = env.observe(m)
            assert obs.observer == m and obs.to_move() == m
            assert list(obs.legal_actions()) == list(s.legal_actions())
            for p in (0, 1):
                o = env.observe(p)
                assert_invariants(o, comp)
                if s.knows_central(p):
                    assert o.deck_top[CENTRAL] == s.deck_top[CENTRAL] and o.knows_central(p)
                else:
                    assert o.deck_top[CENTRAL] == -1 and o.central_known_to == 0
                    assert sum(o.unseen[CENTRAL]) == o.deck_size[CENTRAL]
                    if s.deck_top[CENTRAL] >= 0:
                        stats["hidden_obs"] += 1
                        assert o.unseen[CENTRAL][s.deck_top[CENTRAL]] == s.unseen[CENTRAL][s.deck_top[CENTRAL]] + 1
                # stepping an observation must never touch the environment
                key_before = s.key()
                o.apply_action(list(o.legal_actions())[0])
                assert s.key() == key_before and env.state is s
            if s.deck_top[CENTRAL] >= 0:
                stats["known_central"] += 1
                assert s.central_known_to != 0
                if s.central_known_to == 3:
                    stats["both_know"] += 1
            if s.dkind == D_HALI_CHOOSE:
                stats["hali_choices"] += 1
            assert action in s.legal_actions()

        for seed in range(self.N_GAMES):
            env = random_env_game(seed, on_step=on_step)
            assert env.is_terminal() and env.to_move() == -2
            assert env.scores() == env.state.scores() and env.returns() == env.state.returns()
            physical_matches_belief(env)
            stats["terminal"] += 1
            if env.state.wonder_done:
                stats["wonder_end"] += 1
            elif all(n == 0 for n in env.state.deck_size):
                stats["no_cards_end"] += 1
            with pytest.raises(ValueError):
                env.step(A.PICK_LEFT)
        assert stats["terminal"] == self.N_GAMES
        assert stats["known_central"] > 0 and stats["hidden_obs"] > 0 and stats["hali_choices"] > 0
        assert stats["wonder_end"] + stats["no_cards_end"] == self.N_GAMES and stats["wonder_end"] > 0

    def test_games_are_deterministic_given_the_seed(self):
        for seed in (1, 7, 42):
            trace_a: List[bytes] = []
            trace_b: List[bytes] = []
            env_a = random_env_game(seed, on_step=lambda e, a: trace_a.append(e.state.key() + bytes([a])))
            env_b = random_env_game(seed, on_step=lambda e, a: trace_b.append(e.state.key() + bytes([a])))
            assert trace_a == trace_b and len(trace_a) > 10
            assert env_a.state.key() == env_b.state.key() and env_a.scores() == env_b.scores()
            assert env_a.decks == env_b.decks and env_a.wonders == env_b.wonders
        assert random_env_game(1).state.key() != random_env_game(2).state.key()

    def test_explicit_wonders_and_rules(self):
        env = make_env(3, wonders=(W.Rhodes, W.Babylon), rules=RulesConfig(cat_vp=0))
        assert env.wonders == (W.Rhodes, W.Babylon) and env.state.wonder == (W.Rhodes, W.Babylon)
        assert env.rules.cat_vp == 0 and env.state.rules.cat_vp == 0
        assert env.to_move() == 0 and is_main_pick_of(env.state, 0)
        assert env.state.deck_size == [25, 25, 60] and len(env.state.faceup) == 3
        assert [len(d) for d in env.decks] == [25, 25, 60] and len(env.token_stack) == 12
        assert env.decks[0][0] == env.state.deck_top[0] and env.decks[1][0] == env.state.deck_top[1]
        physical_matches_belief(env)
        env2 = Environment(seed=5)
        assert env2.wonders[0] != env2.wonders[1]

    def test_step_rejects_illegal_actions_and_non_decision_nodes(self):
        env = make_env(0, wonders=(W.Giza, W.Rhodes))
        with pytest.raises(ValueError):
            env.step(A.SKIP)
        before = env.state.key()
        env.step(A.PICK_CENTER)  # the draw is resolved by the environment
        assert env.to_move() == 1 and env.state.key() != before
        assert sum(env.state.cards[0]) == 1 and env.state.deck_size[CENTRAL] == 59


class TestHiddenInformation:
    def holder_env(self, seed: int = 0) -> Environment:
        """Player 0 holds the Cat; the environment is at P0's main pick after the peek."""
        env = make_env(seed, wonders=(W.Giza, W.Rhodes))
        c = edit(env.state)
        c.cat = 0
        sync_env_to_state(env, c, seed)
        env.state = c
        env.step(A.PICK_LEFT)  # P0's turn passes ...
        env.step(A.PICK_LEFT)  # ... P1's turn passes; P0's next turn starts with the peek
        s = env.state
        assert s.to_move() == 0 and s.knows_central(0) and not s.knows_central(1)
        assert s.deck_top[CENTRAL] == env.decks[CENTRAL][0]
        return env

    def test_observe_shows_the_peek_only_to_the_holder(self):
        env = self.holder_env()
        s = env.state
        top = s.deck_top[CENTRAL]
        holder, other = env.observe(0), env.observe(1)
        assert holder.deck_top[CENTRAL] == top and holder.knows_central(0) and holder.observer == 0
        assert other.deck_top[CENTRAL] == -1 and other.central_known_to == 0 and other.observer == 1
        assert other.unseen[CENTRAL][top] == s.unseen[CENTRAL][top] + 1
        assert other.deck_size[CENTRAL] == s.deck_size[CENTRAL]
        assert other.key() != holder.key()
        assert list(other.legal_actions()) == list(holder.legal_actions()) == list(s.legal_actions())
        assert_invariants(other)
        # in the non-holder's belief a central pick is a chance node; in the holder's it is not
        assert other.apply_action(A.PICK_CENTER).is_chance()
        assert not holder.apply_action(A.PICK_CENTER).is_chance()

    def test_holder_draws_the_peeked_card(self):
        env = self.holder_env()
        top = env.decks[CENTRAL][0]
        env.step(A.PICK_CENTER)
        assert env.state.cards[0][top] == 1 and env.state.deck_top[CENTRAL] == -1
        assert len(env.decks[CENTRAL]) == env.state.deck_size[CENTRAL]
        physical_matches_belief(env)

    def test_non_holder_draw_is_resolved_with_the_true_card(self):
        env = self.holder_env()
        top = env.decks[CENTRAL][0]
        env.step(A.PICK_LEFT)  # holder passes
        assert env.to_move() == 1 and env.state.knows_central(0)
        assert env.observe(1).deck_top[CENTRAL] == -1
        n_before = env.state.cards[1][top]
        env.step(A.PICK_CENTER)  # forgetting draw in the belief, true card in the environment
        assert env.state.cards[1][top] == n_before + 1
        physical_matches_belief(env)

    def test_stolen_cat_reveals_the_same_physical_card(self):
        env = self.holder_env()
        top = env.decks[CENTRAL][0]
        env.step(A.PICK_LEFT)
        c = edit(env.state)
        c.deck_top[1] = K.civ2cat  # put a Cat card on top of P1's deck (physically too)
        c.unseen[1][env.state.deck_top[1]] += 1
        c.unseen[1][K.civ2cat] -= 1
        assert c.unseen[1][K.civ2cat] >= 0
        sync_env_to_state(env, c, 1)
        env.step(A.PICK_LEFT)  # P1 takes the Cat
        assert env.state.cat == 1 and env.to_move() == 0
        env.step(A.PICK_LEFT)  # P0 passes; P1 now peeks at the card P0 already saw
        s = env.state
        assert s.to_move() == 1 and s.central_known_to == 3 and s.deck_top[CENTRAL] == top
        assert env.observe(0).deck_top[CENTRAL] == top and env.observe(1).deck_top[CENTRAL] == top
        physical_matches_belief(env)


class TestPhysicalModel:
    def test_impossible_true_outcome_is_rejected(self):
        env = make_env(0, wonders=(W.Giza, W.Rhodes))
        s = env.state
        # corrupt the physical central deck so that its top is a kind the belief thinks is exhausted
        kind = next(k for k in range(NUM_KINDS) if s.unseen[CENTRAL][k] > 0)
        c = edit(s)
        c.deck_size[CENTRAL] -= c.unseen[CENTRAL][kind]
        c.discard[kind] += c.unseen[CENTRAL][kind]
        c.unseen[CENTRAL][kind] = 0
        env.state = c
        # physical deck: the "exhausted" kind on top, same length as the belief's deck size
        env.decks[CENTRAL] = ([kind] + [k for k in env.decks[CENTRAL] if k != kind])[:c.deck_size[CENTRAL]]
        with pytest.raises(RuntimeError, match="impossible under belief"):
            env.step(A.PICK_CENTER)

    def hali_env(self, deck0: List[int], seed: int = 0) -> Environment:
        """Halicarnassus (P0) about to build stage 2 with the given physical deck 0 (top first)."""
        env = make_env(seed, wonders=(W.Halicarnassus, W.Giza))
        c = edit(env.state)
        c.stages[0] = 1
        give_many(c, 0, [K.stone, K.stone])
        set_deck(c, 0, deck0[0], deck0[1:])
        sync_env_to_state(env, c, seed)
        rest = list(deck0[1:])
        np.random.default_rng(seed).shuffle(rest)
        env.decks[0] = [deck0[0]] + rest
        physical_matches_belief(env)
        return env

    def test_halicarnassus_window_and_reshuffle(self):
        env = self.hali_env([K.clay, K.papyrus, K.glass, K.coin, K.civ3, K.tablet, K.gear, K.wood])
        window = list(env.decks[0][:5])
        rest = list(env.decks[0][5:])
        env.state = check(env.state)  # build stage 2 -> deck choice
        env._resolve_chance()
        assert env.state.dkind == D_HALI_DECK
        env.step(A.PICK_LEFT)  # the 4 reveals come from the true deck order
        s = env.state
        assert s.dkind == D_HALI_CHOOSE and list(s.dctx[1]) == window
        physical_matches_belief(env)
        keep = window[2]
        env.step(A.HALI_BASE + keep)
        s = env.state
        assert s.cards[0][keep] == 1
        expected = sorted(window[:2] + window[3:] + rest)
        assert sorted(env.decks[0]) == expected and len(env.decks[0]) == s.deck_size[0] == 7
        assert env.decks[0][0] == s.deck_top[0]
        physical_matches_belief(env)

    def test_auto_resolved_halicarnassus_choice_still_reshuffles(self):
        all_wood_on_top = 0
        n = 20
        for seed in range(n):
            env = self.hali_env([K.wood] * 5 + [K.stone] * 3, seed)
            env.decks[0] = [K.wood] * 5 + [K.stone] * 3  # physical order: the whole window is wood
            env.state = check(env.state)
            env._resolve_chance()
            assert env.state.dkind == D_HALI_DECK
            env.step(A.PICK_LEFT)
            s = env.state
            assert s.cards[0][K.wood] == 1 and s.dkind != D_HALI_CHOOSE
            physical_matches_belief(env)  # the belief is never contradicted ...
            if env.decks[0][:4] == [K.wood] * 4:
                all_wood_on_top += 1
        # ... but under a real reshuffle P(4 woods on top of a 4-wood/3-stone deck) = 1/35 per game
        assert all_wood_on_top < n
