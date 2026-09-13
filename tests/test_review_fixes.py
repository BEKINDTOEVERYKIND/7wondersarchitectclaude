"""Regression tests for defects found by the adversarial review."""
import numpy as np

from sevenwa.engine.actions import Actions as A
from sevenwa.engine.rules import RulesConfig
from sevenwa.engine.state import CENTRAL, D_PAY, GameState
from sevenwa.engine.wonders import WONDER_BY_NAME
from sevenwa.search.gumbel import gumbel_search
from sevenwa.search.mcts import MCTS, MCTSConfig
from sevenwa.search.rollout import RolloutEvaluator
from sevenwa.toygame import RaceState
from sevenwa.train.elo import expected, fit_elo
from sevenwa.train.pipeline import evaluate_agents

from conftest import K, check, edit, first_pick, give_many, set_top, settle, W  # noqa: E402  (tests/conftest.py)


def test_olympia_takes_both_cards_before_the_mandatory_build():
    """Olympia stage 2 (2 identical) built with clay x2; the effect draws wood (own deck) and stone
    (opponent deck).  Stage 3 needs 3 different: with wood + stone + the leftover papyrus the build is
    affordable only once BOTH cards have landed, and the payment must be able to use either card."""
    c = edit(first_pick(W.Olympia, W.Giza))
    c.stages[0] = 1
    give_many(c, 0, [K.clay, K.clay, K.papyrus])
    set_top(c, 0, K.wood)
    set_top(c, 1, K.stone)
    s = settle(check(c))  # resolve the reveal chance nodes after the two Olympia draws
    # stage 2 auto-built, both Olympia cards placed, then stage 3 (3 different) built with wood+stone+papyrus
    assert s.stages[0] == 3, s.describe()
    assert s.cards[0][K.wood] == 0 and s.cards[0][K.stone] == 0 and s.cards[0][K.papyrus] == 0


def test_observation_keeps_public_peek_fact():
    from sevenwa.engine.env import Environment
    env = Environment(seed=11)
    s = env.state._copy()
    s.deck_top[CENTRAL] = K.civ3
    s.unseen[CENTRAL][K.civ3] -= 1
    s.central_known_to = 1 << 0
    env.state = s
    other = env.observe(1)
    assert other.deck_top[CENTRAL] == -1 and not other.knows_central(1)
    assert other.has_peeked(0) and not other.knows_central(0)
    from sevenwa.nn.features import encode_state, GLOBAL_SLICE
    f = encode_state(other, player=1)  # from P1's perspective: P0 (the opponent) has peeked
    assert f[GLOBAL_SLICE[0] + 2 * 14 + 1 + 4 + 1 + 3 + 1] == 1.0  # 'opponent has peeked' flag
    assert f[GLOBAL_SLICE[0] + 2 * 14 + 1 + 4 + 1 + 3] == 0.0  # 'I know the card' flag


def test_fit_elo_recovers_true_ratings():
    rng = np.random.default_rng(0)
    true = {"a": 0.0, "b": 300.0, "c": 600.0}
    res = []
    for x in true:
        for y in true:
            if x < y:
                for _ in range(2000):
                    res.append((x, y, 1.0 if rng.random() < expected(true[x], true[y]) else 0.0))
    fit = fit_elo(res, anchor="a")
    assert abs(fit["b"] - 300) < 40 and abs(fit["c"] - 600) < 60
    # a strong, undefeated player gets a large but finite rating
    fit2 = fit_elo([("x", "y", 1.0)] * 50, anchor="y")
    assert 400 < fit2["x"] < 5000


def test_gumbel_budget_is_exactly_spent_and_last_phase_compares_two():
    rng = np.random.default_rng(0)
    m = MCTS(RolloutEvaluator(2, rng=rng), MCTSConfig(num_simulations=64, batch_size=4), rng=rng, num_actions=2)
    r = gumbel_search(m, RaceState((3, 3), 0), 64)
    assert sum(r.visit_counts.values()) == 64
    assert min(r.visit_counts.values()) >= 16
    for node in [r.root] + list(r.root.children.values()):
        assert node.vN == 0 and abs(node.vW) < 1e-9


def test_evaluate_agents_with_zero_games_is_neutral():
    out = evaluate_agents("random", "random", 0, seed=0, log=lambda m: None)
    assert out["games"] == 0 and out["score_a"] == 0.5 and out["mean_margin"] == 0.0


def test_factory_respects_caller_simulations():
    from sevenwa.agents.factory import make_agent
    cfg = MCTSConfig(num_simulations=37)
    a = make_agent("rollout", seed=0, mcts_config=cfg)
    assert a.mcts.cfg.num_simulations == 37 and cfg.num_simulations == 37
    b = make_agent("rollout:12", seed=0, mcts_config=cfg)
    assert b.mcts.cfg.num_simulations == 12 and cfg.num_simulations == 37
