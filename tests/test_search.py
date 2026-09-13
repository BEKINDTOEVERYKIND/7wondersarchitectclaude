"""Search tests on the toy race game (exact chance handling) and on the real engine."""
import numpy as np
import pytest

from sevenwa.engine import Environment
from sevenwa.engine.actions import Actions
from sevenwa.search.gumbel import gumbel_search
from sevenwa.search.mcts import MCTS, MCTSConfig
from sevenwa.search.rollout import RolloutEvaluator, playout
from sevenwa.toygame import RISKY, SAFE, RaceState


def _mcts(num_actions=2, sims=600, seed=0, batch=8):
    rng = np.random.default_rng(seed)
    ev = RolloutEvaluator(num_actions, playouts_per_leaf=1, rng=rng)
    return MCTS(ev, MCTSConfig(num_simulations=sims, batch_size=batch), rng=rng, num_actions=num_actions)


def test_prefers_risky_when_behind():
    m = _mcts()
    r = m.run(RaceState((7, 9), 0))
    assert r.best_action() == RISKY
    assert r.visit_counts[RISKY] > 3 * r.visit_counts[SAFE]


def test_prefers_safe_win():
    m = _mcts()
    r = m.run(RaceState((7, 9), 1))
    assert r.best_action() == SAFE
    assert r.root_value > 0.8


def test_value_frame_is_mover_perspective():
    m = _mcts()
    r0 = m.run(RaceState((9, 0), 0))
    assert r0.root_value > 0.9
    m.reset()
    r1 = m.run(RaceState((9, 0), 1))  # player 1 to move, far behind
    assert r1.root_value < -0.5


def test_policy_target_normalised_and_masked():
    m = _mcts()
    r = m.run(RaceState((3, 3), 0))
    assert r.policy_target.shape == (2,)
    assert abs(r.policy_target.sum() - 1.0) < 1e-6
    assert sum(r.visit_counts.values()) == 600


def test_virtual_loss_is_removed():
    m = _mcts(sims=200, batch=16)
    r = m.run(RaceState((0, 0), 0))
    for node in [r.root] + list(r.root.children.values()):
        assert node.vN == 0 and abs(node.vW) < 1e-9


def test_chance_node_expectation():
    """At (6,9) for player 0: RISKY hits with p=0.5 -> 9 (then opponent wins with SAFE), so both actions lose.
    With p_hit=1 RISKY wins immediately from 7."""
    m = _mcts(sims=300)
    r = m.run(RaceState((7, 9), 0, p_hit=1.0))
    assert r.best_action() == RISKY and r.root_value > 0.9


def test_gumbel_selects_good_action():
    m = _mcts(sims=200)
    r = gumbel_search(m, RaceState((7, 9), 0), 200)
    assert r.extra["chosen"] == RISKY
    assert abs(r.improved_policy.sum() - 1.0) < 1e-5


def test_tree_reuse_advances():
    m = _mcts(sims=100)
    s = RaceState((0, 0), 0)
    r = m.run(s)
    a = r.best_action()
    m.advance(a)
    s2 = s.apply_action(a)
    if s2.is_chance():
        s2 = s2.apply_chance(0)
    root = m._reusable_root(s2)
    assert root is None or root.state.key() == s2.key()


def test_mcts_on_real_engine_plays_full_game():
    rng = np.random.default_rng(1)
    ev = RolloutEvaluator(Actions.NUM, playouts_per_leaf=1, rng=rng)
    m = MCTS(ev, MCTSConfig(num_simulations=16, batch_size=1), rng=rng, num_actions=Actions.NUM)
    env = Environment(seed=1)
    n = 0
    while not env.is_terminal():
        p = env.to_move()
        obs = env.observe(p)
        r = m.run(obs)
        a = r.best_action()
        assert a in obs.legal_actions()
        env.step(a)
        m.advance(a)
        n += 1
        assert n < 500
    assert env.returns() != (0.0, 0.0) or env.scores()[0] == env.scores()[1]


def test_playout_terminates():
    rng = np.random.default_rng(0)
    env = Environment(seed=2)
    r = playout(env.observe(0), rng)
    assert r in [(1.0, -1.0), (-1.0, 1.0), (0.0, 0.0)]
