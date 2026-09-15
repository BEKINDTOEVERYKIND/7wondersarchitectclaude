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


# ----------------------------------------------------------------------------- review round: search options
class _FixedPriorEvaluator:
    """Uniform value 0, a fixed (possibly over-peaked) prior over the two toy actions."""

    def __init__(self, priors):
        self.priors = np.asarray(priors, dtype=np.float32)

    def evaluate(self, states):
        B = len(states)
        return np.tile(self.priors, (B, 1)), np.zeros(B, dtype=np.float32)


def test_prior_floor_mixes_uniform_into_root_priors_once():
    rng = np.random.default_rng(0)
    ev = _FixedPriorEvaluator([0.98, 0.02])
    m = MCTS(ev, MCTSConfig(num_simulations=30, batch_size=4, prior_floor=0.2), rng=rng, num_actions=2)
    r = m.run(RaceState((3, 3), 0))
    assert abs(r.root.priors[SAFE] - (0.8 * 0.98 + 0.2 * 0.5)) < 1e-9
    assert abs(r.root.priors[RISKY] - (0.8 * 0.02 + 0.2 * 0.5)) < 1e-9
    assert r.root.floored
    # a second search on the same (reused) root must not compound the floor
    r2 = m.run(RaceState((3, 3), 0))
    assert r2.root is r.root and abs(r2.root.priors[RISKY] - (0.8 * 0.02 + 0.2 * 0.5)) < 1e-9
    # interior nodes keep their raw priors
    child = r.root.children[SAFE]
    assert child.expanded and abs(child.priors[RISKY] - 0.02) < 1e-6
    # floor 0 (default) leaves the priors untouched
    m0 = MCTS(ev, MCTSConfig(num_simulations=10, batch_size=2), rng=rng, num_actions=2)
    r0 = m0.run(RaceState((3, 3), 0))
    assert abs(r0.root.priors[SAFE] - 0.98) < 1e-6 and not r0.root.floored


def test_prior_floor_applies_in_gumbel_search():
    rng = np.random.default_rng(1)
    m = MCTS(_FixedPriorEvaluator([0.99, 0.01]), MCTSConfig(num_simulations=16, batch_size=4, prior_floor=0.5),
             rng=rng, num_actions=2)
    r = gumbel_search(m, RaceState((3, 3), 0), 16)
    assert abs(r.root.priors[RISKY] - (0.5 * 0.01 + 0.25)) < 1e-9


def test_best_action_q_prefers_q_among_well_visited():
    from sevenwa.search.mcts import SearchResult
    res = SearchResult(root=None, visit_counts={0: 100, 1: 40, 2: 5}, root_value=0.0, policy_target=np.zeros(3),
                       extra={"q": {0: 0.10, 1: 0.30, 2: 0.90}})
    assert res.best_action() == 0
    assert res.best_action_q() == 1  # action 2 has the best Q but too few visits
    assert res.best_action_q(min_frac=0.01) == 2
    assert res.best_action_q(min_frac=0.5) == 0
    # no Q information at all: falls back to the visit argmax
    empty = SearchResult(root=None, visit_counts={0: 3, 1: 1}, root_value=0.0, policy_target=np.zeros(2))
    assert empty.best_action_q() == 0


def test_agent_final_pick_q_chooses_a_legal_move():
    from sevenwa.agents.mcts_agent import MCTSAgent
    rng = np.random.default_rng(3)
    ev = RolloutEvaluator(2, rng=rng)
    for root_mode in ("puct", "gumbel"):
        ag = MCTSAgent(ev, MCTSConfig(num_simulations=40, batch_size=4), rng=rng, num_actions=2, root_mode=root_mode,
                       final_pick="q")
        a = ag.select_action(RaceState((7, 9), 0))
        assert a == RISKY  # both the Q pick and the visit pick agree on the only winning try
    with pytest.raises(ValueError):
        MCTSAgent(ev, MCTSConfig(), num_actions=2, final_pick="bogus")


def test_playout_can_return_terminal_margin():
    from sevenwa.search.rollout import random_policy
    rng = np.random.default_rng(0)
    s = RaceState((0, 0), 0)
    r = playout(s, np.random.default_rng(0), random_policy)
    (r2, margin) = playout(s, np.random.default_rng(0), random_policy, return_margin=True)
    assert r == r2 and r in [(1.0, -1.0), (-1.0, 1.0)]  # same RNG stream -> same game
    assert (margin > 0) == (r[0] > 0) and abs(margin) >= 1.0
    env = Environment(seed=2)
    (rr, m) = playout(env.observe(0), rng, return_margin=True)
    assert rr in [(1.0, -1.0), (-1.0, 1.0), (0.0, 0.0)] and float(m) == m


def test_chance_sampling_matches_numpy_choice():
    """The bisect sampler used by playout() and by the tree draws the same distribution as
    rng.choice(p=...) (and, on the cached node cdf, the very same indices for the same seed)."""
    from bisect import bisect_right
    from sevenwa.search.rollout import sample_outcome
    p = np.array([0.05, 0.3, 0.1, 0.25, 0.2, 0.1])
    outs = [(10 + i, float(x)) for i, x in enumerate(p)]
    cdf = p.cumsum()
    cdf /= cdf[-1]
    cdf_list = cdf.tolist()
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    for _ in range(3000):
        assert bisect_right(cdf_list, r1.random()) == int(r2.choice(len(p), p=p))
    r3 = np.random.default_rng(6)
    counts = np.zeros(len(p))
    for _ in range(20000):
        counts[sample_outcome(outs, r3) - 10] += 1
    assert np.abs(counts / counts.sum() - p).max() < 0.02


def test_hybrid_margin_weight_blends_playout_margin():
    from sevenwa.search.rollout import HybridEvaluator, random_policy

    class _Zero:
        def evaluate(self, states):
            return np.full((len(states), 2), 0.5, dtype=np.float32), np.zeros(len(states), dtype=np.float32)

    states = [RaceState((8, 2), 0), RaceState((8, 2), 1)]  # player 0 is far ahead
    plain = HybridEvaluator(_Zero(), 2, lam=1.0, policy_fn=random_policy, rng=np.random.default_rng(0))
    _, v0 = plain.evaluate(states)
    assert all(v in (-1.0, 0.0, 1.0) for v in v0)  # default: pure +-1 results, as before
    blend = HybridEvaluator(_Zero(), 2, lam=1.0, policy_fn=random_policy, rng=np.random.default_rng(0),
                            margin_weight=1.0, playouts_per_leaf=8)
    _, v1 = blend.evaluate(states)
    assert -1.0 < v1[0] < 1.0 and v1[0] > 0.3  # tanh(margin / 8) from the mover's perspective
    assert v1[1] < -0.3  # the same game seen by the trailing player 1
    half = HybridEvaluator(_Zero(), 2, lam=1.0, policy_fn=random_policy, rng=np.random.default_rng(0),
                           margin_weight=0.5, playouts_per_leaf=1)
    _, v2 = half.evaluate(states[:1])
    assert v2[0] != 1.0 and 0.0 < v2[0] <= 1.0


def test_tree_reuse_across_opponent_decision():
    for across, expect_reuse in ((True, True), (False, False)):
        rng = np.random.default_rng(2)
        ev = RolloutEvaluator(2, rng=rng)
        m = MCTS(ev, MCTSConfig(num_simulations=120, batch_size=4, reuse_across_opponent=across), rng=rng, num_actions=2)
        s = RaceState((2, 2), 0)
        m.run(s)
        m.advance(SAFE)  # our own action: the retained root is now the opponent's decision node
        opp_node = m._root
        assert opp_node.to_move == 1 and SAFE in opp_node.children
        after_opp = s.apply_action(SAFE).apply_action(SAFE)  # the opponent answered SAFE
        assert after_opp.to_move() == 0
        node = m._reusable_root(after_opp)
        if expect_reuse:
            assert node is opp_node.children[SAFE] and node.N > 0 and node.state.key() == after_opp.key()
            r = m.run(after_opp)
            assert r.root is node and sum(r.visit_counts.values()) >= 120
        else:
            assert node is None  # today's behaviour: only chance nodes are walked through
        # an opponent line the search never visited is not reused either way
        m.reset()


def test_tree_reuse_across_opponent_and_chance():
    rng = np.random.default_rng(4)
    m = MCTS(RolloutEvaluator(2, rng=rng), MCTSConfig(num_simulations=200, batch_size=4, reuse_across_opponent=True),
             rng=rng, num_actions=2)
    s = RaceState((2, 2), 0)
    m.run(s)
    m.advance(RISKY)  # chance node
    chance = m._root
    assert chance.is_chance
    s_hit = s.apply_action(RISKY).apply_chance(0)  # opponent to move after our hit
    opp = chance.children.get(0)
    if opp is not None and RISKY in opp.children:
        # opponent goes RISKY -> chance -> our decision after a miss
        target = s_hit.apply_action(RISKY).apply_chance(1)
        node = m._reusable_root(target)
        expected = opp.children[RISKY].children.get(1)
        assert node is expected
        if node is not None:
            assert node.state.key() == target.key()


def test_factory_parses_options_and_keeps_defaults():
    from sevenwa.agents.factory import make_agent, parse_spec
    kind, pos, opts = parse_spec("hybrid:models/x.pt:300:0.5:pt=1.5:floor=0.05:root=gumbel:pick=q:batch=8:margin=0.5")
    assert kind == "hybrid" and pos == ["models/x.pt", "300", "0.5"]
    assert opts == {"pt": "1.5", "floor": "0.05", "root": "gumbel", "pick": "q", "batch": "8", "margin": "0.5"}
    a = make_agent("rollout:20:1:heuristic:floor=0.1:pick=q:root=gumbel:batch=4:cpuct=2:fpu=0.1:reuse=all", seed=0)
    cfg = a.mcts.cfg
    assert cfg.num_simulations == 20 and cfg.prior_floor == 0.1 and cfg.batch_size == 4 and cfg.c_puct == 2.0
    assert cfg.fpu_reduction == 0.1 and cfg.reuse_across_opponent and a.root_mode == "gumbel" and a.final_pick == "q"
    assert a.name.startswith("rollout20x1-heuristic[")
    b = make_agent("rollout:12", seed=0)
    assert b.mcts.cfg.batch_size == 1 and b.final_pick == "visits" and b.root_mode == "puct"
    assert b.mcts.cfg.prior_floor == 0.0 and not b.mcts.cfg.reuse_across_opponent and b.name == "rollout12x1-heuristic"
    c = make_agent("rollout:floor=0.2:8", seed=0)  # options and positionals may interleave
    assert c.mcts.cfg.num_simulations == 8 and c.mcts.cfg.prior_floor == 0.2
    for bad in ("heuristic:pt=2", "rollout:10:margin=0.5", "rollout:pick=x", "rollout:root=x", "rollout:reuse=x",
                "rollout:floor=2", "rollout:batch=0", "rollout:floor=0.1:floor=0.2", "net", "bogus"):
        with pytest.raises(ValueError):
            make_agent(bad, seed=0)
