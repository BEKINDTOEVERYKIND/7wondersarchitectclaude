"""Feature encoder, network and training-loop tests."""
import os
import tempfile
from dataclasses import asdict

import numpy as np
import torch

from sevenwa.engine import Environment
from sevenwa.engine.actions import Actions
from sevenwa.engine.state import CENTRAL
from sevenwa.nn.evaluator import TorchEvaluator
from sevenwa.nn.features import ENTITY_SLICES, FEATURE_SIZE, encode, encode_state, legal_mask
from sevenwa.nn.model import NetConfig, PolicyValueNet, compute_loss
from sevenwa.train.pipeline import PipelineConfig, new_network
from sevenwa.train.replay import ReplayBuffer
from sevenwa.train.selfplay import SelfPlayConfig, selfplay_worker
from sevenwa.train.trainer import TrainConfig, Trainer


def _random_states(n_games=3, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for g in range(n_games):
        env = Environment(seed=seed + g)
        while not env.is_terminal():
            p = env.to_move()
            obs = env.observe(p)
            out.append(obs)
            la = obs.legal_actions()
            env.step(int(la[rng.integers(len(la))]))
    return out


def test_feature_shapes_and_ranges():
    states = _random_states()
    F = np.stack([encode_state(s) for s in states])
    assert F.shape == (len(states), FEATURE_SIZE)
    assert np.isfinite(F).all()
    assert F.min() >= 0.0 and F.max() <= 2.0
    assert ENTITY_SLICES[-1][1] == FEATURE_SIZE


def test_mask_matches_legal_actions():
    for s in _random_states(1):
        m = legal_mask(s)
        assert m.shape == (Actions.NUM,)
        assert set(np.flatnonzero(m)) == set(s.legal_actions())


def test_encoder_hides_central_card_from_non_holder():
    env = Environment(seed=3)
    s = env.state._copy()
    s.deck_top[CENTRAL] = 0
    s.central_known_to = 1 << (1 - s.mover)
    f_hidden = encode_state(s)
    s2 = s._copy()
    s2.central_known_to = 1 << s.mover
    f_known = encode_state(s2)
    assert not np.array_equal(f_hidden, f_known)
    # hidden view: the 'known' flag of the central deck block is 0
    from sevenwa.nn.features import DECK_CENTER_SLICE
    assert f_hidden[DECK_CENTER_SLICE[0] + 2] == 0.0 and f_known[DECK_CENTER_SLICE[0] + 2] == 1.0


def test_network_forward_and_loss():
    for trunk in ("resmlp", "attention"):
        net = PolicyValueNet(NetConfig(input_dim=FEATURE_SIZE, num_actions=Actions.NUM, trunk=trunk, width=64, depth=2,
                                       entity_slices=list(ENTITY_SLICES), attn_dim=32, attn_heads=2, attn_layers=1))
        x = torch.randn(4, FEATURE_SIZE)
        p, v, s = net(x)
        assert p.shape == (4, Actions.NUM) and v.shape == (4,) and s.shape == (4,)
        assert (v.abs() <= 1).all()
        mask = torch.zeros(4, Actions.NUM, dtype=torch.bool)
        mask[:, :3] = True
        pi = torch.zeros(4, Actions.NUM)
        pi[:, 0] = 1.0
        loss, parts = compute_loss(net, x, mask, pi, torch.zeros(4), torch.zeros(4))
        assert torch.isfinite(loss)
        loss.backward()


def test_evaluator_masks_and_caches():
    net = new_network(PipelineConfig(net_width=64, net_depth=2))
    ev = TorchEvaluator(net, encode)
    states = _random_states(1)[:10]
    pols, vals = ev.evaluate(states)
    assert pols.shape == (10, Actions.NUM) and vals.shape == (10,)
    for i, s in enumerate(states):
        illegal = ~legal_mask(s)
        assert pols[i][illegal].max() < 1e-6
        assert abs(pols[i].sum() - 1.0) < 1e-4
    ev.evaluate(states)
    assert ev.hits == 10


def test_save_load_roundtrip():
    net = new_network(PipelineConfig(net_width=64, net_depth=2))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "n.pt")
        net.save(path, extra={"generation": 3})
        net2 = PolicyValueNet.load(path)
        net.eval()  # the default network has dropout; compare in inference mode
        x = torch.randn(2, FEATURE_SIZE)
        with torch.no_grad():
            assert torch.allclose(net(x)[0], net2(x)[0])


def test_selfplay_and_training_end_to_end():
    net = new_network(PipelineConfig(net_width=64, net_depth=2))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "n.pt")
        net.save(path)
        for mode in ("puct", "gumbel"):
            cfg = SelfPlayConfig(num_simulations=12, batch_size=4, root_mode=mode)
            out = os.path.join(d, f"gen0000_{mode}.npz")
            r = selfplay_worker({"config": asdict(cfg), "checkpoint": path, "seeds": [11], "out_path": out, "generation": 0})
            assert r["games"] == 1 and r["samples"] > 20
        buf = ReplayBuffer(FEATURE_SIZE, Actions.NUM)
        n = buf.load_shards(os.path.join(d, "gen*.npz"))
        assert n > 40
        # targets are consistent: z in {-1,0,1}, policies normalised over legal actions
        assert set(np.unique(buf.z[0])).issubset({-1.0, 0.0, 1.0})
        assert np.allclose(buf.pi[0].sum(1), 1.0, atol=1e-4)
        assert (buf.pi[0][~buf.mask[0]] == 0).all()
        tr = Trainer(net, TrainConfig(batch_size=32, epochs=1.0, num_threads=1))
        stats = tr.train(buf, np.random.default_rng(0), log=lambda m: None)
        assert stats["steps"] >= 1 and np.isfinite(stats["total"])


# ----------------------------------------------------------------------------- review round: evaluator options
def test_jit_evaluator_matches_eager():
    states = _random_states(1)[:10]
    for trunk in ("resmlp", "attention"):
        net = PolicyValueNet(NetConfig(input_dim=FEATURE_SIZE, num_actions=Actions.NUM, trunk=trunk, width=64, depth=2,
                                       entity_slices=list(ENTITY_SLICES), attn_dim=32, attn_heads=2, attn_layers=1,
                                       dropout=0.1))
        eager = TorchEvaluator(net, encode, jit=False)
        jit = TorchEvaluator(net, encode, jit=True)
        assert not eager.jit_enabled
        if trunk == "resmlp":
            assert jit.jit_enabled  # the shipped architecture must actually be traced
        p0, v0 = eager.evaluate(states)
        p1, v1 = jit.evaluate(states)
        assert np.abs(p0 - p1).max() < 1e-5 and np.abs(v0 - v1).max() < 1e-5


def test_policy_temperature_flattens_priors():
    net = new_network(PipelineConfig(net_width=64, net_depth=2))
    states = _random_states(1)[:6]
    p1, v1 = TorchEvaluator(net, encode, jit=False).evaluate(states)
    p2, v2 = TorchEvaluator(net, encode, jit=False, policy_temperature=3.0).evaluate(states)
    assert np.allclose(v1, v2)  # the value is untouched
    for i, s in enumerate(states):
        legal = np.asarray(list(s.legal_actions()))
        assert abs(p2[i].sum() - 1.0) < 1e-4 and p2[i][~legal_mask(s)].max() < 1e-6
        if len(legal) > 1 and p1[i][legal].max() > 1.0 / len(legal) + 1e-3:
            assert p2[i][legal].max() < p1[i][legal].max()  # closer to uniform
            assert np.argmax(p2[i]) == np.argmax(p1[i])  # same ordering
    with np.testing.assert_raises(ValueError):
        TorchEvaluator(net, encode, jit=False, policy_temperature=0.0)


def test_factory_hybrid_options():
    from sevenwa.agents.factory import make_agent
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "n.pt")
        new_network(PipelineConfig(net_width=32, net_depth=1)).save(path)
        a = make_agent(f"hybrid:{path}:8:0.5:pt=1.5:floor=0.05:pick=q:margin=0.5:batch=8:root=gumbel", seed=0)
        ev = a.mcts.evaluator
        assert ev.net.policy_temperature == 1.5 and ev.margin_weight == 0.5 and ev.lam == 0.5
        assert a.mcts.cfg.prior_floor == 0.05 and a.mcts.cfg.batch_size == 8 and a.mcts.cfg.num_simulations == 8
        assert a.final_pick == "q" and a.root_mode == "gumbel"
        b = make_agent(f"hybrid:{path}:8:0.5", seed=0)
        assert b.mcts.evaluator.net.policy_temperature == 1.0 and b.mcts.evaluator.margin_weight == 0.0
        assert b.mcts.cfg.batch_size == 1 and b.mcts.cfg.prior_floor == 0.0 and b.final_pick == "visits"
        assert b.name == f"hybrid8(lam=0.5,{path})"
        c = make_agent(f"net:{path}:8:2.0:pt=2:cpuct=3:batch=2", seed=0)
        assert c.mcts.evaluator.policy_temperature == 2.0 and c.mcts.cfg.c_puct == 3.0 and c.mcts.cfg.batch_size == 2
        r = make_agent(f"netraw:{path}:pt=2", seed=0)
        assert r.evaluator.policy_temperature == 2.0
        env = Environment(seed=1)
        obs = env.observe(0)
        assert a.select_action(obs) in obs.legal_actions()
