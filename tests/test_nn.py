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
from sevenwa.nn import features as FT
from sevenwa.nn.features import (ENTITY_SLICES, ENTITY_SLICES_V2, EXPERT_PRIOR_SLICE, EXPERT_SLICE, FEATURE_SIZE,
                                 FEATURE_SIZE_V1, FEATURE_SIZE_V2, FEATURE_VERSION_LATEST, encode, encode_state,
                                 encode_state_v1, encode_state_v2, encode_v1, encode_v2, encoder_for_dim, entity_slices,
                                 feature_size, legal_mask)
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
    F1 = np.stack([encode_state(s, version=1) for s in states])
    assert F1.shape == (len(states), FEATURE_SIZE_V1) and FEATURE_SIZE_V1 == 581
    assert np.isfinite(F1).all()
    assert F1.min() >= 0.0 and F1.max() <= 2.0
    assert ENTITY_SLICES[-1][1] == FEATURE_SIZE_V1
    F = np.stack([encode_state(s) for s in states])  # default: the latest version
    assert FEATURE_VERSION_LATEST == 2 and FEATURE_SIZE == FEATURE_SIZE_V2 == feature_size(2)
    assert F.shape == (len(states), FEATURE_SIZE)
    assert np.isfinite(F).all() and F.min() >= -3.0 and F.max() <= 4.0
    assert ENTITY_SLICES_V2[:-1] == ENTITY_SLICES and ENTITY_SLICES_V2[-1] == EXPERT_SLICE == (FEATURE_SIZE_V1, FEATURE_SIZE)
    assert entity_slices(1) == ENTITY_SLICES and entity_slices(2) == ENTITY_SLICES_V2


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
    for trunk, version in (("resmlp", 1), ("resmlp", 2), ("attention", 1), ("attention", 2)):
        n = feature_size(version)
        net = PolicyValueNet(NetConfig(input_dim=n, num_actions=Actions.NUM, trunk=trunk, width=64, depth=2,
                                       entity_slices=entity_slices(version), attn_dim=32, attn_heads=2, attn_layers=1))
        x = torch.randn(4, n)
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
                                       entity_slices=entity_slices(), attn_dim=32, attn_heads=2, attn_layers=1,
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


# ----------------------------------------------------------------------------- feature version 2 (EXPERT block)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMITATION_V7 = os.path.join(REPO, "models", "imitation_v7.pt")
VALUE_V8 = os.path.join(REPO, "models", "value_v8.pt")


def _heuristic_states(n_games=4, seed=0, eps=0.3):
    """Observations from ε-greedy heuristic self-play (all decision kinds, both seats)."""
    from dataclasses import replace
    from sevenwa.agents.heuristic import HeuristicParams, heuristic_action
    rng = np.random.default_rng(seed)
    params = replace(HeuristicParams(), epsilon=eps)
    out = []
    for g in range(n_games):
        env = Environment(seed=1000 + seed + g)
        while not env.is_terminal():
            obs = env.observe(env.to_move())
            out.append(obs)
            env.step(heuristic_action(obs, rng, params))
    return out


def _belief_walk(seed=0):
    """Every node (decision, chance and the terminal one) of a random game on the belief state."""
    from sevenwa.engine.state import GameState
    rng = np.random.default_rng(seed)
    s = GameState.initial((seed % 7, (seed + 3) % 7))
    out = [s]
    while not s.is_terminal():
        if s.is_chance():
            outs = s.chance_outcomes()
            probs = np.array([p for _, p in outs])
            s = s.apply_chance(int(outs[int(rng.choice(len(outs), p=probs / probs.sum()))][0]))
        else:
            la = list(s.legal_actions())
            s = s.apply_action(int(la[rng.integers(len(la))]))
        out.append(s)
    return out


def test_v2_extends_v1_and_expert_block_is_sane():
    from sevenwa.agents.heuristic import heuristic_prior
    from sevenwa.engine.state import N_DECISION
    failures = FT.EXPERT_FAILURES
    states = _heuristic_states() + _random_states(2, seed=5)
    kinds = set()
    for i, s in enumerate(states):
        players = (None, 1 - s.mover) if i % 5 == 0 else (None,)  # also the non-mover's perspective
        for pl in players:
            v1 = encode_state(s, pl, version=1)
            v2 = encode_state(s, pl, version=2)
            assert v1.shape == (FEATURE_SIZE_V1,) and v2.shape == (FEATURE_SIZE_V2,) and v2.dtype == np.float32
            assert np.array_equal(v2[:FEATURE_SIZE_V1], v1)
            ex = v2[EXPERT_SLICE[0]:]
            assert np.isfinite(ex).all() and ex.min() >= -3.0 and ex.max() <= 4.0
        kinds.add(s.dkind)
        # the prior part is the heuristic prior of the decision (an observation hides nothing more from its mover)
        prior = encode_state(s)[EXPERT_PRIOR_SLICE[0]:EXPERT_PRIOR_SLICE[1]]
        if len(s.legal_actions()) >= 2:
            assert np.array_equal(prior, heuristic_prior(s)), s.describe_node()
            assert abs(prior.sum() - 1.0) < 1e-5 and (prior[~legal_mask(s)] == 0).all()
        else:
            assert (prior == 0).all()
        assert s.node_type == N_DECISION
    assert len(kinds) >= 4  # picks, payments, tokens, stages ... are all covered
    # chance and terminal nodes: zero EXPERT block, never an exception
    nodes = _belief_walk(3)
    n_dec = 0
    for s in nodes:
        v2 = encode_state(s, version=2)
        assert np.array_equal(v2[:FEATURE_SIZE_V1], encode_state(s, version=1))
        if s.node_type == N_DECISION:
            n_dec += 1
            assert np.abs(v2[EXPERT_SLICE[0]:]).sum() > 0
        else:
            assert (v2[EXPERT_SLICE[0]:] == 0).all()
    assert n_dec > 10 and n_dec < len(nodes)
    assert FT.EXPERT_FAILURES == failures  # the heuristic never raised (no silent zero fallback)


def test_expert_stage_deficits_agree_with_v1_affordable_flags():
    """Per stage: deficit 0 for built stages, and for available ones deficit == 0 exactly when the
    version-1 block marks the stage affordable (both use coins as wilds + Engineering / Economy)."""
    from sevenwa.engine.wonders import NUM_STAGES, WONDERS
    for s in _heuristic_states(3, seed=7):
        f = encode_state(s, version=2)
        for q, p in enumerate((s.mover, 1 - s.mover)):
            base = (FT.ME_SLICE if q == 0 else FT.OPP_SLICE)[0] + FT.NUM_WONDERS + 6
            ex = EXPERT_SLICE[0] + q * FT.EXPERT_PLAYER_FEATS
            avail = WONDERS[s.wonder[p]].available(s.built[p])
            for i in range(NUM_STAGES):
                o = base + i * FT.STAGE_FEATS
                d = f[ex + i]
                assert 0.0 <= d <= 1.0
                if (s.built[p] >> i) & 1:
                    assert f[o] == 1.0 and d == 0.0
                elif i in avail:
                    assert f[o + 1] == 1.0
                    assert (d == 0.0) == (f[o + 2] == 1.0), (s.describe_node(), p, i)
            assert f[ex + NUM_STAGES + 2] == len(avail) / 3.0 or s.built[p] == 31


def test_expert_block_reads_no_hidden_central_card():
    """A central top card only the opponent knows (their Cat peek, as kept by an omniscient state or an
    in-tree sample) must not change the mover's encoding, whatever the card is."""
    env = Environment(seed=3)
    base = env.state._copy()
    base._legal = env.state._legal
    me = base.mover
    base.central_known_to = 1 << (1 - me)
    encs = []
    for kind in (0, 5, 13):  # wood, coin, a horn card (would change the battle probability)
        s = base._copy()
        s._legal = base._legal
        assert s.unseen[CENTRAL][kind] > 0
        s.unseen[CENTRAL][kind] -= 1
        s.deck_top[CENTRAL] = kind
        assert s.has_peeked(1 - me) and not s.knows_central(me)
        encs.append(encode_state(s, version=2))
        # the knower's perspective does see it
        assert not np.array_equal(encode_state(s, 1 - me, version=2), encode_state(base, 1 - me, version=2))
    ref = encode_state(base, version=2)  # the same belief with the card still among the unseen ones
    for e in encs:
        assert np.array_equal(e, ref)
    # the mover peeked: encoded for the other player, the mover's prior must not reveal the card either
    base.central_known_to = 1 << me
    encs = []
    for kind in (0, 5, 13):
        s = base._copy()
        s._legal = base._legal
        s.unseen[CENTRAL][kind] -= 1
        s.deck_top[CENTRAL] = kind
        assert s.knows_central(me)
        encs.append(encode_state(s, 1 - me, version=2))
        assert np.abs(encode_state(s, version=2)[EXPERT_SLICE[0]:]).sum() > 0
    ref = encode_state(base, 1 - me, version=2)
    for e in encs:
        assert np.array_equal(e, ref)


def test_encoder_for_dim_and_versions():
    v, enc, enc_state = encoder_for_dim(581)
    assert v == 1 and enc is encode_v1 and enc_state is encode_state_v1
    v, enc, enc_state = encoder_for_dim(FEATURE_SIZE_V2)
    assert v == 2 and enc is encode_v2 and enc_state is encode_state_v2
    s = _random_states(1)[3]
    for dim in (581, FEATURE_SIZE_V2):
        _, enc, enc_state = encoder_for_dim(dim)
        f, m = enc(s)
        assert f.shape == (dim,) and np.array_equal(f, enc_state(s)) and np.array_equal(m, legal_mask(s))
    assert np.array_equal(encode(s)[0], encode_v2(s)[0]) and np.array_equal(encode(s, 1)[0], encode_v1(s)[0])
    for bad in (437, 0, FEATURE_SIZE_V2 + 1):
        with np.testing.assert_raises(ValueError):
            encoder_for_dim(bad)
    for bad in (0, 3, "x"):
        with np.testing.assert_raises(ValueError):
            feature_size(bad)
        with np.testing.assert_raises(ValueError):
            encode_state(s, version=bad)


def test_shipped_checkpoints_are_served_version_1_unchanged():
    """``net:models/imitation_v7.pt`` (input_dim 581) through the factory gives exactly the priors / value
    of the same network fed with the version-1 encoder directly."""
    import pytest
    from sevenwa.agents.factory import make_agent
    if not os.path.exists(IMITATION_V7):
        pytest.skip("models/imitation_v7.pt not available")
    env = Environment(seed=1)
    obs = env.observe(env.to_move())
    for spec, path in ((f"net:{IMITATION_V7}:8", IMITATION_V7), (f"hybrid:{IMITATION_V7}:8:0.0", IMITATION_V7),
                       (f"netraw:{IMITATION_V7}", IMITATION_V7), (f"net:{VALUE_V8}:8", VALUE_V8)):
        if not os.path.exists(path):
            continue
        agent = make_agent(spec, seed=0)
        ev = agent.evaluator if hasattr(agent, "evaluator") else agent.mcts.evaluator
        tev = ev if isinstance(ev, TorchEvaluator) else ev.net  # HybridEvaluator wraps the TorchEvaluator
        assert tev.net.cfg.input_dim == 581 and tev.encode is encode_v1
        p, v = ev.evaluate([obs])
        net = PolicyValueNet.load(path)
        ref_p, ref_v = TorchEvaluator(net, lambda s: (encode_state(s, version=1), legal_mask(s))).evaluate([obs])
        assert np.array_equal(p, ref_p) and np.array_equal(v, ref_v), spec
        with torch.no_grad():  # and the same as an eager forward pass on the v1 features
            logits, v_eager, _ = net(torch.from_numpy(encode_state(obs, version=1)[None]))
            logits = logits.masked_fill(~torch.from_numpy(legal_mask(obs)[None]), -1e9)
            p_eager = torch.softmax(logits, -1).numpy()
        assert np.abs(p - p_eager).max() < 1e-5 and abs(float(v[0]) - float(v_eager[0])) < 1e-5
    # the search runs end to end on the old checkpoint
    a = make_agent(f"net:{IMITATION_V7}:8", seed=0)
    assert a.select_action(obs) in obs.legal_actions()


def test_v2_network_evaluates_through_torch_evaluator_and_factory():
    from sevenwa.agents.factory import make_agent
    states = _heuristic_states(1)[:12]
    for trunk in ("resmlp", "attention"):
        net = new_network(PipelineConfig(net_width=32, net_depth=1, net_trunk=trunk))
        assert net.cfg.input_dim == FEATURE_SIZE_V2 and net.cfg.entity_slices == ENTITY_SLICES_V2
        version, enc, _ = encoder_for_dim(net.cfg.input_dim)
        assert version == 2
        pols, vals = TorchEvaluator(net, enc).evaluate(states)
        assert pols.shape == (len(states), Actions.NUM) and np.isfinite(vals).all()
        for i, s in enumerate(states):
            assert pols[i][~legal_mask(s)].max() < 1e-6 and abs(pols[i].sum() - 1.0) < 1e-4
    v1_net = new_network(PipelineConfig(net_width=32, net_depth=1, feature_version=1))
    assert v1_net.cfg.input_dim == 581 and v1_net.cfg.entity_slices == ENTITY_SLICES
    with tempfile.TemporaryDirectory() as d:
        for version, n in ((2, new_network(PipelineConfig(net_width=32, net_depth=1))), (1, v1_net)):
            path = os.path.join(d, f"v{version}.pt")
            n.save(path)
            a = make_agent(f"net:{path}:8", seed=0)
            assert a.mcts.evaluator.encode is (encode_v2 if version == 2 else encode_v1)
            assert a.select_action(states[0]) in states[0].legal_actions()


def test_selfplay_writes_the_checkpoint_version():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "v1.pt")
        new_network(PipelineConfig(net_width=32, net_depth=1, feature_version=1)).save(path)
        cfg = SelfPlayConfig(num_simulations=8, batch_size=4)
        out = os.path.join(d, "gen0000_v1.npz")
        r = selfplay_worker({"config": asdict(cfg), "checkpoint": path, "seeds": [5], "out_path": out, "generation": 0})
        assert r["feature_version"] == 1 and np.load(out)["feats"].shape[1] == 581
        with np.testing.assert_raises(ValueError):  # an explicit version that contradicts the checkpoint
            selfplay_worker({"config": asdict(cfg), "checkpoint": path, "seeds": [5], "feature_version": 2})
        # no network (rollout evaluator): the requested version, default the latest
        rcfg = SelfPlayConfig(num_simulations=4, batch_size=1, evaluator="rollout")
        out = os.path.join(d, "gen0000_r.npz")
        r = selfplay_worker({"config": asdict(rcfg), "checkpoint": None, "seeds": [6], "out_path": out})
        assert r["feature_version"] == 2 and np.load(out)["feats"].shape[1] == FEATURE_SIZE_V2


def test_imitation_worker_writes_requested_feature_version():
    from sevenwa.train.selfplay import imitation_worker
    with tempfile.TemporaryDirectory() as d:
        shapes = {}
        for version in (1, 2):
            out = os.path.join(d, f"gen0000_j{version}.npz")
            r = imitation_worker({"seeds": [21, 22], "epsilon": 0.1, "feature_version": version, "out_path": out,
                                  "rng_seed": 3})
            data = np.load(out)
            assert r["games"] == 2 and r["feature_version"] == version and len(data["z"]) == r["samples"] > 40
            shapes[version] = data["feats"].shape
            assert data["feats"].shape[1] == feature_size(version)
            assert np.allclose(data["pi"].sum(1), 1.0, atol=1e-5) and (data["pi"][~data["mask"]] == 0).all()
            if version == 2:  # the policy target is the heuristic prior, which v2 also carries as input
                multi = data["mask"].sum(1) >= 2
                assert np.array_equal(data["pi"][multi], data["feats"][multi, EXPERT_PRIOR_SLICE[0]:EXPERT_PRIOR_SLICE[1]])
        assert shapes[1][0] == shapes[2][0]  # same games (same seeds and rng), different widths
        buf = ReplayBuffer(FEATURE_SIZE_V2, Actions.NUM)
        buf.load_shards(os.path.join(d, "gen0000_j2.npz"))
        from sevenwa.train.pipeline import check_feature_width
        check_feature_width(buf, FEATURE_SIZE_V2)
        buf.load_shards(os.path.join(d, "gen0000_j1.npz"))
        with np.testing.assert_raises(ValueError):
            check_feature_width(buf, FEATURE_SIZE_V2)


def test_leaf_relabel_writes_the_checkpoint_version_by_default():
    """``scripts/leaf_relabel.py`` harvests the network's own search tree, and its shards train that
    network (``expert_iteration.py --teacher <same checkpoint> --extra-search-glob``), which rejects
    another width: by default the samples use the checkpoint's version, not the latest one."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_leaf_relabel", os.path.join(REPO, "scripts", "leaf_relabel.py"))
    leaf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(leaf)
    threads = torch.get_num_threads()
    try:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "v1.pt")
            new_network(PipelineConfig(net_width=32, net_depth=1, feature_version=1)).save(path)
            job = {"net": path, "games": [(4, "self", 0)], "sims": 8, "playouts": 1, "leaves": 2, "every": 3,
                   "min_visits": 1, "opp_epsilon": 0.1, "root_mode": "puct", "temp_moves": 0, "rng_seed": 0,
                   "generation": 1}
            for requested, width in ((None, FEATURE_SIZE_V1), (2, FEATURE_SIZE_V2)):
                out = os.path.join(d, f"leaf_{requested}.npz")
                r = leaf.worker({**job, "out_path": out, "feature_version": requested})
                data = np.load(out)
                assert r["samples"] > 0 and data["feats"].shape == (r["samples"], width)
                assert r["feature_version"] == (1 if requested is None else requested)
    finally:
        torch.set_num_threads(threads)
