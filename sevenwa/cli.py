"""Command-line interface: ``python -m sevenwa.cli <command>`` (or ``sevenwa <command>``)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict

import numpy as np


def cmd_pipeline(args):
    from .train.pipeline import PipelineConfig, apply_overrides, config_from_dict, run_pipeline
    from .train.selfplay import SelfPlayConfig
    from .train.trainer import TrainConfig
    if args.config:
        with open(args.config) as f:
            cfg = config_from_dict(json.load(f))
        cfg.run_dir = args.run_dir if args.run_dir != "runs/default" else cfg.run_dir
        run_pipeline(apply_overrides(cfg, args.set))
        return
    cfg = PipelineConfig(run_dir=args.run_dir, generations=args.generations, games_per_generation=args.games,
                         bootstrap_games=args.bootstrap_games, bootstrap_simulations=args.bootstrap_sims,
                         bootstrap_mode=args.bootstrap_mode, bootstrap_epochs=args.bootstrap_epochs,
                         window_generations=args.window, num_workers=args.workers, seed=args.seed,
                         net_width=args.width, net_depth=args.depth, net_trunk=args.trunk, net_dropout=args.dropout,
                         selfplay=SelfPlayConfig(num_simulations=args.sims, temperature_moves=args.temp_moves, root_mode=args.root_mode, value_mix=args.value_mix),
                         train=TrainConfig(batch_size=args.batch, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, num_threads=args.workers),
                         gate_games=args.gate_games, gate_threshold=args.gate_threshold, gate_simulations=args.gate_sims,
                         eval_games=args.eval_games, eval_opponents=args.eval_opponents.split(",") if args.eval_opponents else [],
                         eval_simulations=args.eval_sims, eval_every=args.eval_every)
    run_pipeline(apply_overrides(cfg, args.set))


def cmd_arena(args):
    from .agents.factory import make_agent
    from .engine.env import make_env
    from .search.mcts import MCTSConfig
    from .train.arena import play_match
    a = make_agent(args.a, seed=args.seed, mcts_config=MCTSConfig(num_simulations=args.sims))
    b = make_agent(args.b, seed=args.seed + 1, mcts_config=MCTSConfig(num_simulations=args.sims))
    t = time.time()
    res = play_match(make_env, (a, b), args.games, seed=args.seed, log=print if args.verbose else None)
    print(res.summary((a.name, b.name)), f"({time.time() - t:.1f}s)")
    if args.json:
        print(json.dumps({"a": args.a, "b": args.b, "score_a": res.score(0), "wins": res.wins, "draws": res.draws,
                          "margins": res.score_diffs}))


def cmd_selfplay(args):
    from .train.selfplay import SelfPlayConfig, run_selfplay
    cfg = SelfPlayConfig(num_simulations=args.sims, evaluator="net" if args.checkpoint else "rollout")
    run_selfplay(cfg, args.games, args.checkpoint, args.out_dir, args.generation, args.workers, seed=args.seed)


def cmd_train(args):
    from .engine.actions import Actions
    from .nn.features import FEATURE_SIZE
    from .nn.model import PolicyValueNet
    from .train.pipeline import PipelineConfig, new_network
    from .train.replay import ReplayBuffer
    from .train.trainer import TrainConfig, Trainer
    buf = ReplayBuffer(FEATURE_SIZE, Actions.NUM)
    n = buf.load_shards(args.data)
    print(f"loaded {n} samples: {buf.stats()}")
    net = PolicyValueNet.load(args.init) if args.init else new_network(PipelineConfig(net_width=args.width, net_depth=args.depth))
    tr = Trainer(net, TrainConfig(batch_size=args.batch, epochs=args.epochs, lr=args.lr))
    print(tr.train(buf, np.random.default_rng(args.seed)))
    net.save(args.out)
    print(f"saved {args.out}")


def cmd_play(args):
    """Play against an agent in the terminal."""
    from .agents.factory import make_agent
    from .engine.actions import Actions
    from .engine.env import Environment
    from .search.mcts import MCTSConfig
    ai = make_agent(args.ai, seed=args.seed, mcts_config=MCTSConfig(num_simulations=args.sims))
    env = Environment(seed=args.seed)
    human = args.human
    ai.new_game()
    while not env.is_terminal():
        p = env.to_move()
        obs = env.observe(p)
        if p == human:
            print(obs.describe())
            legal = list(obs.legal_actions())
            for i, a in enumerate(legal):
                print(f"  [{i}] {Actions.name(a)}")
            while True:
                try:
                    choice = int(input("your choice> "))
                    if 0 <= choice < len(legal):
                        break
                except (ValueError, EOFError):
                    pass
            env.step(legal[choice])
        else:
            a = ai.select_action(obs)
            print(f"AI plays {Actions.name(a)}")
            env.step(a)
    print(env.state.describe())
    print("final scores", env.scores(), "returns", env.returns())


def cmd_transcript(args):
    """Play one annotated game between two agents and write a markdown transcript."""
    from .agents.factory import make_agent
    from .search.mcts import MCTSConfig
    from .train.transcript import play_transcript
    a = make_agent(args.a, seed=args.seed, mcts_config=MCTSConfig(num_simulations=args.sims))
    b = make_agent(args.b, seed=args.seed + 1, mcts_config=MCTSConfig(num_simulations=args.sims))
    wonders = tuple(int(x) for x in args.wonders.split(",")) if args.wonders else None
    text = play_transcript((a, b), seed=args.seed, wonders=wonders)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)


def cmd_bench(args):
    from .agents.factory import make_agent
    from .engine.env import Environment
    from .search.mcts import MCTSConfig
    agent = make_agent(args.agent, seed=0, mcts_config=MCTSConfig(num_simulations=args.sims))
    env = Environment(seed=0)
    agent.new_game()
    t = time.time()
    n = 0
    while not env.is_terminal() and n < args.moves:
        p = env.to_move()
        a = agent.select_action(env.observe(p))
        env.step(a)
        n += 1
    dt = time.time() - t
    print(f"{n} decisions in {dt:.2f}s = {dt / n * 1000:.1f} ms/decision ({args.sims} sims)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="sevenwa", description="7 Wonders: Architects self-play AI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("pipeline", help="run the full self-play training pipeline")
    s.add_argument("--config", default=None, help="JSON file with PipelineConfig fields (nested 'selfplay'/'train')")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override any config field, e.g. --set selfplay.c_puct=2.0 --set train.score_weight=0.5")
    s.add_argument("--run-dir", default="runs/default")
    s.add_argument("--generations", type=int, default=10)
    s.add_argument("--games", type=int, default=64, help="self-play games per generation")
    s.add_argument("--sims", type=int, default=160, help="MCTS simulations per move in self-play")
    s.add_argument("--temp-moves", type=int, default=20)
    s.add_argument("--root-mode", default="puct", choices=["puct", "gumbel"])
    s.add_argument("--value-mix", type=float, default=0.0, help="blend the search root value into the value target (0 = pure game outcome)")
    s.add_argument("--bootstrap-games", type=int, default=64)
    s.add_argument("--bootstrap-sims", type=int, default=64)
    s.add_argument("--bootstrap-mode", default="rollout", choices=["rollout", "imitation"])
    s.add_argument("--bootstrap-epochs", type=float, default=4.0)
    s.add_argument("--window", type=int, default=6)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--width", type=int, default=256)
    s.add_argument("--depth", type=int, default=4)
    s.add_argument("--trunk", default="resmlp", choices=["resmlp", "attention"])
    s.add_argument("--dropout", type=float, default=0.1)
    s.add_argument("--batch", type=int, default=256)
    s.add_argument("--epochs", type=float, default=1.0)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--weight-decay", type=float, default=1e-3)
    s.add_argument("--gate-games", type=int, default=60)
    s.add_argument("--gate-threshold", type=float, default=0.55)
    s.add_argument("--gate-sims", type=int, default=100)
    s.add_argument("--eval-games", type=int, default=20)
    s.add_argument("--eval-sims", type=int, default=100)
    s.add_argument("--eval-opponents", default="heuristic,rollout:100:1")
    s.add_argument("--eval-every", type=int, default=1, help="evaluate against the reference opponents every N generations")
    s.set_defaults(func=cmd_pipeline)

    s = sub.add_parser("arena", help="play two agents against each other")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s.add_argument("--games", type=int, default=20)
    s.add_argument("--sims", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--verbose", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_arena)

    s = sub.add_parser("selfplay", help="generate self-play data")
    s.add_argument("--checkpoint", default=None)
    s.add_argument("--games", type=int, default=16)
    s.add_argument("--sims", type=int, default=100)
    s.add_argument("--out-dir", default="runs/selfplay")
    s.add_argument("--generation", type=int, default=0)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_selfplay)

    s = sub.add_parser("train", help="train a network on saved shards")
    s.add_argument("--data", required=True, help="glob of .npz shards")
    s.add_argument("--init", default=None)
    s.add_argument("--out", required=True)
    s.add_argument("--width", type=int, default=256)
    s.add_argument("--depth", type=int, default=4)
    s.add_argument("--batch", type=int, default=256)
    s.add_argument("--epochs", type=float, default=2.0)
    s.add_argument("--lr", type=float, default=1e-3)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("play", help="play against an agent in the terminal")
    s.add_argument("--ai", default="heuristic")
    s.add_argument("--human", type=int, default=0, help="which seat you take (0 moves first)")
    s.add_argument("--sims", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_play)

    s = sub.add_parser("transcript", help="play one annotated game and write a markdown transcript")
    s.add_argument("--a", default="net:models/imitation_v3.pt:300")
    s.add_argument("--b", default="net:models/imitation_v3.pt:300")
    s.add_argument("--sims", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--wonders", default=None, help="comma-separated wonder ids for P0,P1 (default random)")
    s.add_argument("--out", default=None)
    s.set_defaults(func=cmd_transcript)

    s = sub.add_parser("bench", help="time an agent's decisions")
    s.add_argument("--agent", default="rollout:100:1")
    s.add_argument("--sims", type=int, default=100)
    s.add_argument("--moves", type=int, default=20)
    s.set_defaults(func=cmd_bench)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
