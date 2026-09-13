"""AlphaZero-style training pipeline: bootstrap -> (self-play -> train -> gate -> evaluate)*.

All artefacts go to ``runs/<name>/``:
  data/genXXXX_wY.npz   self-play shards
  ckpt/genXXXX.pt       candidate after training on generation XXXX
  ckpt/champion.pt      current best network (used for self-play)
  metrics.jsonl         one JSON line per generation
"""
from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from ..agents.factory import make_agent
from ..engine.actions import Actions
from ..engine.env import make_env
from ..nn.features import ENTITY_SLICES, FEATURE_SIZE
from ..nn.model import NetConfig, PolicyValueNet
from ..search.mcts import MCTSConfig
from .arena import play_match
from .replay import ReplayBuffer
from .selfplay import SelfPlayConfig, run_imitation, run_selfplay
from .trainer import TrainConfig, Trainer


@dataclass
class PipelineConfig:
    run_dir: str = "runs/default"
    generations: int = 10
    games_per_generation: int = 64
    bootstrap_games: int = 64          # games before the first network exists (0 to skip)
    bootstrap_mode: str = "rollout"    # "rollout": rollout-MCTS games; "imitation": fast ε-greedy heuristic games
    bootstrap_simulations: int = 64
    bootstrap_epochs: float = 4.0
    bootstrap_epsilon: float = 0.1
    window_generations: int = 6        # replay window
    num_workers: int = 4
    seed: int = 0
    net_width: int = 256
    net_depth: int = 4
    net_trunk: str = "resmlp"
    net_dropout: float = 0.1
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    gate_games: int = 40
    gate_threshold: float = 0.55
    gate_simulations: int = 100
    eval_games: int = 20
    eval_opponents: List[str] = field(default_factory=lambda: ["heuristic", "rollout:100:1"])
    eval_simulations: int = 100
    eval_every: int = 1


def config_from_dict(d: Dict) -> PipelineConfig:
    """Build a PipelineConfig from a (possibly partial, nested) dict such as a parsed JSON file."""
    d = dict(d)
    sp = SelfPlayConfig(**d.pop("selfplay", {}))
    tr = TrainConfig(**d.pop("train", {}))
    return PipelineConfig(selfplay=sp, train=tr, **d)


def apply_overrides(cfg: PipelineConfig, overrides: List[str]) -> PipelineConfig:
    """Apply ``section.field=value`` / ``field=value`` overrides (values parsed as JSON when possible)."""
    for ov in overrides:
        key, _, raw = ov.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        target = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            target = getattr(target, p)
        if not hasattr(target, parts[-1]):
            raise KeyError(f"unknown config field {key}")
        setattr(target, parts[-1], value)
    return cfg


def _log_factory(run_dir: str):
    os.makedirs(run_dir, exist_ok=True)
    fh = open(os.path.join(run_dir, "log.txt"), "a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()
    return log


def new_network(cfg: PipelineConfig) -> PolicyValueNet:
    ncfg = NetConfig(input_dim=FEATURE_SIZE, num_actions=Actions.NUM, trunk=cfg.net_trunk, width=cfg.net_width,
                     depth=cfg.net_depth, dropout=cfg.net_dropout, entity_slices=list(ENTITY_SLICES))
    return PolicyValueNet(ncfg)


def fresh_data_loss(net: PolicyValueNet, shard_glob: str) -> Dict[str, float]:
    """Loss of ``net`` on data it has never trained on (the newest generation's shards) — the
    generalisation diagnostic that exposes replay-window overfitting."""
    from ..nn.model import compute_loss
    buf = ReplayBuffer(FEATURE_SIZE, Actions.NUM)
    if buf.load_shards(shard_glob) == 0:
        return {}
    b = buf.sample(min(len(buf), 8192), np.random.default_rng(0))
    net.eval()
    with torch.no_grad():
        _, parts = compute_loss(net, torch.from_numpy(b.feats), torch.from_numpy(b.mask), torch.from_numpy(b.pi),
                                torch.from_numpy(b.z), torch.from_numpy(b.margin))
    return {k: float(v) for k, v in parts.items()}


def evaluate_agents(spec_a: str, spec_b: str, games: int, seed: int, log, mcts_sims: int = 100) -> Dict:
    if games <= 0:
        return {"a": spec_a, "b": spec_b, "score_a": 0.5, "wins": [0.0, 0.0], "draws": 0, "mean_margin": 0.0, "games": 0}
    agent_a = make_agent(spec_a, seed=seed, mcts_config=MCTSConfig(num_simulations=mcts_sims))
    agent_b = make_agent(spec_b, seed=seed + 1, mcts_config=MCTSConfig(num_simulations=mcts_sims))
    res = play_match(make_env, (agent_a, agent_b), games, seed=seed)
    log("  " + res.summary((agent_a.name, agent_b.name)))
    return {"a": spec_a, "b": spec_b, "score_a": res.score(0), "wins": res.wins, "draws": res.draws,
            "mean_margin": float(np.mean(res.score_diffs)) if res.score_diffs else 0.0, "games": games}


def run_pipeline(cfg: PipelineConfig) -> None:
    log = _log_factory(cfg.run_dir)
    data_dir = os.path.join(cfg.run_dir, "data")
    ckpt_dir = os.path.join(cfg.run_dir, "ckpt")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(cfg.run_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    rng = np.random.default_rng(cfg.seed)
    champion_path = os.path.join(ckpt_dir, "champion.pt")
    metrics_path = os.path.join(cfg.run_dir, "metrics.jsonl")

    def record(m: Dict) -> None:
        with open(metrics_path, "a") as f:
            f.write(json.dumps(m) + "\n")

    net = new_network(cfg)
    log(f"network: {cfg.net_trunk} width {cfg.net_width} depth {cfg.net_depth}, {net.num_parameters():,} parameters; "
        f"features {FEATURE_SIZE}, actions {Actions.NUM}")
    buffer = ReplayBuffer(FEATURE_SIZE, Actions.NUM)
    start_gen = 0

    # ---- resume -----------------------------------------------------------
    if os.path.exists(champion_path):
        net = PolicyValueNet.load(champion_path)
        existing = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))
        if existing:
            start_gen = max(int(f[3:7]) for f in existing) + 1
        buffer.load_shards(os.path.join(data_dir, "gen*.npz"), min_generation=max(0, start_gen - cfg.window_generations))
        log(f"resumed champion from {champion_path}; next generation {start_gen}; buffer {len(buffer)} samples")
    else:
        # ---- bootstrap from rollout-MCTS games ------------------------------
        if cfg.bootstrap_games > 0:
            if cfg.bootstrap_mode == "imitation":
                log(f"bootstrap: {cfg.bootstrap_games} ε-greedy heuristic games (imitation targets)")
                run_imitation(cfg.bootstrap_games, data_dir, 0, cfg.num_workers, seed=cfg.seed, epsilon=cfg.bootstrap_epsilon, log=log)
            else:
                log(f"bootstrap: {cfg.bootstrap_games} rollout-MCTS games ({cfg.bootstrap_simulations} sims, heuristic playouts)")
                bcfg = SelfPlayConfig(**{**asdict(cfg.selfplay), "evaluator": "rollout", "num_simulations": cfg.bootstrap_simulations,
                                         "batch_size": 1})
                run_selfplay(bcfg, cfg.bootstrap_games, None, data_dir, 0, cfg.num_workers, seed=cfg.seed, log=log)
            buffer.load_shards(os.path.join(data_dir, "gen0000_*.npz"))
            log(f"bootstrap buffer: {buffer.stats()}")
            trainer = Trainer(net, TrainConfig(**{**asdict(cfg.train), "epochs": cfg.bootstrap_epochs}))
            stats = trainer.train(buffer, rng, log=log)
            record({"generation": 0, "phase": "bootstrap", "train": stats, "buffer": buffer.stats()})
            start_gen = 1
        net.save(champion_path, extra={"generation": start_gen - 1})
        log(f"champion initialised -> {champion_path}")

    # ---- main loop --------------------------------------------------------
    for gen in range(start_gen, start_gen + cfg.generations):
        t_gen = time.time()
        log(f"=== generation {gen} ===")
        run_selfplay(cfg.selfplay, cfg.games_per_generation, champion_path, data_dir, gen, cfg.num_workers,
                     seed=cfg.seed + gen * 100_000, log=log)
        buffer = ReplayBuffer(FEATURE_SIZE, Actions.NUM)
        buffer.load_shards(os.path.join(data_dir, "gen*.npz"), min_generation=max(0, gen - cfg.window_generations + 1))
        log(f"replay buffer: {buffer.stats()}")

        candidate = PolicyValueNet.load(champion_path)
        fresh = fresh_data_loss(candidate, os.path.join(data_dir, f"gen{gen:04d}_*.npz"))
        if fresh:
            log(f"champion loss on the fresh generation (unseen data): policy {fresh['policy']:.3f} value {fresh['value']:.3f} "
                f"score {fresh['score']:.3f} acc {fresh['acc']:.3f}")
        trainer = Trainer(candidate, cfg.train)
        train_stats = trainer.train(buffer, rng, log=log)
        cand_path = os.path.join(ckpt_dir, f"gen{gen:04d}.pt")
        candidate.save(cand_path, extra={"generation": gen})

        # ---- gate (gate_games <= 0 disables gating: always promote) ---------
        gate = evaluate_agents(f"net:{cand_path}:{cfg.gate_simulations}", f"net:{champion_path}:{cfg.gate_simulations}",
                               cfg.gate_games, seed=cfg.seed + gen * 7919, log=log, mcts_sims=cfg.gate_simulations)
        promoted = cfg.gate_games <= 0 or gate["score_a"] >= cfg.gate_threshold
        if promoted:
            shutil.copyfile(cand_path, champion_path)
            log(f"candidate PROMOTED ({gate['score_a'] * 100:.1f}% >= {cfg.gate_threshold * 100:.0f}%)")
        else:
            log(f"candidate rejected ({gate['score_a'] * 100:.1f}%)")

        # ---- evaluation against fixed references ------------------------------
        evals = []
        if cfg.eval_games > 0 and cfg.eval_every > 0 and gen % cfg.eval_every == 0:
            for opp in cfg.eval_opponents:
                evals.append(evaluate_agents(f"net:{champion_path}:{cfg.eval_simulations}", opp, cfg.eval_games,
                                             seed=cfg.seed + gen * 104_729, log=log, mcts_sims=cfg.eval_simulations))
        record({"generation": gen, "train": train_stats, "fresh": fresh, "gate": gate, "promoted": promoted,
                "evals": evals, "buffer": buffer.stats(), "seconds": time.time() - t_gen})
        log(f"generation {gen} done in {time.time() - t_gen:.0f}s")
