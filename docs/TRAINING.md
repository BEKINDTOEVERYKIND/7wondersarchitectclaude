# Training, evaluation and play

## Requirements

Python ≥ 3.10, `numpy`, `torch` (CPU build is enough), `pytest`.  Install with `pip install -e .`.

## Agents (spec strings)

| Spec | Meaning |
|---|---|
| `random` | uniform random legal action |
| `heuristic[:eps]` | hand-written expert policy (`sevenwa/agents/heuristic.py`), optional ε-greedy noise |
| `rollout:<sims>[:<playouts>[:random]]` | MCTS with heuristic (default) or random playouts, no network |
| `net:<ckpt>[:<sims>]` | neural MCTS (PUCT) with the given checkpoint |
| `netraw:<ckpt>` | the network's policy head alone, no search |

## Pipeline

```
sevenwa pipeline --run-dir runs/v1 --generations 20 --games 128 --sims 160 --workers 4 \
                 --bootstrap-games 128 --bootstrap-sims 64 --width 256 --depth 4
```

Stages per run (`sevenwa/train/pipeline.py`):

1. **Bootstrap** (`--bootstrap-games`): rollout-MCTS games with heuristic playouts are
   generated and the network is pre-trained on them (policy = visit distribution, value =
   result, score head = final margin).  This skips the random-network phase entirely.
2. **Self-play**: `--games` games per generation with the current champion, `--sims`
   simulations per decision, Dirichlet noise at the root, temperature 1 for the first
   `--temp-moves` decisions.  `--root-mode gumbel` switches to Sequential Halving with
   Gumbel-Top-k and completed-Q policy targets (recommended for `--sims` ≤ 64).
3. **Training**: candidate = champion trained for `--epochs` passes over the last
   `--window` generations (AdamW, cosine LR, loss = CE(policy) + MSE(value) + 0.25·MSE(margin)).
4. **Gating**: candidate vs champion, `--gate-games` games with alternating seats; promoted if
   it scores ≥ `--gate-threshold` (default 55 %).
5. **Evaluation**: champion vs `--eval-opponents` (default `heuristic,rollout:100:1`),
   logged to `runs/<name>/metrics.jsonl`.

The run is resumable: re-running the same command continues from `ckpt/champion.pt` and the
existing data shards.

### Measured speeds (4-core CPU box, 1 thread per worker)

| Component | Speed |
|---|---|
| Engine, random play | ~33 µs per ply; 50 games in 0.1 s |
| Heuristic policy | ~70 µs per decision; heuristic self-play ~150 games/s |
| Heuristic playout (to game end) | ~4 ms |
| Feature encoding | ~48 µs per state (437 features) |
| Network forward (256×4 residual MLP, 1.4 M params) | ~1.3 ms per batch of 8 |
| Neural MCTS, 96 simulations | ~50 ms per decision (≈ 3.5 s per game) |
| Rollout-MCTS, 64 heuristic playouts | ~0.3 s per decision |
| Self-play throughput, 3 workers, 96 sims (Gumbel) | 70–80 games/min |

### Practical settings

| Hardware | Suggested |
|---|---|
| 4 CPU cores (this development box) | `--games 64 --sims 96 --root-mode gumbel --width 256 --depth 4 --workers 3` ≈ 3–4 min/generation (evaluate against `rollout:*` only every few generations: those games are slow) |
| 16+ cores | `--workers 14 --games 256 --sims 200` |
| GPU available | `--trunk attention --width 512 --depth 6` and increase `--sims` to 400–800 |

Strength grows with total self-play games × simulations.  The rollout-MCTS reference
(`rollout:200:1`) is a strong classical player because its playouts use the expert heuristic;
beating it consistently at 200 simulations is the milestone for a "strong" network.

## Evaluation and Elo ladder

```
sevenwa arena --a net:runs/v1/ckpt/champion.pt:200 --b rollout:200:1 --games 50
python -c "from sevenwa.train.ladder import run_ladder; run_ladder(['random','heuristic','rollout:100:1','net:runs/v1/ckpt/champion.pt:100'], games_per_pair=30)"
```

Arena results are reported as wins-losses-draws with a 95 % Wilson interval; the ladder fits
Elo (Bradley–Terry) with the first spec as anchor.

## Playing

```
sevenwa play --ai net:runs/v1/ckpt/champion.pt:400 --human 0
```

The terminal shows the belief state (both tableaus, deck tops, tokens) and the legal choices.

## Results of the reference run

_(filled in below from the development run on this 4-core, CPU-only machine)_
