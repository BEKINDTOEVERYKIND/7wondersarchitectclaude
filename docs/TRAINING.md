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

## Playing and watching

```
sevenwa play --ai net:models/imitation_v3.pt:400 --human 0
sevenwa transcript --a net:models/imitation_v3.pt:300 --b net:models/imitation_v3.pt:300 --seed 7 --out docs/SAMPLE_GAME.md
```

`play` shows the belief state (both tableaus, deck tops, tokens) and the legal choices.
`transcript` writes an annotated markdown game: the table before every turn, each decision with
its context, and for search agents the root value and per-action visit counts and Q-values.

## What the development runs taught us (read before training)

The first reference run (`runs/v1`, 34 generations × 64 games, window 6, 1.5 epochs) exposed the
dominant failure mode of this game on a small compute budget: **replay-window overfitting**.
Self-play samples are ~60 per game and highly correlated, so a 6-generation window holds only
~380 games.  Diagnostics (all reproducible with the scripts in `docs/TRAINING.md` § below):

| Measurement | Value |
|---|---|
| Champion value MSE on its own training window | 0.04–0.2 |
| Champion value MSE on the *next*, unseen generation | 1.3–1.8 (worse than predicting 0) |
| Fresh network, all 2000 games, held-out generation: value MSE after 1 / 3 / 9 epochs | 0.92 / 1.15 / 1.42 |
| Held-out policy cross-entropy, any setting | ≈ 1.08 (≈ 3 legal actions) |
| `sign(score difference)` predicts the winner at turns 30+ / at turns 0–15 | 82 % / 34 % |

The last line shows that early positions are genuinely unpredictable (an early points lead is
even *negatively* correlated with winning, tempo matters more), so a value MSE near 0.8–0.9 is
the realistic floor with few games, and anything beyond one pass over fresh data is memorisation.
The pipeline therefore logs `champion loss on the fresh generation` every generation — watch it:
if the value term climbs above ~1.0 the network is overfitting and the search is being misled.

Recommended regime (what `runs/v2` uses):

```
sevenwa pipeline --run-dir runs/v2 --games 64 --sims 96 --root-mode gumbel \
    --window 30 --epochs 0.5 --lr 5e-4 --weight-decay 1e-3 --dropout 0.2 \
    --gate-games 30 --gate-threshold 0.5 --eval-opponents heuristic --eval-every 3
```

i.e. a window of ~2000 games, half a pass per generation (every sample is seen ~15 times over
its life in the window, spread over 30 different champions), dropout + weight decay, and a soft
gate.  With more cores, raise `--games` first (data diversity), then `--sims`.

## Imitation bootstrap (recommended start)

`scripts/imitation_bootstrap.py` (or `sevenwa pipeline --bootstrap-mode imitation`) generates
ε-greedy heuristic self-play at ~60 games/s per process and trains the network on the
heuristic's *soft* policy (`heuristic_prior`) plus the real outcomes.  20 000 games (1.2 M
samples, 2.5 min on 3 workers) give, on 1 000 held-out games:

| epochs | policy CE | policy acc | value MSE | train/held-out gap |
|---|---|---|---|---|
| 0.5 | 0.923 | 71.5 % | 0.790 | none |
| 2.0 | 0.871 | 77.7 % | 0.795 | none |

The value head generalises here (0.79 versus ≥ 0.9 from pure self-play data), because 20 000
games contain the outcome variety that 2 000 do not.  Self-play then starts from a network that
is already a competent player instead of from random play (`runs/v3`).

## Results of the reference runs (4-core CPU, single afternoon)

Arena results are wins-losses-draws for the first-named agent, seats alternated, with the
95 % Wilson interval in brackets; the heuristic is the reference (it beats random 97 %).

| Agent | vs heuristic | Note |
|---|---|---|
| `rollout:64:1` (64 heuristic playouts) | 5-7 (42 %) | classical search ≈ its playout policy |
| `rollout:200:1` | 6-6 (50 %) | |
| `net:v1:64` after 33 self-play generations from a 96-game rollout bootstrap | 4-16 … 8-12 (20–40 %) | value head over-fitted the replay window |
| `net:v1:300` | 6-14 (30 %) | more search cannot repair a misleading value |
| `hybrid:v1:100:0.5` | 3-9 (25 %) | |
| `netraw:v3` (imitation network, **no search**) | **55-45 (55 % [45–64])**, 100 games | already ≥ its teacher |
| `net:v3:100` (imitation network + 100-simulation MCTS) | **15-9 (62 % [43–79])** | strongest agent so far |
| `hybrid:v3:100:0.5` | 9-7 (56 %) | playout blending no longer needed |

| `net:v3:300` | **16-8 (67 % [47–82])** | |
| `net:v3:200` vs `rollout:200:1` | **10-2 (83 % [55–95])** | |

The shipped checkpoint `models/imitation_v3.pt` is this network (imitation bootstrap, 2 epochs);
`sevenwa play --ai net:models/imitation_v3.pt:300` plays it.

Continuing self-play from it (`runs/v3`, window 30, 0.5 epochs/generation, lr 3e-4) keeps the
fresh-generation value loss at 0.71–0.90 (vs 1.3–1.8 in the over-fitted v1 run) and policy
accuracy on fresh data at 56–61 %.  In the first generations the gate rejected most candidates at
~43 %: self-play fine-tuning first has to reconcile the imitation policy with the search's
targets, and with 30-game gates the promotion decision is noisy.  This is the point at which
compute matters: the pipeline is doing the right thing, and more games per generation
(`--games 256+` on a bigger machine) and more generations are what turn the imitation-level
network into a super-heuristic one.

### Reproducing

```
python scripts/imitation_bootstrap.py --run-dir runs/v3 --games 20000 --holdout-games 1000
mv runs/v3/data/gen0000_*.npz runs/v3/imitation/          # keep imitation data out of the replay window
sevenwa pipeline --run-dir runs/v3 --games 64 --sims 96 --root-mode gumbel --bootstrap-games 0 \
    --window 30 --epochs 0.5 --lr 3e-4 --weight-decay 1e-4 --dropout 0.1 --gate-threshold 0.5
sevenwa arena --a net:runs/v3/ckpt/champion.pt:300 --b heuristic --games 40
python scripts/learning_curve.py runs/v3/metrics.jsonl
```
