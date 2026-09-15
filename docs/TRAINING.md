# Training, evaluation and play

## Requirements

Python ≥ 3.10, `numpy`, `torch` (CPU build is enough), `pytest`.  Install with `pip install -e .`.

## Agents (spec strings)

| Spec | Meaning |
|---|---|
| `random` | uniform random legal action |
| `heuristic[:eps]` | hand-written expert policy (`sevenwa/agents/heuristic.py`), optional ε-greedy noise |
| `rollout:<sims>[:<playouts>[:random]]` | MCTS with heuristic (default) or random playouts, no network |
| `net:<ckpt>[:<sims>[:<c_puct>[:<value_T>]]]` | neural MCTS (PUCT) with the given checkpoint (default 200 simulations) |
| `netraw:<ckpt>` | the network's policy head alone, no search |
| `hybrid:<ckpt>[:<sims>[:<lam>[:<playouts>]]]` | neural MCTS with leaf values `(1-lam)·net + lam·heuristic playout` (defaults 100, 0.5, 1) |

The positional fields above can be followed (or interleaved, after the checkpoint path) by
`key=value` options; a token containing `=` is an option, every other token keeps its positional
meaning.  Every option defaults to the behaviour of the plain spec, so existing spec strings are
unchanged.  Example:

```
hybrid:models/imitation_v7.pt:300:0.5:pt=1.5:floor=0.05:root=gumbel:pick=q:batch=8:margin=0.5
```

| Option | Applies to | Meaning |
|---|---|---|
| `pt=<T>` | `net`, `netraw`, `hybrid` | policy temperature: priors = softmax(logits / T).  The imitation networks put > 0.95 on one move in a fifth of the decisions, which starves the other legal moves of root visits; `pt=1.5` flattens them (default 1.0) |
| `floor=<f>` | search agents | root prior floor: `p ← (1-f)·p + f·uniform(legal)` at the root only, so every legal move keeps a minimum exploration share (default 0) |
| `root=puct\|gumbel` | search agents | root procedure: PUCT visit counts (default) or Sequential Halving with Gumbel-Top-k (`MCTSAgent(root_mode=...)`) |
| `pick=visits\|q` | search agents | final move: most visits / argmax of the improved policy (default), or the best Q among the children with at least 25 % of the maximum visit count (`SearchResult.best_action_q`) |
| `batch=<n>` | search agents | leaves evaluated per evaluator call: `rollout` and `hybrid` default to 1 (one playout per leaf), `net` to 8; larger batches use virtual loss and a single network forward per batch |
| `margin=<w>` | `hybrid` | playout value = `(1-w)·result + w·tanh(margin/8)` with the final score margin from the mover's perspective — a lower-variance signal than the ±1 result (default 0) |
| `cpuct=<c>` | search agents | PUCT exploration constant (wins over the positional `<c_puct>` of `net`) |
| `fpu=<r>` | search agents | first-play-urgency reduction in Q units (default 0.25) |
| `reuse=chance\|all` | search agents | tree reuse between the agent's decisions only through resolved chance nodes (default) or also across the opponent's decisions (`MCTSConfig.reuse_across_opponent`) |

The network evaluator runs a traced + frozen TorchScript copy of the network (same outputs to
1e-5, about 25 % faster per call on CPU) and falls back to eager mode if tracing fails.

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
| Feature encoding | ~60 µs per state (581 features; 437 before the 2026-09-14 rules correction) |
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

### Self-play continuation from the imitation network (`runs/v3`, 31 generations)

Continuing self-play from `imitation_v3.pt` (64 games and 96 Gumbel simulations per generation,
window 30, 0.5 epochs, gate 30 games at 50 %) produced champions that beat the heuristic 14-6 and
15-5 at 64 simulations, but a 300-simulation head-to-head against the imitation network itself
came out 9-15: after ~2 000 self-play games the network had **not** improved on its starting
point.  Two causes are visible in the logs:

* the fresh-generation value loss drifted from 0.7–0.8 back up to ~1.0 as the window filled —
  the value head over-fits self-play data far faster than it learns from it at this data rate;
* a 30-game gate at 50 % promotes a coin flip: about half of the "promotions" were noise, so the
  champion random-walked instead of climbing.

Fixes now available in the pipeline: `--value-mix 0.5` (blend the search root value into the
value target: lower-variance targets), a stricter gate (`--gate-games 60 --gate-threshold 0.55`),
and simply more games per generation.  On this 4-core box the reliable strength gains came from
the imitation bootstrap, so `models/imitation_v3.pt` remains the shipped agent; a larger
imitation run (`runs/v4`: 40 000 games, 384×5 network) is the next rung.

### Value calibration, search parameters and the teacher (second session)

* **Calibration.** On 90 000 held-out imitation positions the value head is over-confident:
  predictions of ±0.5 correspond to outcomes of ±0.34, ±0.7 to ±0.5.  A logit temperature of
  1.4–1.5 (`net:<ckpt>:<sims>:<c_puct>:<T>`, implemented in `TorchEvaluator`) fixes the
  reliability curve (held-out MSE 0.812 → 0.798).  In 24-game matches at 300 simulations it made
  no measurable difference (17-7 vs the heuristic either way); a PUCT constant of 3.0 was worse
  (13-11).  The imitation value head's real limitation is *information*, not calibration: the
  outcomes of ε-greedy heuristic play are close to coin flips for most of the game (correlation
  0.45 with the outcome), which is also why 600 simulations lost to 100 with that network.
* **Larger imitation network** (`runs/v4`: 40 000 games, 384×5): held-out policy accuracy 84 %
  (v3: 78 %), same value floor; playing strength unchanged (11-13 vs v3, 17-7 vs the heuristic).
  Imitation has saturated at the teacher's level plus search.
* **Tuning the teacher.** `scripts/heuristic_ab.py` plays parameter variants against the default
  heuristic on identical deals from both seats (1 000 paired games in ~9 s).  Raising the value of
  extra-card effects (4 → 6), the tempo bonus (3 → 5), coin flexibility (0.4 → 0.8), the
  extra-card-token scale (0.6 → 0.9) and shortening the military look-ahead (3 → 2) wins
  56.1 % [54.3–57.8] over 3 000 paired games, and the tuned heuristic also does better against
  the search agent (12-18 vs 10-20 at 100 simulations).  Raising the Cat's value *hurt* in
  self-play; higher blue-card weight and denial weights were neutral.
* **Expert iteration** (`scripts/expert_iteration.py`): a fixed teacher generates thousands of
  Gumbel-search games, the network is fine-tuned on them mixed with imitation data, and the
  candidate must beat the teacher in a 40-game match at 300 simulations to be accepted — one
  decisive match instead of the pipeline's noisy per-generation gate (now 60 games at 55 %).

### Third imitation run with the tuned teacher (`runs/v5`)

40 000 games of the A/B-tuned heuristic, 384×5 network, 3 epochs: held-out policy accuracy 86 %
(v4: 84 %).  Matches at 300 simulations: **24-16 (60 %) against v4** and **21-3 (87.5 %, +6.6
mean margin) against the tuned heuristic** — the strongest agent so far and the shipped model
`models/imitation_v5.pt`.  A stronger teacher transferred directly into a stronger imitation
network; `scripts/heuristic_tune.py` automates the teacher tuning (coordinate descent with paired
A/B confirmation) for the next rounds.

### Expert iteration round 2 (`runs/ei2`): the first accepted improvement over imitation

Teacher `imitation_v5`, 3 000 Gumbel-search games (96 simulations, 176 000 samples, 108 min on
3 workers), fine-tuned for 2 epochs on 70 % search data / 30 % imitation data with
`--value-mix 0.3`: the candidate's fit to the search targets rose from 65.5 % to 72.4 % held-out
accuracy, and in the decisive 40-game match at 300 simulations it beat its teacher **25-15
(62.5 %, +4.4 mean margin)**.  It is shipped as `models/expert_v6.pt` (16-8 vs the tuned
heuristic in a 24-game check).  Round 1 (teacher v4, 2 400 games) had drawn 20-20: the
difference is the stronger teacher — search-play data is only richer than imitation data once
the searcher clearly outplays the imitated policy.  Round 3 (`runs/ei3`, teacher v6, the
round-2 shards accumulated) is the next step and takes ~2.5 h per round on this box.

### Play settings for the shipped model

`imitation_v5` (and its descendant `expert_v6`) benefit from deeper search, unlike the first
imitation network: at 600 simulations v5 beats itself at 150 simulations 15-9 (62.5 %).  For
the strongest play use as many simulations as your time budget allows, e.g.
`net:models/expert_v6.pt:600` (about 0.5 s per decision on one CPU core); the optional value
temperature (`:1.75:1.45`) is neutral.

### Rules correction of 2026-09-14 and the retraining that followed

The exact per-deck card distribution and the seven Wonder boards (per-stage VP, effect stages
and the printed construction prerequisites) were supplied from the physical components and
replaced the assumed data.  The engine's stage model changed from a counter to a bitmask with a
stage-choice decision (`D_STAGE`, 5 new actions), and the feature encoder now describes every
stage (built / available / affordable + static description), so `FEATURE_SIZE` went from 437 to
581 and `Actions.NUM` from 44 to 49.  All measurements above were made under the old data; the
old checkpoints cannot be loaded by the new code and were removed (git history up to `bb15851`
keeps them).  The heuristic teacher was re-validated on the new rules with the paired A/B
harness (see the entries below) and the network was retrained from scratch: imitation
bootstrap, then expert iteration.

### Teacher check under the corrected rules (2026-09-14)

Paired A/B (3 000 deals, both seats, `scripts/heuristic_ab.py --seed 11`) of two knowledge
knobs suggested by the user's strategic answers — "horn military cards are worthless early" and
"early green cards are strategically valuable" — against the tuned defaults:

| variant | score | 95 % CI | mean margin |
|---|---|---|---|
| `early_horn_discount=0.5` | 50.4 % | 49.1–51.6 | −0.08 |
| `early_horn_discount=1.0` | 50.9 % | 49.7–52.2 | −0.11 |
| `green_early_bonus=1.0` | 50.5 % | 49.2–51.8 | +0.14 |
| `green_early_bonus=2.0` | 50.1 % | 48.8–51.4 | +0.04 |
| `green_early_bonus=3.0` | 49.1 % | 47.9–50.4 | −0.03 |
| both (0.5 / 1.0) | 50.2 % | 49.0–51.5 | −0.02 |

All neutral: the tuned model already encodes both facts.  At the start of a Rhodes game the
heuristic's card values (VP-like units) are: any resource 7.5, coin 8.3, **green 8.5**, civ3
3.0, civ2cat 3.8, hornless shield 5.9, **horn cards 2.7** — greens are the most valuable pick
and horn cards the least, because the green value is the potential of the best available
Progress token (a token such as Architecture is worth ~24 at that point) while a horn card's
value is only the battle-probability-weighted military swing.  The knobs stay in
`HeuristicParams` at 0 for future experiments.

A broader re-tune sweep under the corrected rules (2 000 deals per variant, `--seed 23`, one
parameter moved at a time from the tuned defaults) found nothing better either — the best
variants (`perm_shield_bonus=0.2`, `res_discount=0.95`, `tempo_bonus=7`, `mil_lookahead=1`)
score 51.0–51.1 % with confidence intervals containing 50 %, and the clear losers are the same
as before (`extra_card_value=4` 45.3 %, `sci_token_factor=0.7` 47.9 %, `tok_extra_scale=0.6`
47.9 %, `tempo_bonus=3` 48.0 %).  The defaults tuned on the assumed data therefore carry over
unchanged.

### Imitation bootstrap on the corrected rules (`runs/v7`, shipped as `models/imitation_v7.pt`)

40 000 ε-greedy heuristic games (2.42 M samples, 5.3 min on 4 workers), 384×5 network, 3 epochs
(30 min): held-out policy accuracy **85.2 %**, value MSE 0.81, score-margin MSE 0.27.  By decision
kind the raw policy agrees with the teacher on 86 % of picks, 89 % of token choices, 100 % of the
(rare) stage choices, 96 % of Halicarnassus keeps and 58 % of payment steps.

Matches against the tuned heuristic (40 or 20 games, both seats):

| agent | score vs heuristic | note |
|---|---|---|
| `netraw:imitation_v7` (no search) | 45.0 % (45-55, n = 100) | the usual imitation deficit |
| `net:imitation_v7:300` | 52.5 % (21-19) | far below the 87.5 % the old-rules imitation net reached |
| `net:imitation_v7:300` vs `netraw` | 65.0 % (13-7) | search helps, but not much |
| `rollout:200:1` (heuristic playouts) | 75.0 % (15-5) | search now clearly beats the greedy teacher (it only drew level under the old rules) |
| **`hybrid:imitation_v7:300:0.5`** | **80.0 % (16-4, +9.6 margin)** | net priors, values blended 50/50 with playouts |

The diagnosis is the value head: its sign agrees with the final result on only 65 % of
early/mid-game positions of heuristic games (73 % late), so plain net-MCTS cannot exploit its
own (good) priors, while the same priors with playout-grounded values win 80 %.  The corrected
rules (stage graphs, more even card counts) make the game harder to value from ε-greedy
outcomes than the old linear Wonders did.  **`hybrid:models/<checkpoint>:300:0.5` is the strongest play setting** (about 1 s per decision);
see the value-head section below for the shipped checkpoints.

Consequence for training: `SelfPlayConfig.evaluator = "hybrid"` (and `scripts/expert_iteration.py
--evaluator hybrid --hybrid-lambda 0.5`) lets the expert-iteration teacher search with
playout-blended values, so the search-play outcomes and root values that train the candidate's
value head come from 80 %-strength play instead of 52 %-strength play.

### Value-head experiments on the corrected rules (`runs/relabel*`)

Because the hybrid evaluator (playout-grounded values) wins 80 % where the plain network wins
52.5 %, the first attempt to close the gap was to retrain the value head on *playout-averaged*
targets: `scripts/playout_relabel.py` samples every 3rd decision of ε-greedy heuristic games and
labels it with the mean result / margin of 8 heuristic playouts from the player's belief state
(119 000 positions from 6 000 games in 26 min on 4 workers).  Fine-tuning `imitation_v7` on them
for 2 epochs (+15 % raw imitation samples) lowered the value MSE against playout-mean targets
from 0.158 to 0.133 (score-margin MSE 0.053 → 0.046, policy unchanged), but the play did not
follow: the candidate scored 21-19 against its teacher and 22-18 (55 %) against the heuristic at
300 simulations — the same level as `imitation_v7`.  A K = 8 playout mean has a noise variance of
roughly 0.1 on uncertain positions, so 0.133 is close to that floor: on the imitation
distribution the network's value is already about as accurate as a playout mean.  The
remaining difference to the hybrid is therefore off-distribution accuracy — the search visits
lines the heuristic never plays, where a learned value extrapolates and a playout does not.  The
follow-up round labelled every 2nd decision with 16 playouts (45 000 more positions before the
machine was recycled).  Training `imitation_v7` for 4 epochs on all 164 000 playout-labelled
positions (+10 % raw imitation) brought the value MSE on the playout targets from 0.147 to
0.118 and the score-margin MSE from 0.050 to 0.041, policy unchanged (86 %).  This candidate
is shipped as **`models/value_v8.pt`**: at 300 simulations it scores **25-15 (62.5 %, +3.5
margin) against the heuristic** (`imitation_v7`: 21-19), while its 40-game head-to-head against
`imitation_v7` was 19-21 — the two are within the noise of 40-game matches, but the trend over
the three value-head variants (52.5 % → 55 % → 62.5 % against the heuristic as the value MSE
falls) is consistent.  With the hybrid evaluator, however, `value_v8` scored only 11-9 (55 %)
against the heuristic where `imitation_v7` scored 16-4 (80 %) — 20-game samples whose
confidence intervals overlap almost entirely, so `hybrid:models/imitation_v7.pt:300:0.5` stays
the recommended setting on the measured evidence.  The playout-blended hybrid remains the
strongest configuration; the next
lever is search-generated (off-policy) positions for the value head — `scripts/selfplay_chunks.py`
with the hybrid evaluator produces them, at about 5 games per minute on this box.

### Are the heuristic's card values "points"?  (user question, 2026-09-15)

No — they are *ranking scores* in VP-like units, and they are deliberately inflated relative
to the points a card ends up scoring.  A player finishes with about 42 points from about 30
cards (1.4 VP per card), whereas the heuristic values an early resource at 7.5 and an early
green at 8.5.  The inflation is the value of *tempo*: completing a stage is worth its VP plus 5
(`tempo_bonus`), and a green card is worth its share of the best obtainable token, which is
itself valued by the extra cards it yields over the rest of the game.  Compressing the values
towards points loses in paired self-play (3 000 deals, both seats, vs the tuned defaults):

| variant | idea | score |
|---|---|---|
| `res_unit=0.35` | resource ≈ 2.6 < a 3-VP card | 41.4 % |
| `res_unit=0.5` | resource ≈ 3.7 | 45.2 % |
| `w_blue=2.5` / `1.8` | blue cards worth 2.5× / 1.8× their printed VP | 45.6 % / 47.3 % |
| `perm_shield_bonus=3.0` / `1.5` | hornless shield ≈ 8.5 / 7 (like a green) | 45.1 % / 47.4 % |
| `res_unit=0.4 perm_shield_bonus=2.0` (± `w_blue=1.5`) | greens and shields on top, resources below a 3-VP card | 39.5 % / 39.4 % |

Note the margins: several of the losing variants score *more points on average* (positive mean
margin) while winning fewer games — they collect points instead of racing the Wonder, and the
player who completes five stages first ends the game while ahead.  That race is what the
inflated resource and green values encode, so "points per card" understates them.  The
ordering the user proposed (greens and hornless shields best at the start) is already the
model's ordering at game start (green 8.5 > coin 8.3 > resource 7.5 > hornless shield 5.9 >
cat card 3.8 > 3-VP card 3.0 > horn cards 2.7), except that resources sit above blue cards —
and every attempt to lower them below a 3-VP card lost.

### Round 3 (2026-09-15): play-time settings and search-level knowledge tests

Hybrid blend weight and search size against the tuned heuristic (20 games each, so ± 20 points):

| agent | score |
|---|---|
| `hybrid:imitation_v7:300:0.5` | 80 % (16-4) — earlier measurement |
| `hybrid:imitation_v7:300:0.75` | 65 % (13-7) |
| `hybrid:imitation_v7:300:1.0` (pure playout values, net priors) | 70 % (14-6) |
| `hybrid:imitation_v7:300:0.5:2` (two playouts per leaf) | 80 % (16-4, +6.5) |
| **`hybrid:imitation_v7:600:0.5`** | **90 % (18-2, +6.8)** — about 2 s per decision |

Doubling the search is the one change that clearly helps; the blend weight and a second
playout per leaf do not.  `hybrid:models/imitation_v7.pt:600:0.5` is the recommended setting
and the one the viewer's games were recorded with.

`scripts/search_ab.py` tests a change of *knowledge* at search level: rollout-MCTS (heuristic
prior + heuristic playouts, 200 simulations) built from a variant `HeuristicParams` against the
same search built from the defaults, on paired deals.  The owner's two rules of thumb and the
point-calibration idea were tested this way as well as in greedy self-play:

| variant | greedy A/B (6 000 games) | search A/B (20–40 games) |
|---|---|---|
| `green_early_bonus=3` (greens worth +3 early) | 49.1 % | 45 % (18/40) |
| `early_horn_discount=1.0` (horn cards worthless early) | 50.9 % | 45 % (9/20) |
| `res_unit=0.5` (resources below a 3-VP card) | 45.2 % | 50 % (10/20) |
| `deny_reveal` 0 / 0.1 / 0.2 / 0.5, `deny` 0.3 | 49–50 % | — |

None helps.  The heuristic (and therefore the search built on it) draws blind from the central
deck in 40 % of its main picks even when a side card is visible, passing up visible early greens
40 % of the time; the reason is not the denial terms (neutral above) but the valuation: the
expected value of a random central card (resources 7.5, coins 8.3 …) exceeds a single visible
green unless a valuable token is face-up.  Whether that habit is right cannot be settled by
self-play of the same policy — every evaluation here (playouts, imitation value) shares the
heuristic's style — which is the strongest argument for the network-only route (§ next).

### Value probe and leaf relabelling

Two tools for the off-distribution value question (the search visits lines the heuristic never
plays, where a learned value extrapolates and a playout does not):

* `scripts/value_probe.py --nets a.pt,b.pt --epsilons 0.1,0.5 --positions 1000 --playouts 8 --every 4`
  samples decision positions from ε-greedy heuristic games at each ε (0.1 ≈ the imitation
  distribution, 0.5 = positions the heuristic would never reach by itself), labels each with the
  mean result / margin of K heuristic playouts from the mover's belief state, and prints per
  network the value MSE, correlation, mean bias and sign agreement — overall, by turn bucket, by
  `min(cards_needed)` (the heuristic's count of cards still needed to finish a Wonder) and by
  decision kind; `z_var` is the MSE of predicting 0.  About 4 ms per playout, so a 1 000-position
  probe at two ε takes ~30 s.  On 300 positions per ε (seed 0):

  | network | ε | value MSE | corr | bias | sign agreement | margin MSE / corr |
  |---|---|---|---|---|---|---|
  | `imitation_v7` | 0.1 | 0.174 | 0.71 | −0.13 | 75.7 % | 22.7 / 0.79 |
  | `imitation_v7` | 0.5 | 0.150 | 0.77 | −0.08 | 81.9 % | 22.5 / 0.83 |
  | `value_v8` | 0.1 | 0.132 | 0.75 | −0.04 | 79.4 % | 18.1 / 0.82 |
  | `value_v8` | 0.5 | 0.139 | 0.78 | −0.01 | 81.2 % | 21.8 / 0.83 |

  Neither network degrades at ε = 0.5 (those positions are more decided: `z_var` 0.35 vs 0.30),
  `value_v8` removes most of `imitation_v7`'s pessimistic bias, and the weak spots are the same
  for both: turns 0–20 and positions with 10+ cards still needed (correlation 0.3–0.5, sign
  agreement 59–73 %) against 0.8+ / ~90 % in the last third of the game.
* `scripts/leaf_relabel.py --net a.pt --out-dir runs/leaf1/data --games 400 --sims 96 --workers 4
  --playouts 8 --leaves-per-decision 8 --every 2 --jobs 16` produces value training data from the
  **search's own tree positions**: the network-only MCTS agent (`net:<ckpt>:<sims>`) plays half its
  games against itself and half against `heuristic:0.1` (seat alternating); after every
  `--every`-th search the tree below the root is walked (`Node.state` holds the belief state),
  the visited decision nodes with ≥ `--min-visits` visits are subsampled to
  `--leaves-per-decision` with weight `(1 + depth) / √N` (deeper, rarer nodes preferred), and
  each is labelled from its own mover's perspective with the mean of `--playouts` heuristic
  playouts, with the heuristic prior as the policy target.  Shards use the self-play format,
  one per job, and finished jobs are skipped on re-run; train with
  `scripts/expert_iteration.py --skip-generation --extra-search-glob '<out-dir>/*.npz'`.  At
  96 simulations the harvested nodes sit 1–2 decisions below the root (mostly the opponent's
  replies).  Smoke test (`--games 2 --sims 32 --leaves-per-decision 4 --playouts 2`): 71
  searches, 123 labelled positions in 3 s.

### Reproducing

```
python scripts/imitation_bootstrap.py --run-dir runs/v3 --games 20000 --holdout-games 1000
mv runs/v3/data/gen0000_*.npz runs/v3/imitation/          # keep imitation data out of the replay window
sevenwa pipeline --run-dir runs/v3 --games 64 --sims 96 --root-mode gumbel --bootstrap-games 0 \
    --window 30 --epochs 0.5 --lr 3e-4 --weight-decay 1e-4 --dropout 0.1 --gate-threshold 0.5
sevenwa arena --a net:runs/v3/ckpt/champion.pt:300 --b heuristic --games 40
python scripts/learning_curve.py runs/v3/metrics.jsonl
```
