# 7 Wonders: Architects — self-play AI (2-player)

An AlphaZero-style agent for the 2-player game of *7 Wonders: Architects*: an exact
belief-state rules engine with chance nodes, a chance-aware PUCT/Gumbel Monte-Carlo tree
search with batched neural evaluation, a policy/value/score network, and a self-play
training pipeline with gating and an Elo ladder against classical baselines.

* `docs/DESIGN.md`   — why the system is built this way (game analysis, architecture).
* `docs/RULES.md`    — the exact rules the engine implements, with provenance and confidence.
* `docs/TRAINING.md` — how to train, evaluate and play; results of the reference run.

## Quick start

```bash
pip install -e .            # numpy + torch (CPU is fine)
python -m pytest -q         # engine / search / network tests

# play against the heuristic bot (you are player 0)
sevenwa play --ai heuristic
# play against a rollout-MCTS bot with 400 playouts per move
sevenwa play --ai rollout:400:1

# train: bootstrap from rollout-MCTS games, then self-play generations with gating
sevenwa pipeline --run-dir runs/v1 --generations 20 --games 128 --sims 160 --workers 4

# evaluate a checkpoint against baselines
sevenwa arena --a net:runs/v1/ckpt/champion.pt:200 --b heuristic --games 40
sevenwa arena --a net:runs/v1/ckpt/champion.pt:200 --b rollout:200:1 --games 40
# strongest agent while the value network is still young: network priors + playout-blended values
sevenwa arena --a hybrid:runs/v1/ckpt/champion.pt:200:0.5 --b heuristic --games 40
```

## Agent ladder (measured during development, see `docs/TRAINING.md`)

The rules data was corrected on 2026-09-14 from the physical components (exact per-deck card
counts, the seven Wonder boards with their stage dependency graphs).  Every network trained
before that date (`imitation_v3`, `imitation_v5`, `expert_v6`; git history up to commit
`bb15851`) used assumed data and an incompatible feature/action layout and was removed.
Under the corrected rules (scores against the tuned heuristic, see `docs/TRAINING.md`):
`random` ≪ `heuristic` (A/B-tuned expert policy) < `net:models/imitation_v7.pt:300` (52.5 %)
< `net:models/value_v8.pt:300` (62.5 %; same policy, value head trained on playout-averaged
targets) < `rollout:200:1` (75 %) < `hybrid:models/imitation_v7.pt:300:0.5` (80 %, network
priors with playout-blended values) < **`hybrid:models/imitation_v7.pt:600:0.5`** (90 %, 18-2).
Deeper search is what helps; the blend weight, a second playout per leaf and the `value_v8`
head do not move the hybrid.

```bash
sevenwa play --ai hybrid:models/imitation_v7.pt:600:0.5  # strongest measured setting (~2 s per move)
sevenwa play --ai net:models/value_v8.pt:600             # network-only search (~0.5 s per move)
sevenwa play --ai heuristic                              # the tuned expert policy, instant
```

## Layout

```
sevenwa/game.py          abstract GameState / Environment protocols (decision + chance nodes)
sevenwa/engine/          rules engine: cards, wonders, tokens, rules options, state machine, environment
sevenwa/engine/data/     editable JSON game data (decks, wonders, progress tokens)
sevenwa/search/          chance-aware PUCT MCTS, Gumbel root policy, rollout evaluator
sevenwa/nn/              feature encoder, policy/value/score network, batched evaluator
sevenwa/agents/          random, heuristic, rollout-MCTS and neural-MCTS agents (+ spec factory)
sevenwa/train/           self-play, replay buffer, trainer, arena, Elo ladder, pipeline
sevenwa/cli.py           `sevenwa pipeline | arena | selfplay | train | play | bench`
tests/                   rules-fidelity, search, network and heuristic tests
```

## Game data

Deck sizes, card types, Wonder point totals and effects, progress-token effects and all
2-player rules were cross-checked against multiple sources, and on 2026-09-14 the exact
per-deck card distribution and the seven Wonder boards (per-stage VP, effect stages and the
construction prerequisites printed on the trays) were entered from the physical components.
Everything in `sevenwa/engine/data/*.json` is now marked `CONFIRMED`; see `docs/RULES.md` for
the provenance of every item.  The engine models each Wonder as a dependency graph (Rhodes may
start with either foundation, Babylon may build its 4-different stage before its 3-identical
one, ...), with an explicit stage-choice decision when several stages are affordable.
