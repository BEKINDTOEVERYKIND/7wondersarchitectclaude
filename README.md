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

## Game data caveat

Deck sizes, card types, Wonder point totals, Wonder effect types, progress-token effects and
all 2-player rules were cross-checked against multiple sources (see `docs/RULES.md`).  The
exact per-deck card counts and the per-stage cost/VP split of each Wonder could not be
verified during development and are marked `ASSUMED` in `sevenwa/engine/data/*.json`;
correct them there and everything (engine, features, tests) follows automatically.
