# AI design: a self-play reinforcement-learning agent for 2-player *7 Wonders: Architects*

This document explains **why** the system is built the way it is.  The rules the engine
implements are in `docs/RULES.md`; how to train and evaluate is in `docs/TRAINING.md`.

## 1. What kind of game is this?

* **Two players, alternating turns, zero-sum at the end** (higher score wins).
* **Small branching factor**: on a normal turn you choose one of three cards (your deck's
  top card, the opponent's deck's top card, or the face-down central card).  Sub-decisions
  (which cards to spend on a Wonder stage, which Progress token to take, ...) are also small.
* **Stochastic**: every time a card is taken the next card of that deck is revealed; the
  central card is drawn blind; Progress tokens are refilled blind.
* **Almost perfect information**: both players see the same table.  The only private
  information is the Cat holder's peek at the central deck.  Deck *orders* are unknown to
  both players, but the *multiset* of unseen cards in each deck is public knowledge
  (the deck lists are fixed).
* **Short**: ~20–30 decisions per player, ~50–60 plies including sub-decisions.
* **Mandatory building** makes it a race with tempo (the game ends the moment a 5th stage
  is built), so the value of a card depends heavily on the opponent's position.

These properties make the game an excellent fit for **AlphaZero-style self-play with an
expectimax tree search** (chance nodes inside MCTS), which is the most reliable route to
super-human strength in short, low-branching stochastic games (backgammon, 2048-style
games, Skat-like card games with public multisets).

## 2. Architecture overview

```
                     ┌──────────────────────────────┐
   self-play workers │  Engine (belief state, chance │
   (multiprocess)    │  nodes, exact rules)          │
        │            └──────────────┬───────────────┘
        │                           │ GameState protocol (sevenwa/game.py)
        ▼                           ▼
   ┌─────────────┐        ┌───────────────────────┐        ┌──────────────────┐
   │ Replay      │◄───────│ Chance-aware PUCT MCTS │◄──────►│ Policy/Value net │
   │ buffer      │        │ (+ Gumbel root option) │        │ (ResMLP / attn)  │
   └──────┬──────┘        └───────────────────────┘        └────────▲─────────┘
          │  (features, π, z, Δscore)                                │
          ▼                                                          │
   ┌─────────────┐   gate: new net must beat champion   ┌────────────┴─────────┐
   │ Trainer     │─────────────────────────────────────►│ Arena / Elo ladder    │
   └─────────────┘                                      │ vs random, heuristic, │
                                                        │ rollout-MCTS, old nets│
                                                        └───────────────────────┘
```

### 2.1 Engine as a *belief state*

The engine (`sevenwa/engine`) does **not** store deck orders.  A deck is
`(visible top card, multiset of unseen cards)`.  Taking a card creates a **chance node**
whose outcome distribution is "one card drawn uniformly from the unseen multiset".  The
real game (`Environment`) resolves chance nodes with the true shuffled order; the search
resolves them by sampling.  Consequences:

* The search tree is *exactly* the belief tree of a fully-informed observer: no
  determinization, no strategy fusion for deck draws.
* The same state class serves the environment, the search, the feature encoder and the
  heuristics — no separate "information set" code.
* The Cat peek (the one piece of private information) is handled with a `visible_to`
  marker: the known central card is exposed to the feature encoder only when the Cat
  holder is the player to move, so the network of the non-holder never "sees" it.  Inside
  the non-holder's tree the opponent's peek is a sampled chance node (a mild, standard
  approximation; the effect on strength is negligible because the peek only informs one
  of three options).

### 2.2 Chance-aware PUCT MCTS (`sevenwa/search/mcts.py`)

* Decision nodes use PUCT with first-play urgency; chance nodes are traversed by sampling
  their outcome, so the node value converges to the expectation.
* Leaves are evaluated in mini-batches (virtual loss) — one network forward per 8 leaves —
  which is what makes CPU-only training viable.
* Tree reuse between moves, root Dirichlet noise and temperature sampling for self-play.
* Optional Gumbel root (Sequential Halving with Gumbel-Top-k) gives a *policy-improvement*
  target that is much better than raw visit counts at 32–128 simulations, so the pipeline
  can trade simulations for games early in training.

### 2.3 Network (`sevenwa/nn`)

Input is a mover-centric fixed vector (581 floats): both players' wonder, the five stages
each flagged built / available / affordable-now together with their cost, kind, VP and
effect (the Wonders are dependency *graphs*, not ladders — Rhodes may start with either
foundation, Ephesus opens three stages at once), resource/wild counts, blue points, science
symbols, progress tokens, shields, war tokens, cat; per-deck top-card features and the
**unseen-multiset histogram** of each deck (so the net can reason about draw odds); face-up
and unseen progress tokens; conflict-token state; pending decision type (including the
stage-choice decision and the stage being paid for); turn counter.

Trunk: residual MLP (6 × 512, LayerNorm + GELU) — fast on CPU.  An attention variant
(entities = players, decks, tokens) is provided for larger runs.

Heads: **policy** over the global action space (masked), **value** (tanh; win/draw/loss
from the mover's perspective) and an **auxiliary score-difference head** which provides a
dense learning signal (final score margin) that speeds up value learning considerably in
score-based games.

### 2.4 Training loop (`sevenwa/train`)

1. **Bootstrap**: before the network knows anything, generate games with rollout-MCTS
   (heuristic playouts) and pre-train on them.  This removes the slow random-play phase.
2. **Self-play** with N worker processes (each a 1-thread torch model + MCTS), Dirichlet
   noise, temperature 1 for the first *k* moves then greedy, resign disabled (games are
   short).  Targets: search policy π, game result z (from mover's perspective) and final
   score margin.
3. **Train** on a sliding replay window (last *W* generations) with AdamW + cosine LR,
   loss = CE(π) + MSE(z) + λ·MSE(margin).
4. **Gate**: the candidate plays the champion (both colours, fixed seeds); it is promoted
   only if it scores ≥ 55 %.  This keeps the training signal monotone.
5. **Evaluate**: Elo ladder against fixed references (random, heuristic, rollout-MCTS with
   several budgets, and every promoted champion) so progress is measurable.

### 2.5 Baselines (`sevenwa/agents`)

* `RandomAgent` – sanity floor.
* `HeuristicAgent` – a hand-written expert policy (finish stages, deny the opponent's
  completing card, blue points per turn, science pairs, war timing).  Also used as the
  playout policy for rollout-MCTS and as the bootstrap teacher.
* `RolloutMCTSAgent` – MCTS with heuristic playouts (no network); a strong classical
  reference (hundreds of playouts per move).
* `NeuralMCTSAgent` – the trained network + MCTS (the product).

### 2.6 Hybrid evaluation (network + playouts)

`HybridEvaluator` blends the network value with heuristic playouts at every leaf
(`value = (1-λ)·v_net + λ·playout`).  It exists because the value head is the slowest
component to become trustworthy: with few thousand self-play games it memorises the replay window
rather than generalising (see `docs/TRAINING.md`), and a mis-calibrated value head misleads the
search more than a noisy-but-unbiased playout does.  The hybrid agent is therefore the strongest
configuration early in a run, and `λ → 0` recovers the pure AlphaZero agent once the fresh-data
value loss is clearly below 1.

## 3. Why not something else?

* **Pure minimax/expectimax**: the tree is shallow-ish but chance branching (~15 distinct
  unseen cards per deck) × 50 plies is far beyond exact search; MCTS + a learned value
  function is the standard remedy.
* **Information-set MCTS / CFR**: unnecessary — the only hidden information is the Cat
  peek; deck order is *chance*, not opponent knowledge.
* **Policy-gradient only (PPO)**: works but is far less sample-efficient than
  search-guided policy iteration on a CPU budget.
* **MuZero**: learning the dynamics is pointless when the simulator is exact and cheap.

## 4. Correctness safeguards

* Rules fidelity is documented per rule (`docs/RULES.md`), with ambiguous rules exposed as
  `RulesConfig` options rather than silently chosen.
* Property tests: card conservation, mandatory-build invariants, score bookkeeping,
  chance-distribution normalisation, search value convergence on a toy game.
* Arena results are reported with binomial confidence intervals.
