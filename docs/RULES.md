# Rules specification — 2-player *7 Wonders: Architects* as implemented by `sevenwa.engine`

This is the canonical description of the rules the engine implements, written for a programmer
or reviewer auditing `sevenwa/engine/state.py` (the belief-state rules engine), `env.py` (the real
game with hidden information), `rules.py` (`RulesConfig`) and the data files in
`sevenwa/engine/data/`.  It merges the rules research
(`research/core-rules.md`, `research/two-player-and-faq.md`, compiled from search-engine
summaries of the official rulebook and of BoardGameArena/BGG pages — direct downloads were
blocked) with what the code actually does.  Where the two disagree, or where the code goes beyond
what the sources confirm, this document says so explicitly.  It describes `state.py` as of
2026-09-13 (the version whose module docstring is quoted in §2.2: science sets before
construction, canonical non-decreasing payment order, `hali_event`).

Confidence tags used throughout:

* **CONFIRMED** — same statement relayed from at least two independent sources (or quoted rulebook
  text mirrored by rules sites).
* **SINGLE** — one source only.
* **ASSUMED** — not verifiable offline; an internally consistent placeholder (see §11 for the exact
  JSON fields to correct).
* **ENGINE** — a modelling decision of this implementation, not a rule.

Section §14 lists engine defects and discrepancies found while writing this document.

---

## 0. Vocabulary and identifiers

| Term | Engine meaning |
|---|---|
| player `p` ∈ {0, 1} | `state.mover` is the player to move; the opponent is `1 - p`. |
| deck `d` ∈ {0, 1, 2} | **Absolute** indices: `0` = player 0's Wonder deck, `1` = player 1's Wonder deck, `2 = CENTRAL` (the face-down common deck). |
| `PICK_LEFT` / `PICK_RIGHT` | **Relative** actions: `PICK_LEFT` = the mover's *own* deck (`d = mover`), `PICK_RIGHT` = the *opponent's* deck (`d = 1 - mover`), `PICK_CENTER` = deck 2.  Rulebook: each player's deck is placed "between the player to your left and yourself", so your own deck is on your left and, with 2 players, the opponent's deck is on your right. |
| kind `k` ∈ 0..13 | A card *kind* (`NUM_KINDS = 14`).  Cards of the same kind are indistinguishable; every collection of cards (tableau, unseen part of a deck, discard) is a count vector indexed by kind. |
| tableau | `state.cards[p]` — the cards in front of player `p` (count vector). |
| unseen / top | A deck is `(deck_top[d], unseen[d])`: the visible top kind (`-1` = hidden/none) plus the multiset of cards below it.  `deck_size[d] = (1 if deck_top[d] >= 0 else 0) + sum(unseen[d])`; the central deck's top is normally hidden and counted inside `unseen`, and is set only after a Cat peek. |
| node | A state is a **decision node** (`to_move()` ∈ {0,1}, `legal_actions()` non-empty), a **chance node** (`to_move() == CHANCE == -1`, `chance_outcomes()` is a distribution) or **terminal** (`to_move() == -2`). |
| stage | `stages[p]` = number of constructed stages (0..5); the *next* stage to build is `WONDERS[wonder[p]].stages[stages[p]]`. |

Card kinds (`data/decks.json`, `cards.py`):

| id | name | type | properties |
|---|---|---|---|
| 0 | `wood` | grey | resource 0 |
| 1 | `stone` | grey | resource 1 |
| 2 | `clay` | grey | resource 2 |
| 3 | `papyrus` | grey | resource 3 |
| 4 | `glass` | grey | resource 4 |
| 5 | `coin` | yellow | 1 coin (wild resource) |
| 6 | `civ3` | blue | 3 VP |
| 7 | `civ2cat` | blue | 2 VP + Cat icon |
| 8 | `tablet` | green | symbol 0 |
| 9 | `gear` | green | symbol 1 |
| 10 | `compass` | green | symbol 2 |
| 11 | `shield` | red | 1 shield, 0 horns |
| 12 | `shield_h1` | red | 1 shield, 1 horn |
| 13 | `shield_h2` | red | 1 shield, 2 horns |

Global action space (`actions.py`, `Actions.NUM = 44`; the policy head has this width and
illegal actions are masked):

| id | action | used at |
|---|---|---|
| 0 | `PICK_LEFT` (own deck) | pick decisions |
| 1 | `PICK_RIGHT` (opponent's deck) | pick decisions |
| 2 | `PICK_CENTER` | pick decisions |
| 3 | `SKIP` | optional picks only (decline an extra card / an optional effect) |
| 4–8 | `PAY_BASE + r` — spend one grey card of resource `r` (payment code `r`) | payment decisions |
| 9 | `PAY_COIN` — spend one yellow card as 1 coin (code 5) | payment decisions |
| 10 | `PAY_COIN2` — spend one yellow card as 2 coins, Economy (code 6) | payment decisions |
| 11–13 | `SCI_PAIR_BASE + s` — discard 2 identical symbols `s` | science decisions |
| 14 | `SCI_TRIPLE` — discard tablet + gear + compass | science decisions |
| 15–28 | `TOKEN_BASE + t` — take face-up Progress token type `t` | token decisions |
| 29 | `TOKEN_BLIND` — take the top (face-down) token of the stack | token decisions |
| 30–43 | `HALI_BASE + k` — keep the revealed card of kind `k` | Halicarnassus decisions |

---

## 1. Components and setup (2 players)

### 1.1 Components in play

| Component | Count in a 2-player game | Engine |
|---|---|---|
| Wonders | 2 of the 7 (Giza, Rhodes, Alexandria, Halicarnassus, Olympia, Ephesus, Babylon — ids 0..6 in `wonders.json` order), 5 stages each | `state.wonder = (w0, w1)`; `Environment(wonders=None)` draws 2 *distinct* Wonders uniformly. **CONFIRMED** (7 Wonders × 5 stages). |
| Wonder decks | 1 per Wonder, 25 cards, **face up** | decks 0 and 1; composition from `decks.json` (§1.2). Deck sizes **CONFIRMED**, composition **ASSUMED**. |
| Central (common) deck | 60 cards, **face down** | deck 2. Size **CONFIRMED**, composition **ASSUMED**. |
| Discard pile | face up next to the central deck; never reshuffled | `state.discard` (count vector, inert). **CONFIRMED**. |
| Progress tokens | 15: 14 types, Culture ×2; shuffled face-down stack, top **3** face up | `prog_unseen` (per-type counts still in the stack), `prog_stack` (its size), `faceup` (list of type ids). `RulesConfig.faceup_progress_tokens = 3`. **CONFIRMED**. |
| Conflict tokens | **3** Peace-side up (the other 3 of the 6 are removed) | `conflict` = number flipped to the Battle side (0..3); `RulesConfig.conflict_tokens = 3`. Table 3/3/4/5/6/6 for 2..7 players: **CONFIRMED** by DE + IT sources (no EN snippet gave the numbers). |
| Military Victory tokens | 28 in the box, 3 VP each | `mil_tokens[p]`; supply treated as unlimited (**ENGINE**; no source addresses exhaustion). |
| Cat pawn | 1, starts in the middle, unowned | `cat = -1`. **CONFIRMED** (start position wording **SINGLE**). |

In a 2-player game exactly 2 × 25 + 60 = **110 cards** are in play.  The **card conservation
invariant** holds at every node: for each kind `k`,

```
cards[0][k] + cards[1][k] + Σ_d unseen[d][k] + Σ_d [deck_top[d] == k] + discard[k]
  + (cards currently "in transit" inside the queue: a taken card between its `take` and `placed` items,
     Halicarnassus revealed cards held in the chance/decision context)
  = wonder_deck_counts(w0)[k] + wonder_deck_counts(w1)[k] + central_deck_counts()[k]
```

(`tests/conftest.py::assert_invariants` checks it over random games.)

### 1.2 Deck compositions (`data/decks.json`) — **ASSUMED**

Only the deck *sizes* (25 / 60) and rough proportions are confirmed (BoostYourPlay: "over 1/3 of
the cards are grey, ~12 % yellow, science / military / civilian ~17 % each; each Wonder deck differs
from the standard configuration by 2–3 cards; Alexandria has +1 coin, +1 gear, −1 glass, −1 one-horn
military" — **SINGLE**).  The exact per-deck counts below are placeholders that satisfy those
constraints; Babylon/Ephesus/Halicarnassus/Olympia are *assumed* to rotate the favoured science
symbol.

| kind | standard Wonder deck | Alexandria | Babylon | Ephesus | Giza | Halicarnassus | Olympia | Rhodes | central |
|---|---|---|---|---|---|---|---|---|---|
| wood | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 |
| stone | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 4 |
| clay | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 |
| papyrus | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 |
| glass | 2 | **1** | 2 | 2 | 2 | 2 | 2 | 2 | 3 |
| coin | 3 | **4** | 3 | 3 | 3 | 3 | 3 | 3 | 8 |
| civ3 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 6 |
| civ2cat | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 6 |
| tablet | 2 | 2 | **1** | **1** | 2 | **1** | **1** | 2 | 4 |
| gear | 1 | **2** | **2** | 1 | 1 | 1 | **2** | 1 | 4 |
| compass | 1 | 1 | 1 | **2** | 1 | **2** | 1 | 1 | 4 |
| shield | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 3 |
| shield_h1 | 2 | **1** | 2 | 2 | 2 | 2 | 2 | 2 | 6 |
| shield_h2 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 3 |
| **total** | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 60 |

`cards.wonder_deck_counts(name)` = `standard_wonder_deck` updated with `wonder_deck_overrides[name]`
(keys starting with `_` are comments); `central_deck_counts()` = `central_deck`.

### 1.3 Setup sequence as executed by the engine

`GameState.initial(wonders, rules)` starts with the work queue

```
("reveal", 0), ("reveal", 1), ("token_reveal",) × faceup_progress_tokens, ("turn_start",)
```

i.e. **five chance nodes** before the first decision: the top card of each Wonder deck is
revealed (distribution = that deck's composition), then three Progress tokens are turned face up
(distribution = token copies / 15, drawn without replacement).  The central deck's top stays
hidden.  `conflict = 0`, `cat = -1`, all tableaus empty, `mover = 0`, `turn = 0`.

**First player.**  Rulebook: "starting with the youngest player" (**SINGLE**).  **ENGINE:** player
0 always moves first; `Environment` randomises the Wonder pair but not the seat, so evaluation code
must alternate seats (the arena does).

---

## 2. Turn structure and the engine's canonical resolution order

### 2.1 What the rulebook says (**CONFIRMED**)

> "On your turn, choose one of three cards: the top card of the deck directly to your left, the
> top card of the deck directly to your right, or the top card of the central deck.  Place your
> chosen card in front of you, near your Wonder.  Depending on the cards and tokens you have, take
> the different actions explained below, **in any order**.  When you cannot take any more actions,
> your turn is over and the player to your left can begin their turn."

> "During your turn, you can use your Progress tokens in any order and whenever you want.  Each of
> your Progress tokens can only be used **once per turn**, except for those that provide Victory
> points, which will only be used at the end of the game."

The base turn is exactly one card.  Extra cards come only from Progress tokens and Wonder-stage
effects.  Wonder decks are face up (the next card is visible as soon as the top is taken); the
central card is a blind draw unless the Cat peek applies.  Construction and science sets are
**mandatory**; Battle (if triggered) resolves at the end of the turn; the game ends at the end of
the turn in which a 5th stage is built.

### 2.2 The engine's canonical order (quoted from the `state.py` module docstring)

> Canonical order of resolution within a turn (the rulebook lets the player choose any order;
> fixing an order keeps the tree small and, because every mandatory action is still taken, does
> not change what a player can achieve):
>
> 1. (Cat holder) peek at the central deck.
> 2. Main pick: left (own) deck, right (opponent's) deck or central deck.
> 3. Card placed -> immediate effects (Cat pawn, horns) -> Progress-token extra picks that the
>    card triggers -> mandatory science sets -> Progress token choice -> mandatory construction
>    (with payment choices) -> stage effect -> Architecture extra pick -> repeat until nothing is
>    pending.  Science is resolved before construction so that a token gained this turn
>    (Engineering, Economy, Architecture...) already applies to this turn's build; doing science
>    first never prevents a build (green cards are not building resources), so this order weakly
>    dominates the alternative.
> 4. End of turn: battle (if triggered), game end (if a Wonder is complete), next player.
>
> Payments are sequences of "pay one card" decisions (grey resource, coin, or a coin worth 2 with
> Economy).  Codes must be chosen in non-decreasing order (wood < stone < clay < papyrus < glass <
> coin < coin×2), which removes permutations of the same multiset from the tree: a payment
> decision only appears when genuinely different sets of cards could be spent.

### 2.3 How the order is realised: the work queue

Turn processing is a small work queue (`state.queue`, a tuple of items).  `_run()` pops the first
item and handles it until a handler reports a *real* choice (a decision with ≥ 2 legal actions or a
chance event with ≥ 2 possible outcomes) or the game ends.  **Forced choices are auto-resolved**
(`_set_decision` applies a single legal action immediately; `_set_chance` applies a single possible
outcome immediately), so the tree contains only genuine decisions.  `_push(*items)` places items at
the **front** of the queue in the order given (the first argument executes first).

| Queue item | Handler behaviour |
|---|---|
| `("turn_start",)` | Reset `tokens_used` and `econ_used`.  If every deck is empty → game over (unconditionally, see §9.1).  Prepend `("pick","main",False,(0,1,2))`, `("end_turn",)`.  Cat holder peek (§8). |
| `("pick", reason, optional, sources)` | Offer `PICK_*` for every source deck with cards; add `SKIP` if optional.  If no source has cards, the item is dropped silently. |
| `("take", d, reason)` | Take deck `d`'s top: Wonder deck → `deck_size -= 1`, top hidden, prepend `("reveal", d)`, `("placed", k, d, reason)`; central deck → §8.2. |
| `("reveal", d)` | If deck `d` is non-empty and its top is hidden: chance node `C_REVEAL` over `unseen[d]`. |
| `("placed", k, d, reason)` | Add the card to the mover's tableau; immediate effects (Cat, horns); mark the triggering token used if `reason == "token:<t>"`; prepend `("check",)`, then (in front of it) one `("pick", "token:<t>", extra_card_optional, (0,1,2))` for each held, not-yet-used token the card triggers. |
| `("check",)` | **Science first**: if a science set is possible and tokens remain → science decision (§6.2).  Otherwise, if `stages < 5` and the next stage is affordable → payment decision (§4.3).  Otherwise nothing. |
| `("pay", chosen)` | Continue a multi-step payment. |
| `("token_choice", reason)` | Offer the face-up tokens and, if the stack is non-empty, `TOKEN_BLIND`; nothing if no token is left. |
| `("token_reveal",)` / `("token_blind",)` | Chance node over `prog_unseen` (refill a face-up slot / blind draw). |
| `("hali_deck",)`, `("hali_reveal", d, remaining, revealed)` | Halicarnassus (§5.3). |
| `("end_turn",)` | Resolve the Battle if pending; end the game if a Wonder is complete; else `mover = 1 - mover`, `turn += 1`, queue `("turn_start",)`. |

Because `_push` prepends, the concrete order after a card is placed is:

```
[token extra pick(s)] → check → (science set → token choice → [face-up refill] → check …)
                              → (build → [stage effect] → [Architecture pick] → check …)
```

and after the main pick the queue is just `(("end_turn",),)` — everything else is generated by
the handlers.  §13 shows a real trace.

Decision kinds: `D_PICK`, `D_PAY`, `D_SCIENCE`, `D_TOKEN`, `D_HALI_DECK`, `D_HALI_CHOOSE`
(`state.dkind`, context in `state.dctx`).  Chance kinds: `C_REVEAL` (new top of a Wonder deck),
`C_DRAW_CENTRAL` (blind central draw), `C_PEEK` (Cat peek), `C_TOKEN_REVEAL` (face-up refill),
`C_TOKEN_BLIND` (blind token), `C_HALI_REVEAL` (Halicarnassus reveal).  Every chance distribution
is "one item drawn uniformly from the relevant unseen multiset" (§2.5).

### 2.4 Why fixing an order is (almost) safe

The rulebook's "any order" only matters when actions interact.  The engine's order guarantees:

* **No mandatory action can be dodged.**  `("check",)` is re-queued after every placement, every
  build and every science set, so construction and science sets are enforced whenever they become
  possible at any point of the turn ("If you have the Resources to construct multiple Stages in one
  turn, you must do so" — **CONFIRMED**).
* **Every card / token the mover is entitled to by the cards placed is received in the same turn**
  (extra picks are queued at placement time; stage effects at build time), and a Battle or the game
  end is only evaluated once the queue is drained.
* **Science before construction weakly dominates**: a token gained from a set (Architecture,
  Engineering, Economy, …) already applies to the build that follows in the same check, and green
  cards are never building resources, so resolving the set first can never prevent a build.
* **Information is maximised before choices:** token extra picks resolve *before* the science /
  build check, and the next top card of a Wonder deck is revealed *before* the taken card's
  effects resolve, so a payment choice always sees every card the player could have had in hand.

What the fixed order *does* change is enumerated in §12.1 (essentially: a token acquired during
the turn cannot fire for a card that was already placed, and the two Olympia cards are separated
by a check).  These are second-order effects; the design accepts them for tree size.

### 2.5 Chance-node semantics (belief state)

The state never stores deck orders.  Revealing/drawing a card from deck `d` is a chance node whose
outcome distribution is `unseen[d][k] / Σ unseen[d]`; a Progress-token draw uses
`prog_unseen[t] / prog_stack`.  This is exactly the belief of an observer who has seen everything
public, so the search tree is the true expectimax tree (no determinisation), except for the Cat
peek (§8).  The real game (`Environment`) resolves every chance node with the true shuffled order
(`decks[d][0]`, `token_stack.pop(0)`, …) and raises if the true outcome has probability 0 under the
belief — a permanent consistency check between the physical model and the belief.

---

## 3. Card types and their rules

| Type | Rule (**CONFIRMED** unless tagged) | Engine |
|---|---|---|
| **Grey — Resources** | One of Wood, Stone, Clay, Papyrus, Glass per card.  Used (and discarded) to construct Wonder stages; unused grey cards stay in front of you. | `_placed` may trigger Urbanism (wood/clay), Crafts (papyrus/glass), Jewellery (stone).  Spent cards go to `discard`. |
| **Yellow — Coins** | "Yellow cards provide Coins that act as a wild Resource.  These Coins must replace any missing Resource needed to construct a Stage of your Wonder."  One coin per card; discarded when used. | Kind `coin` (`COIN_KIND = 5`).  Wild in identical and different requirements (§4.2).  Triggers Jewellery. |
| **Blue — Civilian** | Either **3 VP**, or **2 VP + a Cat icon**.  "As soon as you take a Blue card with a Cat icon, take the Cat pawn from wherever it is on the table and place it in front of you."  Never discarded. | `civ3` (3 VP), `civ2cat` (2 VP, `cat=True` → `state.cat = mover` immediately, stealing it if the opponent held it).  Cat icons feed Politics. |
| **Green — Science** | One of 3 symbols per card (tablet, gear/cog, compass/protractor).  "If you have 2 identical or 3 different Science symbols, you **must** take a Progress token during your turn.  To obtain a Progress token, discard any Green cards used, then choose … any one of the three face-up tokens, or the top token of the face-down stack."  Leftover green cards stay. | §6.  Triggers the Science token. |
| **Red — Military** | Each red card shows shield(s) and 0, 1 or 2 War-Horn icons.  Horns flip Conflict tokens (§7).  Red cards **with** horns are discarded after the next Battle; red cards **without** horns are kept for the whole game. | Every red kind has exactly **1 shield** (**INFERRED** from "every military card holding a shield"; **ASSUMED**).  `shield` (0 horns), `shield_h1`, `shield_h2`.  Horn cards trigger Propaganda. |

Card retention summary: grey/yellow → discarded when spent on a stage; green → discarded when
converted to a Progress token; red with horns → discarded after the next Battle; red without horns
and blue → kept (FR Wikipedia summary, **CONFIRMED**).  Leftover grey/yellow/green/red cards score 0.

---

## 4. Wonder construction

### 4.1 Rules (**CONFIRMED**)

* Each Wonder has 5 stages built strictly bottom to top; the next stage is
  `stages[stages[p]]`.
* Each stage shows a cost: **2, 3 or 4 resources**, either all **identical** ("=" icon) or all
  **different** ("≠" icon).
* "If you have the Resources needed to construct a Stage of your Wonder, you **must** construct it
  during your turn."  "This includes the use of available yellow cards."  Several stages may (and
  must) be built in the same turn.
* Only the cards used are discarded; leftover resource cards remain.
* Stage effects apply immediately after the flip (§5).
* The game ends at the end of the turn in which a player builds the 5th stage (§9).

### 4.2 Payment rules as implemented

Let `cost` and `kind ∈ {identical, different}` be the next stage's requirement.  A payment is a
multiset of the mover's grey and yellow cards with a total *value* of exactly `cost` where

* a grey card is worth 1 and counts as its resource;
* a yellow card is worth **1 coin**, a wild that "must replace any missing Resource": for an
  *identical* stage it counts as the resource of the grey cards used; for a *different* stage it
  counts as a resource not already present.  Any number of coins may be used (up to the cost), so
  a stage can be paid with coins alone (BGG "Mandatory or not?" thread: yes, mandatory even with
  only yellow cards — **CONFIRMED**);
* with **Economy**, one yellow card per turn may be worth **2 coins** (`PAY_COIN2`), see §6.3 — in
  a *different* stage the doubled coin stands for two distinct missing resources;
* with **Engineering**, the identical/different constraint is dropped: any `cost` grey/yellow cards
  suffice (BGG threads 2773240, 3517431 — **CONFIRMED**).

Payment *mode* (`_mode()`): `2` (any) if the mover holds Engineering, else `0` (identical) or `1`
(different) from the stage.  `_max_value(grey_avail, coins_avail, econ_used, used_res, mode,
min_code)` is the maximum value still reachable using only payment codes `≥ min_code` (§4.3):

| mode | maximum value |
|---|---|
| identical | `grey_avail[r*] + coin_cap` where `r*` is the resource already committed (else the best resource with code ≥ `min_code`) |
| different | number of distinct resources with code ≥ `min_code`, still available and not yet used, `+ coin_cap` |
| any (Engineering) | `Σ_{r ≥ min_code} grey_avail[r] + coin_cap` |

with `coin_cap = coins_avail` (only if `min_code ≤ 5`) `+ 1` if Economy is held, unused this turn,
at least one coin is available and `min_code ≤ 6`.

**Affordability** (`_affordable`): `_max_value(..., min_code = 0) >= cost` on the current tableau.
Note that this counts the Economy doubling, i.e. **a stage that can only be completed thanks to
Economy is mandatory** — the BoardGameArena behaviour (bug #64468 developer reply: "players must
use the economic token to double their coin") rather than the "tokens are optional" reading of BGG
thread 2758424.  **ENGINE / SINGLE**; no `RulesConfig` flag exposes the alternative (§14).

### 4.3 The payment protocol (decision `D_PAY`)

Paying is a sequence of `PAY_*` actions, one card at a time (`dctx` = the tuple of codes chosen so
far; codes 0..4 = grey resource in `RESOURCES` order, 5 = coin, 6 = coin ×2).  **Codes must be
non-decreasing** (`min_code` = the last code chosen): grey cards are named in resource order, then
plain coins, then the doubled coin last.  Every distinct *multiset* of cards remains reachable, but
its permutations collapse to one path, so a `D_PAY` node appears only when genuinely different sets
of cards could be spent (e.g. exactly wood + stone for a "2 different" stage is built without any
decision).

At each step the legal actions are exactly the cards whose use still leaves the remainder payable:

* `PAY_BASE + r` is legal iff `r ≥ min_code`, a grey card of resource `r` is available, the mode
  allows it (identical: same resource as the first grey card; different: a resource not yet used)
  and `1 + _max_value(after using it, min_code = r) >= need` where `need = cost − value so far`;
* `PAY_COIN` is legal iff `min_code ≤ 5`, a coin is available, `1 + _max_value(after, min_code = 5)
  >= need`, **and** either `coins_free_choice` is on or no grey option is legal.  Because only
  coins may follow a coin, a coin can be named before a grey card only when the rest of the cost is
  paid with coins too;
* `PAY_COIN2` is legal iff Economy is held and unused this turn, a coin is available, `need >= 2`
  (never overpay) and `2 + _max_value(after, min_code = 6) >= need`; since nothing may follow code
  6, the doubled coin is always the last card of a payment.

When the accumulated value reaches `cost`, `_build` runs: the chosen cards move to `discard`,
`stages[p] += 1`, `econ_used = True` if a doubled coin was used, `wonder_done = True` if the 5th
stage was built, then `("check",)`, the Architecture pick (if held and unused) and the stage effect
are queued (effect first, Architecture second, check last).  Steps with a single legal action are
auto-resolved, so e.g. "2 coins for a 2-identical stage" is built without any decision node.

`RulesConfig.coins_free_choice = True` (**ENGINE**): the player may spend a coin even when a grey
card would do ("These Coins must replace any missing Resource" does not say you may not spend a
coin *instead of* an available grey card; allowing it is the superset).  With the flag off, coins
are legal only when no grey option is.

Known defect (§14 item 6): for an *identical* stage the feasibility test after a grey card ignores
`min_code`, so `pay_coin` can be offered where it dead-ends and the build is silently dropped.

---

## 5. The Wonders

### 5.1 Effects (`wonders.py`)

| effect id | Wonder | Rule text (as relayed) | Engine (`_build`) |
|---|---|---|---|
| `none` | Giza (all), others' plain stages | — | nothing |
| `shield` | Rhodes | "Add 1 Shield to your total Shields" — permanent, survives Battles.  **CONFIRMED** (2 stages) | `wonder_shields[p] += 1` |
| `progress_token` | Babylon | Take a Progress token: one of the 3 face-up or the top of the stack.  **CONFIRMED** (2 stages) | `("token_choice","babylon")`; fizzles if no token is left.  Never optional. |
| `any_deck_card` | Alexandria | "Take the top card from any deck anywhere on the table and place it in front of you."  **SINGLE** (2 stages) | `("pick","alexandria", wonder_effect_optional, (0,1,2))` |
| `central_card` | Ephesus | "Take the top card from the central deck and place it in front of you."  **CONFIRMED** effect; **3** stages per BGG ranking / "3 columns" (one review says 2) | `("pick","ephesus", wonder_effect_optional, (2,))` — auto-resolved (forced) when not optional; blind unless the Cat-peeked card is still on top (§8) |
| `left_and_right_cards` | Olympia | "Take the top card from the decks to your left and your right."  **CONFIRMED** (2 stages) | `("take", own deck)`, `("take", opponent's deck)` — no choice, never optional, an empty deck is skipped |
| `look_5_choose_1` | Halicarnassus | "Take the top 5 cards from the deck to your left or your right.  Choose 1 and place it in front of you.  Shuffle the other cards back into their deck."  **SINGLE** (2 stages) | §5.3 |

Cards taken through an effect are placed like any other card: they trigger tokens, Cat, horns and
the science / construction check (§2.3).  With `wonder_effect_optional = True` the Alexandria /
Ephesus picks and the Halicarnassus deck choice gain a `SKIP` action (BGG thread 2835875 "Is it
mandatory to use a Wonder's effect?" — answer not retrievable; default **mandatory**).

### 5.2 Per-stage data currently encoded (`data/wonders.json`)

> **PROMINENT WARNING.**  For every Wonder the **total VP** (30/26/25/24/22/22/20), the **effect
> type** and the **number of effect stages** are **CONFIRMED** (BGG "Wonders. Which is the best?",
> BGA game page).  The **per-stage costs**, the **per-stage VP split** and **which stages carry the
> effects** are **ASSUMED**: costs follow the pattern reported by reviews ("2 different, 2 identical,
> 3 different, 3 identical, 4 different" — **SINGLE**), effects were placed on stages 2 and 4 by
> analogy with Rhodes (the only Wonder whose effect stages, 2 and 4, are confirmed — BGG thread
> 2869259), and Ephesus's three effects on stages 1, 3, 5.  VP splits are monotone placeholders
> summing to the confirmed totals.  Verify against the physical trays before trusting any
> Wonder-specific strength conclusion.

Format of a row: `cost kind / VP [effect]` (`≠` = different, `=` = identical).

| id | Wonder | stage 1 | stage 2 | stage 3 | stage 4 | stage 5 | total VP |
|---|---|---|---|---|---|---|---|
| 0 | Giza | 2≠ / 4 | 2= / 5 | 3≠ / 6 | 3= / 7 | 4≠ / 8 | **30** |
| 1 | Rhodes | 2≠ / 3 | 2= / 4 **shield** | 3≠ / 5 | 3= / 6 **shield** | 4≠ / 8 | **26** |
| 2 | Alexandria | 2≠ / 3 | 2= / 4 **any_deck_card** | 3≠ / 5 | 3= / 6 **any_deck_card** | 4≠ / 7 | **25** |
| 3 | Halicarnassus | 2≠ / 3 | 2= / 4 **look_5_choose_1** | 3≠ / 5 | 3= / 5 **look_5_choose_1** | 4≠ / 7 | **24** |
| 4 | Olympia | 2≠ / 3 | 2= / 3 **left_and_right_cards** | 3≠ / 5 | 3= / 5 **left_and_right_cards** | 4≠ / 6 | **22** |
| 5 | Ephesus | 2≠ / 2 **central_card** | 2= / 4 | 3≠ / 4 **central_card** | 3= / 6 | 4≠ / 6 **central_card** | **22** |
| 6 | Babylon | 2≠ / 3 | 2= / 3 **progress_token** | 3≠ / 4 | 3= / 4 **progress_token** | 4≠ / 6 | **20** |

**How to correct `wonders.json`.**  Each Wonder is `{"name", "total_vp", "stages": [5 × {"cost": int,
"kind": "identical"|"different", "vp": int, "effect": <effect id or omitted for none>}]}`, listed
bottom stage first.  Edit the `cost`, `kind`, `vp` and `effect` fields of the affected stages; keep
exactly 5 stages; `effect` must be one of the ids in §5.1 (`wonders.py` asserts this at import).
`total_vp` is informational (the engine sums the stage VP); keep it in sync.  The order of the
`wonders` array defines the Wonder ids — do not reorder.  `tests/test_engine.py::TestData::
test_wonder_stage_tables` pins the totals, the effect type per Wonder, the number of effect stages
(Ephesus 3, others 2, Giza 0) and Rhodes's effect stages `[1, 3]`; update those expectations
together with the data if the physical trays differ (e.g. if Ephesus turns out to have 2 effect
stages).

### 5.3 Halicarnassus in detail

1. `("hali_deck",)`: choose the own or the opponent's Wonder deck among the non-empty ones
   (`D_HALI_DECK`; `SKIP` only with `wonder_effect_optional`); auto-resolved if only one deck has
   cards; dropped if both are empty.
2. The visible top of the chosen deck is the first revealed card; `min(5, deck_size) − 1` further
   cards are revealed by successive `C_HALI_REVEAL` chance nodes (draws without replacement from
   `unseen[d]`).
3. `D_HALI_CHOOSE`: keep one of the revealed *kinds* (`HALI_BASE + k`; identical kinds are one
   action; auto-resolved when all revealed cards are of one kind).  The other cards return to
   `unseen[d]` ("shuffle the other cards back"), `deck_size[d] −= 1`, the state records
   `hali_event = (d, number of revealed cards, kept kind)`, a new top is revealed (`C_REVEAL`) and
   the kept card is placed with reason `"hali"` (it triggers tokens / Cat / horns / the check like
   any card).
4. `Environment` mirrors this physically: after every `apply_action` and after every resolved
   chance node it consumes `hali_event` — the kept card is removed from the top-`n` window and
   `window_rest + deck[n:]` is reshuffled (the belief already treats the returned cards as unseen).
   This also covers the auto-resolved case.  `hali_event` is never copied into successor states
   (`_copy` resets it), so it is not part of `key()`.

**ENGINE:** the revealed cards are stored in the public decision context, i.e. they are treated as
visible to both players (the rulebook does not say whether the opponent sees them).

---

## 6. Science and Progress tokens

### 6.1 The 15 tokens (`data/tokens.json`)

| id | name | kind | effect as implemented | timing | confidence |
|---|---|---|---|---|---|
| 0 | Architecture | `extra_card_on_build` | When you construct a stage, choose 1 extra card from the 3 available. | once per turn, optional | **CONFIRMED** |
| 1 | Propaganda | `extra_card_on_horn` | When you take a red card with 1 or 2 horns, choose 1 extra card. | once per turn, optional | **CONFIRMED** |
| 2 | Urbanism | `extra_card_on_resource` (wood, clay) | When you take a Wood or Clay grey card, choose 1 extra card. | once per turn, optional | wording **CONFIRMED**, resource split **ASSUMED** |
| 3 | Crafts | `extra_card_on_resource` (papyrus, glass) | When you take a Papyrus or Glass grey card, choose 1 extra card. | once per turn, optional | wording **CONFIRMED**, resource split **ASSUMED** |
| 4 | Jewellery | `extra_card_on_resource` (stone, coin) | When you take a Stone grey card **or a Yellow card**, choose 1 extra card. | once per turn, optional | wording **CONFIRMED**, split **ASSUMED** |
| 5 | Science | `extra_card_on_green` | When you take a Green card, choose 1 extra card. | once per turn, optional | **SINGLE**; wording inferred by analogy |
| 6 | Engineering | `engineering` | Construct stages with any resources, ignoring identical/different. | passive | **CONFIRMED** |
| 7 | Economy | `economy` | One of your yellow cards is worth 2 coins. | once per turn (§6.3), optional | **CONFIRMED** |
| 8 | Tactics | `shields` (+2) | Add 2 shields to your total. | passive | **CONFIRMED** |
| 9 | Strategy | `vp_per_military_token` (1) | End: 1 VP per Military Victory token. | end-game | **CONFIRMED** |
| 10 | Education | `vp_per_progress_token` (2) | End: 2 VP per Progress token you have, including this one. | end-game | **CONFIRMED** |
| 11 | Decor | `vp_wonder` (4 / 6) | End: 4 VP if your Wonder is unfinished, 6 VP if complete. | end-game | **CONFIRMED** |
| 12 | Politics | `vp_per_cat_icon` (1) | End: 1 VP per Cat icon on your blue cards (icons, not the pawn). | end-game | **CONFIRMED** |
| 13 | Culture (×2) | `culture` (4 / 12) | End: 4 VP with one Culture token, 12 VP with both. | end-game | **CONFIRMED** |

13 types + 1 duplicate = 15 tokens (**INFERRED** that these are exactly the 15; every name was
found individually and the total matches).

### 6.2 Acquiring tokens (science sets, Babylon)

* After every placement / build / token choice the `("check",)` item **first** tests whether the
  mover can form a science set: `SCI_PAIR_BASE + s` for each symbol with ≥ 2 cards, `SCI_TRIPLE`
  if all three symbols are present.  If at least one set is possible **and at least one Progress
  token remains** (face-up or in the stack — **ENGINE**: with no token to gain, green cards are not
  discarded), the set is **mandatory**: the decision `D_SCIENCE` has no skip; when several sets are
  possible the player chooses which (**INFERRED** from "discard any Green cards used").  A single
  possible set is auto-resolved; two-option science decisions arise only when an extra card lands
  before the check (e.g. a Science-token extra card completing a second set), so they are rare.
* The used green cards go to `discard` (2 for a pair, 3 for a triple); leftovers stay.  Then
  `("token_choice","science")`, `("check",)` are queued — so a second set, or the build that the
  new token enables / that was already affordable, follows in the same turn.
* `D_TOKEN`: choose any face-up token (`TOKEN_BASE + t`) or `TOKEN_BLIND` (only if the stack is
  non-empty).  A face-up token is replaced **immediately** from the stack (`C_TOKEN_REVEAL`,
  distribution `prog_unseen / prog_stack`); a blind token is a `C_TOKEN_BLIND` chance node with the
  same distribution.  With an empty face-up row the blind draw is forced (auto-resolved); the very
  last token is taken automatically.
* Babylon stages queue the same `token_choice` (reason `"babylon"`).

### 6.3 Using tokens

* **Once per turn.**  `tokens_used` is a bitmask reset at `turn_start`.  For the extra-card tokens
  the bit is set when the extra card is *placed* (`reason == "token:<t>"`); a token whose extra pick
  was **declined** (`SKIP`) or could not be served (all decks empty) is **not** consumed and may
  trigger again later in the turn.  Economy uses its own flag `econ_used` (set when a doubled coin
  is spent, reset at `turn_start`), i.e. **once per turn**, across all stages built that turn.
* **Optional.**  Extra picks carry `SKIP` (`extra_card_optional = True`; rulebook "you can use").
  Economy is optional at payment time (`PAY_COIN2` is one option among the legal ones) but counts
  towards *affordability* (§4.2).
* **Trigger points** (`_placed` / `_build`): a grey card of a listed resource → Urbanism / Crafts /
  Jewellery; a yellow card → Jewellery; a green card → Science; a red card with ≥ 1 horn →
  Propaganda; a constructed stage → Architecture.  Extra picks are offered from **all three decks**
  ("from the 3 available"); a central extra pick is a blind draw unless the Cat-peeked card is still
  on top (§8.3).  Extra picks resolve **before** the check of the card that triggered them
  (§2.3); the Architecture pick resolves **after** the stage effect.
* **Chaining.**  An extra card may trigger a *different* token (a wood taken through Jewellery
  triggers Urbanism) — BGG "recursivity" thread, rule-derived — but never the same token twice.
* **Usable the same turn.**  A token gained during the turn applies to every *subsequent* event of
  that turn: because science sets are resolved before construction, Architecture / Engineering /
  Economy gained from a set already apply to the build of the same check; extra-card tokens apply
  to cards placed later in the turn — but **not retroactively** to cards already placed (§12.1).
  BGG thread 3429296 (activation on acquisition) had no retrievable answer; the rulebook's
  "whenever you want" leans yes.
* End-game tokens do nothing during play (§9.2).

---

## 7. Military

### 7.1 Horns and Conflict tokens (**CONFIRMED**)

> "When you take a Red card with 1 or 2 Horn icons, flip over 1 or 2 Conflict tokens respectively
> to their Battle side.  If you must flip over multiple Conflict tokens, but there is only one left
> on its Peace side, only flip over this one token and ignore any extra Horn icons.  When you flip
> the last Conflict token to its Battle side, you trigger a Battle, at the end of your turn, for
> all players."

Engine (`_placed`): `conflict += min(horns, conflict_tokens − conflict)`; when `conflict` reaches
`conflict_tokens` (3), `battle_pending = True`.  Further horn cards taken in the same turn flip
nothing (their shield still counts in the Battle; the card is discarded afterwards like every horn
card).  Horn cards taken through extra picks / effects count exactly like the main card.

### 7.2 Shields

`shields_of(p)` = number of red cards (1 shield each) + `wonder_shields[p]` (Rhodes) + 2 per
Tactics token.  **CONFIRMED** that Rhodes stages and Tactics count in Battles.

### 7.3 Battle resolution with 2 players (**CONFIRMED**, at end of turn)

Rulebook 2-player section: "If you have more Shields than your opponent, take 1 Military Victory
token.  If you have **at least twice as many** Shields as your opponent, take 2 Military Victory
tokens instead of just one."  Ties (including 0 vs 0) award nothing.

`_resolve_battle`, run by `("end_turn",)` after all other actions of the turn, for **both** players
regardless of who triggered the Battle:

| shields (you vs opponent) | tokens | note |
|---|---|---|
| `a > b` and `a >= 2b`, with `b ≥ 1` | 2 | e.g. 2 vs 1, 4 vs 2 |
| `a > b` and `a < 2b` | 1 | e.g. 3 vs 2 |
| `a ≥ 2` vs `b = 0` | 2 | BGA: "if the opponent has 0 Shields and you have 2 or more, you also take 2 tokens" |
| `a = 1` vs `b = 0` | **1** with `double_vs_zero_requires_two = True` (default, BGA ruling, **SINGLE**); 2 with the flag off (literal "1 ≥ 2·0") |
| `a = b` | 0 | |

Then **all** red cards with ≥ 1 horn of **both** players go to `discard` (hornless red cards
stay), `conflict = 0`, `battle_pending = False`.  Each Military Victory token is worth
`military_token_vp = 3` VP.  A Battle triggered on the turn that completes a Wonder is resolved
before the game ends, so its tokens count (§9).

---

## 8. The Cat pawn

### 8.1 Rules (**CONFIRMED**)

* Taking a blue card with a Cat icon moves the Cat to you immediately, from the centre or from the
  opponent (stealing).
* "As long as you have the Cat pawn, at the beginning of your turn, you can secretly look at the
  top card of the central deck before choosing your card."  You are never obliged to take it.
* BGA ruling (**SINGLE**): the Cat cannot be used for additional draws in the turn; only for the
  first draw.
* The Cat is worth **2 VP** to its holder at the end (`cat_vp = 2`; BGA help, FR, DE sources).

### 8.2 Engine model of the private information

The peek is the only private information in the game.  The engine keeps the belief of *one*
observer per state (`state.observer`: `-1` for the environment's omniscient view, `0`/`1` for a
player's belief returned by `Environment.observe(p)`), plus `central_known_to`, a bitmask of the
players who currently know the identity of `deck_top[CENTRAL]`.  `knows_central(p)` is true iff
`p ≥ 0`, the central top is set and bit `p` is on.

* **Peek** (`turn_start`, canonical step 1): if the mover holds the Cat, the central deck is
  non-empty and the mover does not already know its top: if the top is already *set* (known to the
  other player — the Cat changed hands while a peeked card stayed on top) the mover simply learns
  it (`central_known_to |= 1 << mover`, both players now know it); otherwise a `C_PEEK` chance node
  draws the top from `unseen[CENTRAL]` (it is removed from `unseen` and stored in
  `deck_top[CENTRAL]`, `central_known_to = 1 << mover`).  The Environment resolves it with the true
  top card.  No new peek happens while the known card is still on top (it has not changed).
* **Central draw** (`_take(CENTRAL)`):
  * if the top is set and the **mover or the observer** knows it → deterministic: the known card
    is placed, `central_known_to = 0`;
  * if the top is set but is known only to the *other* player and the observer is not that player
    (the observer is the non-holder searching, or the omniscient environment) → the engine
    **forgets** it: the card returns to `unseen[CENTRAL]` and a `C_DRAW_CENTRAL` chance node over
    the full unseen multiset follows;
  * otherwise (nothing known) → `C_DRAW_CENTRAL`.
  In the Environment (`observer = -1`) a non-holder's central draw therefore becomes a chance node
  that the environment resolves with the true top — which is the very card the holder peeked, so
  the belief and the physical deck stay consistent.
* **`observe(p)`** copies the environment state, sets `observer = p` and, if `p` does not know the
  central top, hides it (returns it to `unseen`, clears `central_known_to`).  Consequently the
  feature encoder of the non-holder never sees the peeked card, while the holder's belief carries
  it (and, in the holder's tree, the opponent's central draws are deterministic because
  `knows_central(observer)`).
* **Inside the non-holder's search** the opponent's peek is a *sampled* chance node: the tree
  commits to one sampled top card for the holder's decisions in that branch; if the non-holder
  then draws from the central deck in the same branch the sample is forgotten and re-drawn (the
  non-holder has no information).  This is the one place where the tree is not the exact belief
  tree (DESIGN.md §2.1 accepts it).  Symmetrically, if the observer acquires the Cat inside such a
  branch, the subsequent peek "learns" the sampled card rather than re-sampling (consistent within
  the branch).
* `key()` contains `deck_top`, `unseen` and `central_known_to` but not `observer`; transposition
  tables must therefore be per-observer (one tree per searching player).

### 8.3 Extra draws and the Cat

The engine performs the peek only at `turn_start`.  A central card taken later in the turn through
Ephesus, Alexandria, Architecture or another token is deterministic **only if the peeked card is
still on top** (it is physically the same card); once that card has been taken, every further
central draw in the turn is blind.  This is exactly the BGA ruling; `RulesConfig.cat_peek_main_draw_only`
is a description of this hard-coded behaviour and is **not read by the code** (§10, §14).

---

## 9. End of game, scoring, tie-break, empty decks

### 9.1 End conditions

1. **Wonder complete** (**CONFIRMED**): "Once one player has finished constructing their Wonder,
   the game is over at the end of that player's turn."  Engine: `_build` sets `wonder_done`; the
   rest of the turn is completed (extra picks, science sets, stage effect, Architecture pick — no
   build beyond stage 5), a pending Battle is resolved, then `("end_turn",)` sets `game_over`
   (**INFERRED** ordering Battle-then-end; no source contradicts it).  The other player gets
   **no** equalising turn.  Construction being mandatory, a player cannot delay this.
2. **No card left anywhere** (**ENGINE / open question**): at `turn_start`, if all three decks are
   empty, the game ends immediately (before the peek) — unconditionally; the `end_when_no_cards`
   flag is no longer consulted (its comment argues both settings coincide with 2 players, since
   "the player to move has no card to draw" and "all decks are empty" are the same condition).  If
   decks empty in the middle of a turn, that turn completes normally and the game ends at the next
   `turn_start`.  Rulebook: "If a deck is empty, it remains empty until the end of the game"
   (**CONFIRMED**); what happens when *every* deck is empty was not retrievable (BGG threads
   3102322, 3256390).  While at least one deck has cards, a pick simply offers the non-empty decks.

### 9.2 Scoring (`score_of(p)`, **CONFIRMED** components)

```
score(p) = Σ VP of constructed stages
         + 3 × civ3 + 2 × civ2cat
         + military_token_vp (3) × mil_tokens[p]
         + Strategy: 1 × mil_tokens[p]
         + Education: 2 × (total Progress tokens held, including Education)
         + Decor: 6 if stages[p] == 5 else 4
         + Politics: 1 × civ2cat cards held
         + Culture: 12 if 2 copies held else 4
         + cat_vp (2) if cat == p
```

Leftover grey / yellow / green / red cards, Engineering, Economy, Tactics and the extra-card
tokens score 0.  `score_of` is defined at every node (it is the auxiliary training target via
`score_diff()`); the "final" score is its value at the terminal node.

### 9.3 Winner and tie-break (**CONFIRMED**)

"The player with the highest score wins.  In case of a tie, the player who constructed the most
Stages wins.  If there's still a tie, both players share the victory."  `returns()` gives
`(+1, −1)` / `(−1, +1)` / `(0, 0)` (with `tiebreak_stages = False` an equal score is a draw
regardless of stages).

---

## 10. `RulesConfig` options (`rules.py`)

| option | default | meaning in the engine | confidence / source |
|---|---|---|---|
| `conflict_tokens` | 3 | Conflict tokens in play; horns fill this counter; Battle when full. | **CONFIRMED** (DE brettspielblog / siegpunktsammler / hall9000 cluster + IT Balena Ludens: 3 for 2–3 players). |
| `faceup_progress_tokens` | 3 | Face-up token slots, refilled immediately. | **CONFIRMED** (rulebook setup step 6). |
| `military_token_vp` | 3 | VP per Military Victory token. | **CONFIRMED** (rulebook, BGA). |
| `cat_vp` | 2 | VP for holding the Cat at the end. | **CONFIRMED** (BGA help, FR AccessiJeux, DE reviews). |
| `double_vs_zero_requires_two` | True | 1 shield vs 0 gives 1 token (2+ vs 0 gives 2).  Off: literal "≥ twice" → 1 vs 0 gives 2. | **SINGLE** (BGA gamepanel summary); rulebook wording is just "at least twice as many". |
| `extra_card_optional` | True | Extra-card token picks carry `SKIP`.  Off: the extra draw is forced. | **CONFIRMED** rulebook "you can use"; BGG 3404463. |
| `wonder_effect_optional` | False | Alexandria / Ephesus picks and the Halicarnassus deck choice may be skipped.  Olympia, Babylon and Rhodes are never optional. | Open (BGG 2835875, 3004888 not retrievable); default = mandatory ("immediately benefit"). |
| `economy_once_per_build` | True | **Not consulted by the code.**  Actual behaviour: the doubled coin is available once per **turn** (`econ_used`, reset at `turn_start`), which is the rulebook's "each token once per turn". | **CONFIRMED** behaviour; dead flag (§14). |
| `coins_free_choice` | True | A coin may be spent even when a grey card could be used.  Off: coins only when no grey option exists. | **ENGINE** (rulebook silent; superset of options). |
| `end_when_no_cards` | True | **Not consulted by the code** (since the 2026-09-13 revision).  Actual behaviour: the game always ends at `turn_start` when all decks are empty. | Open question (BGG 3102322 / 3256390); dead flag. |
| `token_usable_same_turn` | True | **Not consulted by the code.**  Actual behaviour: a token gained this turn applies to later events of the same turn (including the build that follows a science set), never retroactively. | Leaning yes (rulebook "whenever you want"); BGG 3429296 open; dead flag. |
| `cat_peek_main_draw_only` | True | **Not consulted by the code.**  Actual behaviour: peek at `turn_start` only; later central draws use the peeked card only while it is still on top. | **SINGLE** (BGA Gamehelp); dead flag. |
| `tiebreak_stages` | True | Equal score → more stages wins; off → draw. | **CONFIRMED** (rulebook). |

Every option is read by `GameState` from `state.rules`; `Environment(rules=...)` passes it through.
Changing a live option changes the game and invalidates checkpoints trained under another.

---

## 11. Confidence and provenance

### 11.1 CONFIRMED (two or more independent sources, or mirrored rulebook text)

Sources are the search-engine summaries listed in `research/core-rules.md §14` and
`research/two-player-and-faq.md §13`: rulespal.com, officialgamerules.org, manuals.plus and
playeraid.net (mirrors of the official EN rulebook text), the BGA game panel / Gamehelp / Tips
pages, BGG threads (2943868 wonder ranking, 2869259 Rhodes, 2757150 military tie, 2758424 Economy,
2773240 / 3517431 Engineering, 2743222 recursivity, 2739970 mandatory build, 2726508 deck info),
BoostYourPlay strategy guide, Board Game Family review, FR AccessiJeux / FR Wikipedia, DE reviews
(brettspielblog.ch, siegpunktsammler.de, hall9000.de, spiele-akademie.de), IT Balena Ludens, ES
studocu rules copy.

* Components: 7 Wonders × 5 stages, 7 × 25-card face-up Wonder decks, 60-card face-down central
  deck, 15 Progress tokens with 3 face up, 6 Conflict tokens (3 in play at 2 players), Military
  Victory tokens = 3 VP, Cat pawn = 2 VP starting unowned.
* Setup: own deck between you and your left neighbour; discard pile never reshuffled.
* Turn: one card from left / right / central; actions in any order; tokens once per turn; extra
  cards only from tokens and effects; empty decks stay empty.
* Cards: five grey resources; yellow = wild coin that must replace missing resources and is
  consumed; blue 3 VP or 2 VP + Cat; green three symbols, 2 identical or 3 different → mandatory
  token, used greens discarded; red shields and 0–2 horns, horn cards discarded after a Battle,
  hornless kept.
* Construction: bottom to top, cost 2–4 identical/different, mandatory (also multiple stages per
  turn, also with yellow cards only), cards discarded, effects immediate.
* Wonder totals 30 / 26 / 25 / 24 / 22 / 22 / 20; effect types; Rhodes effects on stages 2 and 4;
  Olympia 2 effects; Babylon 2 effects; Ephesus effect (count 3 per BGG ranking, one review says 2).
* Tokens: the 15 names and the effects of Architecture, Propaganda, Engineering, Economy, Tactics,
  Strategy, Education, Decor, Politics, Culture; face-up refill immediate; blind draw allowed;
  extra-card tokens optional.
* Military: horn flips with overflow rule, Battle at end of turn for all players, 2-player "more →
  1, at least twice → 2" rule, ties give nothing, discards and Peace reset, Rhodes / Tactics count.
* Cat: acquisition steals, peek every turn at the start, 2 VP.
* End: end of the completing player's turn; scoring components; tie-break by stages then shared.

### 11.2 SINGLE-source items adopted as defaults

BGA "1 vs 0 = 1 token, 2+ vs 0 = 2"; BGA "Cat only for the first draw"; Alexandria and
Halicarnassus effect wording; the youngest-player start (not modelled); BGA "Economy must be
used" (adopted in the affordability test); the cost pattern 2≠ 2= 3≠ 3= 4≠.

### 11.3 ASSUMED (placeholders) and exactly which JSON fields to edit

| Item | Where | What to edit once the physical components are available |
|---|---|---|
| Per-deck card counts (all 7 Wonder decks and the central deck) | `data/decks.json`: `standard_wonder_deck`, `wonder_deck_overrides.<Wonder>` (only the differing kinds need listing), `central_deck` | Set the count of each kind name; each Wonder deck must sum to 25 and the central deck to 60 (`TestData::test_deck_compositions`).  New kinds (e.g. a 2-shield red card) can be added to `kinds` with the next id; `NUM_KINDS` and the action space grow automatically (`HALI_BASE + kind`), so checkpoints must be retrained. |
| Every red card has exactly 1 shield | `data/decks.json` `kinds[*].shields` | Change `shields` on the relevant kind(s) or add kinds. |
| Wonder per-stage cost / kind / VP / effect placement | `data/wonders.json` `wonders[*].stages[*]` | See §5.2. |
| Science token wording | `data/tokens.json` id 5 `text` | Text only (behaviour "extra card on any green card" is the analogy). |
| Urbanism / Crafts / Jewellery resource splits | `data/tokens.json` ids 2, 3, 4 `resources` (list of resource names; `"coin"` = yellow cards) | Edit the lists; `tokens.py` derives `resources` and `triggers_on_coin` from them. |
| Cat start position wording, Progress-token supply exhaustion, Military-token supply, all-decks-empty ending, Halicarnassus reveals being public | code (`state.py`) | Behavioural assumptions; no data field. |

`data/README.md` and the `_status` fields inside each JSON file carry the same tags.

---

## 12. Known modelling approximations

### 12.1 Consequences of the canonical action order

1. **A token acquired during the turn never fires for a card already placed.**  Extra-card
   triggers are evaluated in `_placed`, i.e. when the card arrives, and a token extra pick resolves
   *before* the check of the card that triggered it.  So the Science token gained from the set that
   a green card completed does not give an extra card for that green card; a resource token gained
   from a set does not fire for the extra card that was placed before the check; under "any order"
   a player could sequence these the other way round.  (Construction is not affected: science
   sets are resolved first, so Architecture / Engineering / Economy from a set apply to the build
   of the same check.)
2. **Olympia's two cards are separated by a check.**  `("take", own)`, `("take", opp)` are queued
   in that order, and the first card's `("check",)` runs before the second card is taken; a science
   set or a build (with its payment choice and effects) may therefore occur between the two cards.
   Every mandatory action still happens, but the payment choice for such a build does not see the
   second card.
3. **Extra picks before the check** and **effect before Architecture** are fixed; the rulebook
   allows the reverse.  Since builds and sets are mandatory either way, this only changes what is
   known when the payment / set is chosen (the engine's order maximises information).
4. **Science before construction** is fixed (weakly dominant, §2.4); a player who would rather
   build first and then take a token cannot express that, but nothing reachable that way is lost:
   the token choice is free and Economy's doubled coin is optional at payment time.

### 12.2 Hidden information

5. **The opponent's Cat peek is a sampled chance node inside the non-holder's search** (§8.2):
   the search tree of the non-holder is not the exact belief tree in branches after the opponent
   peeks.  The non-holder's own central draw re-samples ("forgets"), so its value is unbiased; the
   opponent's in-tree play, however, reacts to one sampled card per branch.  The effect is
   negligible because the peek informs only one of three options.
6. Halicarnassus revealed cards are public (§5.3).

### 12.3 Choices and options

7. **Coins as a free choice** (`coins_free_choice`): the engine lets a player burn a coin instead
   of a grey card; the rulebook's "must replace any missing Resource" is silent on this.
8. **Economy counts towards affordability** (mandatory use, BGA behaviour; §4.2).  There is no
   flag for the "optional" reading.
9. **Canonical payment order** (§4.3): payments name cards in code order; this changes only the
   *path*, never the set of reachable multisets.
10. **Player 0 always starts** (§1.3); the rulebook's "youngest player" is a seat assignment
    handled outside the engine.
11. **Unlimited Military Victory tokens** and **science sets only while a token remains** (§6.2).
12. **All decks empty ⇒ game over at the next `turn_start`** (§9.1).

---

## 13. Worked example: one full turn

Real trace from `Environment(wonders=(2, 6), seed=5)` (player 0 = Alexandria, player 1 = Babylon)
with uniformly random actions (`random.Random(5)`), turn 36, player 0 to move.  `describe()`
output has been condensed; the queue shown is `state.queue` at each decision node (the popped
item being handled is not in it).

**State at the start of turn 36** — conflict 0/3, Cat held by P1, face-up tokens Economy /
Decor / Science, stack 9.

| | P0 — Alexandria | P1 — Babylon |
|---|---|---|
| stages | 3/5 (VP 3+4+5 = 12) | 3/5 (VP 3+3+4 = 10) |
| tableau | wood, stone, glass, coin, civ2cat, tablet, gear, shield | civ3, civ2cat×2, shield |
| tokens | — | Tactics, Education, Politics |
| shields / mil tokens | 1 / 2 | 1 + 2 (Tactics) = 3 / 2 |
| score | 12 + 2 (blue) + 6 (mil) = **20** | 10 + 7 (blue) + 6 (mil) + 6 (Education: 3 tokens) + 2 (Politics: 2 icons) + 2 (Cat) = **33** |

Decks: deck 0 (P0's) 7 cards, top `coin`; deck 1 (P1's) 16 cards, top `tablet`; central 50 cards,
top `civ2cat` **known to both players** (`central_known_to = 3`: one player peeked while holding
the Cat, the Cat then changed hands and the new holder's `turn_start` learned the same card, §8.2;
nobody has drawn it since).

1. **`turn_start`** (P0): `tokens_used = 0`, `econ_used = False`; not all decks empty; P0 does
   not hold the Cat → no peek (P0 still knows the central top).  Queue → `pick main`, `end_turn`.
2. **Main pick** — `decision[pick] ctx=('main', False, (0,1,2))`, legal `PICK_LEFT` (deck 0:
   coin), `PICK_RIGHT` (deck 1: tablet), `PICK_CENTER` (civ2cat, known — would be deterministic);
   queue `(end_turn)`.  **Chosen: `PICK_LEFT`.**  `take(0)`: deck 0 → 6 cards, top hidden; queue
   `reveal 0`, `placed coin`, `end_turn`.
3. **Chance `reveal` deck 0** — 5 distinct kinds possible, probabilities = unseen counts / 6; the
   environment resolves it with the true next card: **civ3** (now visible to both).
4. **`placed(coin, main)`** — P0 tableau coin×2.  A yellow card would trigger Jewellery, which
   P0 does not hold.  Queue `check`, `end_turn`.
5. **`check`** — science first: tablet×1, gear×1 → no set.  Construction: Alexandria stage 4
   costs **3 identical**.  P0 holds wood, stone, glass (one each) and coin×2, no Engineering /
   Economy: `_max_value` (identical) = best single resource 1 + 2 coins = 3 ≥ 3 → **mandatory
   build**.  `decision[pay] ctx=()`, legal `pay_wood`, `pay_stone`, `pay_glass` (each: 1 + 2 coins
   = 3).  `pay_coin` is *not* legal first: codes must be non-decreasing, so after a coin only coins
   could follow (1 + 1 = 2 < 3).  **Chosen: `pay_wood`.**  `ctx=(0,)`: need 2, the same-resource
   grey is exhausted → `pay_coin` is the single option → auto-resolved; `ctx=(0,5)`: need 1 →
   `pay_coin` auto-resolved.  `_build`: wood + coin + coin to the discard, `stages[0] = 4`
   (+6 VP → 26), effect `any_deck_card`; no Architecture; queue `pick alexandria` (mandatory),
   `check`, `end_turn`.
6. **Alexandria pick** — `decision[pick] ctx=('alexandria', False, (0,1,2))`, legal `PICK_LEFT`
   (civ3), `PICK_RIGHT` (tablet), `PICK_CENTER` (the known civ2cat).  **Chosen: `PICK_RIGHT`.**
   Chance `reveal` deck 1 (10 kinds) → **shield_h1**.  `placed(tablet, alexandria)`: P0
   tablet×2; a green card would trigger the Science token, not held; queue `check`, `check`,
   `end_turn`.
7. **`check`** — science first: tablet×2 is the only possible set → **auto-resolved**: tablet×2
   to the discard; queue `token_choice science`, `check`, `check`, `end_turn`.
   **`decision[token] ctx=science`**, legal `token_Science`, `token_Economy`, `token_Decor`,
   `token_blind`.  **Chosen: `token_Decor`** (+4 VP while the Wonder is unfinished → 30).  Queue
   `token_reveal`, …; chance `token_reveal` (8 distinct types among the 9 in the stack) →
   **Engineering** fills the slot; face-up now Economy / Science / Engineering, stack 8.
8. **`check` × 2** — no science set (gear×1); stage 5 costs **4 different**; P0 holds stone and
   glass → 2 < 4, not affordable.  Nothing happens.
9. **`end_turn`** — no Battle pending (conflict 0/3), no Wonder complete → `mover = 1`,
   `turn = 37`, queue `turn_start`.  At P1's `turn_start` P1 holds the Cat but already knows the
   central top → no new peek → P1's main pick.

**State at the start of turn 37**: P0 **30** points (stages 3+4+5+6 = 18, blue 2, mil 6, Decor 4),
P1 33; P0 tableau stone, glass, civ2cat, gear, shield, tokens Decor; deck 0: 6 cards top civ3;
deck 1: 15 cards top shield_h1; central 50, top civ2cat still known to both.  Card count check:
110 = tableaus (5 + 4) + unseen (5 + 14 + 49) + visible / known tops (3) + discard (30).

Decisions actually presented to the agent this turn: 4 (main pick, first payment step, Alexandria
pick, token choice); chance nodes resolved by the environment: 3 (two Wonder-deck reveals, one
token refill); auto-resolved: two payment steps and the science set.

---

## 14. Engine defects and discrepancies found while writing this specification

None of these were patched here (`state.py` / `env.py` are owned elsewhere); each comes with a
reproduction and a suggested fix.

1. **Architecture can fire twice in one turn** (rule: once per turn).  Repro (conftest helpers):
   P0 = Alexandria at stage 1 with Architecture, tableau wood×2 + stone + clay, opponent's deck top
   papyrus; run `("check",)`: stage 2 (2 identical) is built → queue = `pick alexandria`,
   `pick token:0`, `check`; the Alexandria pick takes papyrus → stone+clay+papyrus → stage 3
   (3 different) is built **before** the first Architecture pick executes → `_build` queues a
   *second* `pick token:0` (the `tokens_used` bit is only set when the extra card is *placed*) →
   both picks are offered; the second one runs with `tokens_used` already containing Architecture.
   Observed (current engine): two `token:0` picks in one turn.  Expected: one.  Fix: in
   `_handle("pick")`, drop the item when its reason is `token:<t>` and bit `t` of `tokens_used` is
   already set (`return False`), or de-duplicate in `_build`.  (Reachable with any card-drawing
   stage effect: Alexandria, Ephesus, Olympia, Halicarnassus.)
2. **Four dead `RulesConfig` flags**: `economy_once_per_build` (behaviour is once per *turn*; the
   field's comment says "per stage construction"), `end_when_no_cards` (the all-decks-empty check
   is now unconditional), `token_usable_same_turn`, `cat_peek_main_draw_only`.  Either implement
   them (`economy_once_per_build`: reset `econ_used` in `_build`; `token_usable_same_turn=False`:
   record the turn a token was gained and ignore it in `_placed`/`_build`/`_mode`/`_max_value`
   that turn; `cat_peek_main_draw_only=False`: allow a `C_PEEK` before each central extra draw of
   the holder; `end_when_no_cards=False` has no sensible meaning with 2 players) or remove them and
   keep the description in this document.
3. **Economy mandatory for affordability** with no option for the BGG "optional" reading (§4.2).
4. **Docstring claim** "does not change what a player can achieve" is slightly too strong: see
   §12.1 items 1–2.
5. **Data-dependent test expectations**: `TestData::test_wonder_stage_tables` and
   `test_deck_compositions` pin the ASSUMED data (§5.2, §11.3); correcting the JSON requires
   updating them together.
6. **Identical-stage payment can dead-end and silently skip a mandatory build** (introduced with
   the canonical payment order).  `_max_value` in mode 0 (identical) with `used_res` non-empty
   returns `grey_avail[used_res[0]] + coin_cap` *without* honouring `min_code`, so after a grey
   card the engine believes that resource can still be paid even when the next code is a coin
   (after which only coins are allowed).  Repro: Giza at stage 3 (next: **3 identical**), tableau
   wood×2 + coin; `check` → the first `pay_wood` is auto-resolved (only option), so the first real
   decision is `ctx=(0,)` offering `pay_wood` **and** `pay_coin`; choosing `pay_coin` gives
   `ctx=(0, 5)`, need 1, `_pay_options` returns `[]`, `_pay_decision` returns `False` and the turn
   passes to the opponent with `stages` unchanged and all three cards still in hand (verified on
   the current engine) — the mandatory-construction rule is violated, and a search can exploit the
   dead-end to dodge a build.  Fix: in `_max_value`, mode 0 with `used_res`: `(grey_avail[used_res[0]] if
   used_res[0] >= min_code else 0) + coin_cap`; and make `_pay_decision` raise (or build) instead
   of returning `False` when a payment in progress has no completion, so such states cannot pass
   silently.  Documented by the strict xfail `test_identical_stage_coin_after_grey_never_dead_ends`
   (the rest of `tests/test_engine.py` / `tests/test_env.py` passes against the current engine).
