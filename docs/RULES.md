# Rules specification — 2-player *7 Wonders: Architects* as implemented by `sevenwa.engine`

This is the canonical description of the rules the engine implements, written for a programmer
or reviewer auditing `sevenwa/engine/state.py` (the belief-state rules engine), `env.py` (the real
game with hidden information), `rules.py` (`RulesConfig`) and the data files in
`sevenwa/engine/data/`.  It merges the rules research — a first pass
(`research/core-rules.md`, `research/two-player-and-faq.md`) and a second, confirmation pass of
2026-09-13 (`research2/decks.md`, `research2/wonders.md`, `research2/rules-confirm.md`, whose
per-item verdicts are tabulated in `research2/PROPOSAL.md`; the notes live outside the
repository, but the `_status` / `_note` strings inside the JSON files and §11 below carry the
same provenance), both compiled from search-engine summaries of the official rulebook and of
BoardGameArena/BGG pages because direct downloads were blocked — with what the code actually
does.  Where the two disagree, or where the code goes beyond what the sources confirm, this
document says so explicitly.  It describes `state.py` as of 2026-09-13 (the version whose module
docstring is quoted in §2.2: science sets before construction, canonical non-decreasing payment
order, `hali_event`, `tokens_new`, Olympia's two cards placed before the check, the payment
dead-end guard) and `rules.py` with the 14 `RulesConfig` fields listed in §10.

Confidence tags used throughout:

* **CONFIRMED** — same statement relayed from at least two independent sources (or quoted rulebook
  text mirrored by rules sites).
* **SINGLE** — one source only (a single page, or one BGG post relayed more than once); written
  SINGLE-SOURCE in the JSON `_status` strings and in `research2/PROPOSAL.md`.
* **ASSUMED** — not verifiable offline (UNKNOWN in the proposal's scale); an internally consistent
  placeholder (see §11.3 for the exact JSON fields to correct).
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
| Wonder decks | 1 per Wonder, 25 cards, **face up** | decks 0 and 1; composition from `decks.json` (§1.2). Deck sizes **CONFIRMED**; composition partly confirmed (coins, Giza's horned cards, Alexandria's deviations), the rest **ASSUMED**. |
| Central (common) deck | 60 cards, **face down** | deck 2. Size **CONFIRMED**, composition **ASSUMED** (only game-wide proportions constrain it). |
| Discard pile | face up next to the central deck; never reshuffled | `state.discard` (count vector, inert). **CONFIRMED**. |
| Progress tokens | 15: 14 types, Culture ×2; shuffled face-down stack, top **3** face up | `prog_unseen` (per-type counts still in the stack), `prog_stack` (its size), `faceup` (list of type ids). `RulesConfig.faceup_progress_tokens = 3`. **CONFIRMED**. |
| Conflict tokens | **3** Peace-side up (the other 3 of the 6 are removed) | `conflict` = number flipped to the Battle side (0..3); `RulesConfig.conflict_tokens = 3`. Table 3/3/4/5/6/6 for 2..7 players: **CONFIRMED** by DE (brettspielblog.ch / siegpunktsammler.de / hall9000.de: "3 Konfliktmarker bei 2-3 Spielern, 4 bei 4, 5 bei 5, 6 bei 6-7") + IT (Balena Ludens: "3 fino a 3 giocatori") sources; the EN mirrors only say "consult the table", which is an image. |
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

### 1.2 Deck compositions (`data/decks.json`) — sizes and a few counts CONFIRMED, the rest **ASSUMED**

Provenance after the second research pass (2026-09-13, `research2/decks.md`; items D1–D17 in
`research2/PROPOSAL.md`; the `_status` and `wonder_deck_overrides._note` strings in the JSON say
the same):

* **CONFIRMED** — 235 cards = 7 × 25 Wonder decks + 60 central (BGG thread 2726508, BGA game
  panel, retailer component lists).  The 11 non-red kinds (rulebook mirrors, BGA Gamehelp).  Every
  red card is exactly **1 shield with 0, 1 or 2 horns** (Board Game Family: "Every Military card
  holds a shield … Some Military cards also feature a horn or two"; FR reviews: "les cartes rouges
  ont un bouclier chacune"; BGA: "shields and 0–2 horns"; no 2-shield card is reported anywhere).
  The standard Wonder deck holds **3 coins** and Alexandria **4** — two independent sources:
  BoostYourPlay "In the Alexandria deck, there are one more Gold and Gear Science cards than
  normal, but 1 fewer Glass and 1-Horn Military cards" and the description of BGG file 233664
  "Alexandria contains 4 gold and 1/6 you will discover a gold for your opponent".  Each Wonder
  deck differs from the "Standard" configuration by "about 2-3 cards … no real extremes"
  (BoostYourPlay + BGG file 233664).
* **SINGLE** (BoostYourPlay strategy guide unless noted) — Alexandria +1 gear, −1 glass, −1
  one-horn (hence standard glass ≥ 2, gear ≥ 1, one-horn ≥ 1); **Giza "has 2 1-horned Military
  cards and 0 2-horned Military cards"** (applied on 2026-09-13: the earlier placeholder gave Giza
  the standard 2 / 1); game-wide "over 1/3 of the cards … are gray resource cards, and about 12 %
  are Gold cards.  Science, Military, and Civilian cards are close to even at around 17 % each";
  "more Gold+Stone cards than Glass+Papyrus or Wood+Clay"; BGA game panel: "over 11 % of the cards
  let you take control of the cat pawn" (≥ 26 cat cards).
* **ASSUMED** — everything else: the standard wood / stone / clay / papyrus, hornless-red, tablet
  and compass counts; the exact glass, gear and two-horn counts (only bounded, two-horn ≥ 1 is a
  mere inference from Giza being "the deck to avoid"); the blue split; which card replaces Giza's
  missing two-horn card (a hornless red — the minimal change: Giza keeps 4 red cards and 4 shields
  and the game-wide type proportions are unchanged); whether Giza deviates in any other card; every
  deviation of Babylon / Ephesus / Halicarnassus / Olympia / Rhodes (placeholders rotating the
  favoured science symbol; Rhodes = standard); the whole central deck.  The blue split **1 × civ3 /
  3 × civ2cat** per Wonder deck (2026-09-13) is a re-fit of the placeholder so that 7 × 3 + 6 = **27
  cat cards = 11.5 %** of 235 satisfies the BGA "over 11 %" constraint (the earlier 2 / 2 split gave
  20 = 8.5 %, violating it); the split itself is unverified.

The per-deck table, regenerated from the JSON (bold = differs from the standard deck):

| kind | standard Wonder deck | Alexandria | Babylon | Ephesus | Giza | Halicarnassus | Olympia | Rhodes | central |
|---|---|---|---|---|---|---|---|---|---|
| wood | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 |
| stone | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 4 |
| clay | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 |
| papyrus | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 3 |
| glass | 2 | **1** | 2 | 2 | 2 | 2 | 2 | 2 | 3 |
| coin | 3 | **4** | 3 | 3 | 3 | 3 | 3 | 3 | 8 |
| civ3 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 6 |
| civ2cat | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 6 |
| tablet | 2 | 2 | **1** | **1** | 2 | **1** | **1** | 2 | 4 |
| gear | 1 | **2** | **2** | 1 | 1 | 1 | **2** | 1 | 4 |
| compass | 1 | 1 | 1 | **2** | 1 | **2** | 1 | 1 | 4 |
| shield | 1 | 1 | 1 | 1 | **2** | 1 | 1 | 1 | 3 |
| shield_h1 | 2 | **1** | 2 | 2 | 2 | 2 | 2 | 2 | 6 |
| shield_h2 | 1 | 1 | 1 | 1 | **0** | 1 | 1 | 1 | 3 |
| **total** | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 25 | 60 |

Confidence per cell: `coin` (standard 3, Alexandria 4) CONFIRMED; Alexandria's `glass`, `gear`,
`shield_h1` and Giza's `shield_h1` / `shield_h2` SINGLE; every other cell ASSUMED (the standard
`glass` ≥ 2, `gear` ≥ 1 and `shield_h1` ≥ 1 are lower-bounded only).  Game-wide, all 235 cards:
grey 85 (36.2 %), yellow 30 (12.8 %), blue 40 (17.0 %), green 41 (17.4 %), red 39 (16.6 %); cat
cards 27 (11.5 %); gold + stone 48 > wood + clay 34 > glass + papyrus 33 — every SINGLE aggregate
constraint above is satisfied.  The full table exists but was unreachable offline: BGG file 233664
"7 Wonders Architets cards distribution"
(https://boardgamegeek.com/filepage/233664/7-wonders-architets-cards-distribution), most likely
the same table as BGG image 6620199 (https://boardgamegeek.com/image/6620199/7-wonders-architects),
mirrored at https://bgmanual.com/boardgames/1092/7-wonders-architects — see §11.3.

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
>    dominates the alternative.  (A token gained mid-turn never fires for a card placed earlier in
>    the same turn, and Olympia's two cards are placed one after the other with a check in between;
>    these are the only ways the fixed order differs from a fully free ordering.)
> 4. End of turn: battle (if triggered), game end (if a Wonder is complete), next player.
>
> Payments are sequences of "pay one card" decisions (grey resource, coin, or a coin worth 2 with
> Economy).  Codes must be chosen in non-decreasing order (wood < stone < clay < papyrus < glass <
> coin < coin×2), which removes permutations of the same multiset from the tree: a payment
> decision only appears when genuinely different sets of cards could be spent.

The Olympia clause of that parenthesis is stale: since the `olympia_nocheck` revision both Olympia
cards land *before* the check (§5.1, §12.1 item 2, §14 item 8); the first clause still holds.

### 2.3 How the order is realised: the work queue

Turn processing is a small work queue (`state.queue`, a tuple of items).  `_run()` pops the first
item and handles it until a handler reports a *real* choice (a decision with ≥ 2 legal actions or a
chance event with ≥ 2 possible outcomes) or the game ends.  **Forced choices are auto-resolved**
(`_set_decision` applies a single legal action immediately; `_set_chance` applies a single possible
outcome immediately), so the tree contains only genuine decisions.  `_push(*items)` places items at
the **front** of the queue in the order given (the first argument executes first).

| Queue item | Handler behaviour |
|---|---|
| `("turn_start",)` | Reset `tokens_used`, `tokens_new` and `econ_used`.  If every deck is empty → game over (unconditionally, see §9.1).  Prepend `("pick","main",False,(0,1,2))`, `("end_turn",)`.  Cat holder peek (§8). |
| `("pick", reason, optional, sources)` | A `token:<t>` pick whose token has already been used this turn is dropped (once per turn; §14 item 1).  Offer `PICK_*` for every source deck with cards; add `SKIP` if optional.  If no source has cards, the item is dropped silently.  With `cat_peek_main_draw_only = False`, a Cat holder's non-main pick that offers the central deck is preceded by a `C_PEEK` (§8.3). |
| `("take", d, reason)` | Take deck `d`'s top: Wonder deck → `deck_size -= 1`, top hidden, prepend `("reveal", d)`, `("placed", k, d, reason)`; central deck → §8.2. |
| `("reveal", d)` | If deck `d` is non-empty and its top is hidden: chance node `C_REVEAL` over `unseen[d]`. |
| `("placed", k, d, reason)` | Add the card to the mover's tableau; immediate effects (Cat, horns); mark the triggering token used if `reason == "token:<t>"`; prepend `("check",)` (skipped for the first Olympia card, reason `"olympia_nocheck"`: the check runs once both cards have landed), then (in front of it) one `("pick", "token:<t>", extra_card_optional, (0,1,2))` for each held, active (`_token_active`, §6.3) token the card triggers. |
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
the turn cannot fire for a card that was already placed).  This is a second-order effect; the
design accepts it for tree size.

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
| **Red — Military** | Each red card shows a shield and 0, 1 or 2 War-Horn icons.  Horns flip Conflict tokens (§7).  Red cards **with** horns are discarded after the next Battle; red cards **without** horns are kept for the whole game. | Every red kind has exactly **1 shield** — **CONFIRMED** (Board Game Family "Every Military card holds a shield … Some Military cards also feature a horn or two"; FR reviews "un bouclier chacune"; BGA Gamehelp "shields and 0–2 horns"; no source mentions a 2-shield card).  `shield` (0 horns), `shield_h1`, `shield_h2`.  Horn cards trigger Propaganda. |

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
* with **Economy**, one yellow card per turn (per stage with `economy_once_per_build = True`) may
  be worth **2 coins** (`PAY_COIN2`), see §6.3 — in a *different* stage the doubled coin stands for
  two distinct missing resources;
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
With `economy_forces_build = True` (default) this counts the Economy doubling, i.e. **a stage that
can only be completed thanks to Economy is mandatory** — the BoardGameArena behaviour (bug #64468
developer reply: "players must use the economic token to double their coin") rather than the
"you can use" reading of BGG thread 2758424 (relayed reply: "you are not forced to").  With the
flag off, `_affordable` treats Economy as already used, so the doubled coin never *forces* a
build, but `PAY_COIN2` is still offered once the stage is affordable without it (e.g. 2 coins +
Economy for "2 identical" builds, with the choice `PAY_COIN` / `PAY_COIN2`; 1 coin + Economy does
not build).  The rulebook words Economy as a static value ("is worth 2 Coins") but tokens as
optional ("you can use"); no official ruling — **UNKNOWN** (§10, §11.3).

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
`stages[p] += 1`, `econ_used = True` if a doubled coin was used (cleared again at once when
`economy_once_per_build = True`), `wonder_done = True` if the 5th stage was built, then
`("check",)`, the Architecture pick (if held and active) and the stage effect are queued (effect
first, Architecture second, check last).  Steps with a single legal action are auto-resolved, so
e.g. "2 coins for a 2-identical stage" is built without any decision node.

`RulesConfig.coins_free_choice = True` (**ENGINE**): the player may spend a coin even when a grey
card would do ("These Coins must replace any missing Resource" does not say you may not spend a
coin *instead of* an available grey card; allowing it is the superset).  With the flag off, coins
are legal only when no grey option is.

Dead-end guard: in identical mode `_max_value` counts the committed resource only while its code
is still `≥ min_code` (`(grey_avail[r] if r >= min_code else 0) + coin_cap`), and `_pay_decision`
raises `RuntimeError` if a payment in progress has no completion — a payment can no longer
dead-end and silently skip a mandatory build (§14 item 6, fixed;
`test_identical_stage_coin_after_grey_never_dead_ends` passes and is no longer an xfail).

---

## 5. The Wonders

### 5.1 Effects (`wonders.py`)

| effect id | Wonder | Rule text (as relayed) | Engine (`_build`) |
|---|---|---|---|
| `none` | Giza (all), others' plain stages | — | nothing |
| `shield` | Rhodes | "Add 1 Shield to your total Shields" — permanent, survives Battles.  Wording **CONFIRMED**; 2 stages **CONFIRMED** (BGG 2943868 "26 VP 2 shields" + Board Game Family + BGG 2869259) | `wonder_shields[p] += 1` |
| `progress_token` | Babylon | "Choose 1 Progress token from the 4 available": one of the 3 face-up or the top of the stack.  Wording **CONFIRMED**; 2 stages **CONFIRMED** (BGG 2943868 "20 VP 2 tokens" + Board Game Family) | `("token_choice","babylon")`; fizzles if no token is left.  Never optional. |
| `any_deck_card` | Alexandria | "Take the top card from any deck anywhere on the table and place it in front of you."  Wording **CONFIRMED** (rulebook text mirrored by officialgamerules / rulespal / manuals.plus / playeraid); 2 stages **SINGLE** (BGG 2943868 "25 VP 2 cards") | `("pick","alexandria", wonder_effect_optional, (0,1,2))` |
| `central_card` | Ephesus | "Take the top card from the central deck and place it in front of you."  Wording **CONFIRMED**; **3** stages **CONFIRMED** (BGG 2943868 "22 VP 3 cards" + Board Game Family "the 3 columns in Ephesus"; dissent: one review — the Meeple Mountain / Boardgameshots result set — says the temple "twice grants" the card) | `("pick","ephesus", wonder_effect_optional, (2,))` — auto-resolved (forced) when not optional; blind unless the Cat-peeked card is still on top (§8) |
| `left_and_right_cards` | Olympia | "Take the top card from the decks to your left and your right and place them in front of you."  Wording **CONFIRMED**; 2 stages (= "4 cards in all the game") **SINGLE** (BGG 2943868 "22 VP 4 cards") | `("take", own deck, "olympia_nocheck")`, `("take", opponent's deck, "olympia")` — no choice, never optional, an empty deck is skipped; both cards land before the mandatory check (§12.1) |
| `look_5_choose_1` | Halicarnassus | "Take the top 5 cards from the deck to your left or your right.  Choose 1 and place it in front of you.  Shuffle the other cards back into their deck."  Wording **CONFIRMED** (rulebook mirrors + FR "mélangez les cartes restantes dans leur pioche"); 2 stages **SINGLE** (BGG 2943868 "24 VP 2 cards") | §5.3 |

Cards taken through an effect are placed like any other card: they trigger tokens, Cat, horns and
the science / construction check (§2.3).  With `wonder_effect_optional = True` the Alexandria /
Ephesus picks and the Halicarnassus deck choice gain a `SKIP` action (BGG thread 2835875 "Is it
mandatory to use a Wonder's effect?" — answer not retrievable; default **mandatory**).

### 5.2 Per-stage data currently encoded (`data/wonders.json`)

> **PROMINENT WARNING.**  For every Wonder the **effect type** and its wording are **CONFIRMED**
> (rulebook mirrors).  The **number of effect stages** is **CONFIRMED** for Rhodes (2), Babylon (2)
> and Giza (0) (BGG thread 2943868 + Board Game Family review) and for Ephesus (3: BGG 2943868 +
> Board Game Family, against one review saying "twice"), but only **SINGLE** for Olympia (2, i.e.
> 4 cards), Alexandria (2) and Halicarnassus (2).  The **total VP** (30/26/25/24/22/22/20) are
> **SINGLE-SOURCE**: one post in BGG thread 2943868 "Wonders. Which is the best?" by a 300+-play
> BGA player, relayed identically in two queries; no contradicting figure was found in 11
> languages, but the BGA game page prints no totals (the earlier "CONFIRMED (BGG/BGA)" label was
> wrong).  The **per-stage costs**, the **per-stage VP split** and **which stages carry the
> effects** are **ASSUMED**: the *set* of costs (2–4 resources, identical or different; "the first
> stage of most wonders" is 2 different, "the upper levels generally require a set of four") is
> **CONFIRMED** by FR / DE / IT / JA / EN reviews, but the order 2≠ 2= 3≠ 3= 4≠ used for every
> Wonder is a placeholder; effects were placed on stages 2 and 4 by analogy with Rhodes (the only
> Wonder whose effect stages, 2 and 4, have a source — BGG thread 2869259, **SINGLE**), Ephesus's
> three effects on stages 1, 3, 5, and Olympia's first effect is at least "early" (stage 1 or 2 —
> BGG 2943868; stage 2 is consistent).  VP splits are monotone placeholders summing to the totals.
> **Giza may not share the cost pattern**: the official site's marketing copy says "The Pyramid's
> stable structure allows you to build with fewer matching resource cards", i.e. Giza probably has
> *fewer* "identical" stages than the others — not encoded.  Verify against the physical trays
> (rulebook page 6) before trusting any Wonder-specific strength conclusion.

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
   (`_copy` resets it), so it is not part of `key()`.  The rulebook wording "Shuffle the other
   cards back into their deck" is **CONFIRMED** (EN mirrors + FR "mélangez les cartes restantes dans
   leur pioche"); that this re-randomises the *whole* deck rather than returning the four cards to
   the top in some order is the engine's reading — no source addresses it (**UNKNOWN**, §11.3).

**ENGINE:** the revealed cards are stored in the public decision context, i.e. they are treated as
visible to both players (the rulebook does not say whether the opponent sees them).

---

## 6. Science and Progress tokens

### 6.1 The 15 tokens (`data/tokens.json`)

| id | name | kind | effect as implemented | timing | confidence |
|---|---|---|---|---|---|
| 0 | Architecture | `extra_card_on_build` | When you construct a stage, choose 1 extra card from the 3 available. | once per turn, optional | **CONFIRMED** |
| 1 | Propaganda | `extra_card_on_horn` | When you take a red card with 1 or 2 horns, choose 1 extra card. | once per turn, optional | **CONFIRMED** |
| 2 | Urbanism | `extra_card_on_resource` (wood, clay) | When you take a Wood or Clay grey card, choose 1 extra card. | once per turn, optional | **CONFIRMED** (rulespal rulebook mirror names the resources; BoostYourPlay "more Gold+Stone cards than Glass+Papyrus or Wood+Clay" corroborates the three splits) |
| 3 | Crafts | `extra_card_on_resource` (papyrus, glass) | When you take a Papyrus or Glass grey card, choose 1 extra card. | once per turn, optional | **CONFIRMED** (same two sources) |
| 4 | Jewellery | `extra_card_on_resource` (stone, coin) | When you take a Stone grey card **or a Yellow card**, choose 1 extra card. | once per turn, optional | **CONFIRMED** (same two sources) |
| 5 | Science | `extra_card_on_green` | When you take a Green card, choose 1 extra card. | once per turn, optional | **CONFIRMED** (rulebook mirror officialgamerules / rulespal: "When you take a Green card, choose 1 extra card from the 3 available and place it in front of you" + BGG thread 2767925 "a progress token that allows you to take an extra card if you draft a green card") |
| 6 | Engineering | `engineering` | Construct stages with any resources, ignoring identical/different. | passive | **CONFIRMED** |
| 7 | Economy | `economy` | One of your yellow cards is worth 2 coins. | once per turn (§6.3; per stage with `economy_once_per_build`), optional at payment time but counted for affordability by default (`economy_forces_build`, §4.2) | text **CONFIRMED**; interaction with the mandatory build **UNKNOWN** |
| 8 | Tactics | `shields` (+2) | Add 2 shields to your total. | passive | **CONFIRMED** |
| 9 | Strategy | `vp_per_military_token` (1) | End: 1 VP per Military Victory token. | end-game | **CONFIRMED** |
| 10 | Education | `vp_per_progress_token` (2) | End: 2 VP per Progress token you have, including this one. | end-game | **CONFIRMED** |
| 11 | Decor | `vp_wonder` (4 / 6) | End: 4 VP if your Wonder is unfinished, 6 VP if complete. | end-game | **CONFIRMED** |
| 12 | Politics | `vp_per_cat_icon` (1) | End: 1 VP per Cat icon on your blue cards (icons, not the pawn). | end-game | **CONFIRMED** |
| 13 | Culture (×2) | `culture` (4 / 12) | End: 4 VP with one Culture token, 12 VP with both. | end-game | **CONFIRMED** |

13 types + 1 duplicate = 15 tokens — **CONFIRMED** (rulebook setup "Shuffle the Progress tokens
… Take the top 3"; every name found individually; Culture: "There are 2 copies of this token").

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
  is spent, reset at `turn_start`), i.e. **once per turn** across all stages built that turn
  (`economy_once_per_build = False`, the rulebook's "each Progress token … once per turn"); with
  `economy_once_per_build = True` (house rule) `_build` clears `econ_used` after every stage, so
  each stage built in the turn may double one coin.
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
  This is `token_usable_same_turn = True`.  With `False`, `tokens_new` (bitmask of the tokens
  gained this turn, reset at `turn_start`) makes `_token_active` ignore them until the next turn —
  but only for the six extra-card tokens: Engineering and Economy are read directly from `tokens`
  by `_mode` / `_max_value` / `_pay_options` and still apply to the same turn's builds (§14
  item 7).  BGG thread 3429296 (activation on acquisition) had no retrievable answer; the
  rulebook's "whenever you want" leans yes (**UNKNOWN**).
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
`conflict_tokens` (3), `battle_pending = True`.  The value 3 for 2 players is **CONFIRMED** (§1.1:
DE "3 Konfliktmarker bei 2-3 Spielern …" + IT "3 fino a 3 giocatori"; the rulebook's own table is
an image and never appeared in an EN relay).  Further horn cards taken in the same turn flip
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
| `a ≥ 2` vs `b = 0` | 2 | BGA: "if the opponent has 0 Shields and you have 2 or more, you take 2 Military Victory tokens (6 points)" |
| `a = 1` vs `b = 0` | **1** | with `double_vs_zero_requires_two = True` (default — **CONFIRMED**: BGA game panel + Gamehelp, and a FR rules page "si votre adversaire n'a aucun bouclier, si vous en avez au moins 2, vous prenez 2 jetons Victoire militaire"; the EN mirrors only say "at least twice as many"); 2 with the flag off (literal "1 ≥ 2·0") |
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
* The Cat is worth **2 VP** to its holder at the end (`cat_vp = 2` — **CONFIRMED**: Board Game
  Family "The cat token is also worth 2 points if you have it at the end of the game", BGA "the
  cat pawn (2 pts)", FR AccessiJeux / videoregles.net scoring order).

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
central draw in the turn is blind.  This is exactly the BGA ruling (`cat_peek_main_draw_only =
True`).  With the flag off (house rule) `_handle("pick")` inserts a `C_PEEK` before every non-main
pick that offers the central deck to a Cat holder who does not yet know its top (the pick item is
re-queued behind the peek), so each extra central draw of the holder — Ephesus, Alexandria, token
picks — is informed.

---

## 9. End of game, scoring, tie-break, empty decks

### 9.1 End conditions

1. **Wonder complete** (**CONFIRMED**, EN mirrors + FR videoregles.net / AccessiJeux "la partie se
   termine à la fin du tour du joueur qui a terminé sa Merveille"): "Once one player has finished
   constructing their Wonder, the game is over at the end of that player's turn."  Engine: `_build` sets `wonder_done`; the
   rest of the turn is completed (extra picks, science sets, stage effect, Architecture pick — no
   build beyond stage 5), a pending Battle is resolved, then `("end_turn",)` sets `game_over`
   (**INFERRED** ordering Battle-then-end; no source contradicts it).  The other player gets
   **no** equalising turn.  Construction being mandatory, a player cannot delay this.
2. **No card left anywhere** (**ENGINE / open question**): at `turn_start`, if all three decks are
   empty, the game ends immediately (before the peek) — unconditionally; the `end_when_no_cards`
   flag is no longer consulted (its comment argues both settings coincide with 2 players, since
   "the player to move has no card to draw" and "all decks are empty" are the same condition).  If
   decks empty in the middle of a turn, that turn completes normally and the game ends at the next
   `turn_start`.  Rulebook: "If a deck is empty, it remains empty until the end of the game" and
   "When you cannot take any more actions, your turn is over" (**CONFIRMED** text); whether a player
   with no reachable card *passes* or the game *ends* was never answered (BGG threads 2781153,
   3102322, 3256390 — **UNKNOWN**).  With 2 players the three decks (110 cards) exhaust together,
   so ending is the only non-stalling choice.  While at least one deck has cards, a pick simply
   offers the non-empty decks.

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

The 14 fields of `RulesConfig`, in `rules.py` order (frozen dataclass; verified against the code on
2026-09-13):

| option | default | meaning in the engine | confidence / source |
|---|---|---|---|
| `conflict_tokens` | 3 | Conflict tokens in play; horns fill this counter (`_placed`); Battle when full. | **CONFIRMED** (DE brettspielblog / siegpunktsammler / hall9000: "3 Konfliktmarker bei 2-3 Spielern"; IT Balena Ludens: "3 fino a 3 giocatori"). |
| `faceup_progress_tokens` | 3 | Face-up token slots (`token_reveal` × 3 at setup; a taken face-up token is replaced immediately). | **CONFIRMED** (rulebook setup). |
| `military_token_vp` | 3 | VP per Military Victory token (`score_of`). | **CONFIRMED** (rulebook, BGA "(3 points)"). |
| `cat_vp` | 2 | VP for holding the Cat at the end (`score_of`). | **CONFIRMED** (Board Game Family, BGA game panel, FR AccessiJeux / videoregles). |
| `double_vs_zero_requires_two` | True | `_resolve_battle`: 1 shield vs 0 gives 1 token (2+ vs 0 gives 2).  Off: literal "≥ twice" → 1 vs 0 gives 2. | **CONFIRMED** (BGA game panel + Gamehelp; a FR rules page); the EN rulebook wording is just "at least twice as many". |
| `extra_card_optional` | True | Extra-card token picks (`_placed`, `_build`) carry `SKIP`.  Off: the extra draw is forced. | **CONFIRMED** rulebook "you can use"; BGG 3404463 (question only). |
| `wonder_effect_optional` | False | Alexandria / Ephesus picks and the Halicarnassus deck choice may be skipped.  Olympia, Babylon and Rhodes are never optional. | **UNKNOWN** (BGG 2835875, 3004888 not retrievable); default = mandatory ("immediately benefit"). |
| `economy_once_per_build` | **False** | False = the doubled coin is available once per **turn** (`econ_used`, reset at `turn_start`).  True (house rule) = `_build` clears `econ_used` after every stage, so each stage built in the turn may double one coin. | Default **CONFIRMED** (rulebook "each Progress token … once per turn"); True is a house rule. |
| `economy_forces_build` | True | `_affordable`: the doubled coin counts towards the mandatory-construction check (BGA reading).  Off: `_affordable` treats Economy as already used, so it never forces a build; `PAY_COIN2` is still offered once the stage is affordable without it (§4.2). | **UNKNOWN** (community split: BGA bug #64468 "must use"; BGG 2758424 relayed reply "not forced"). |
| `coins_free_choice` | True | `_pay_options`: a coin may be spent even when a grey card could be used.  Off: coins only when no grey option exists. | **ENGINE** (rulebook silent; superset of options). |
| `end_when_no_cards` | True | **No effect** (kept for configuration compatibility).  The game always ends at `turn_start` when all three decks are empty; with 2 players "the mover cannot draw" ≡ "all decks are empty", so both settings coincide (`test_all_decks_empty_ends_the_game_even_with_the_rule_disabled`). | Rule text **CONFIRMED** (decks never refill; "when you cannot take any more actions, your turn is over"); pass-vs-end **UNKNOWN** (BGG 2781153 / 3102322 / 3256390 unanswered). |
| `token_usable_same_turn` | True | True = a token gained this turn applies to later events of the same turn (including the build that follows a science set), never retroactively.  False = `_token_active` ignores tokens in `tokens_new` (gained this turn, reset at `turn_start`) — covers the six extra-card tokens only; Engineering / Economy gained this turn still apply to this turn's builds (§14 item 7). | **UNKNOWN** — leaning yes (rulebook "whenever you want"); BGG 3429296 answer not retrievable. |
| `cat_peek_main_draw_only` | True | True = peek at `turn_start` only; later central draws use the peeked card only while it is still on top.  False (house rule) = `_handle("pick")` also peeks (`C_PEEK`) before every extra central pick of a Cat holder whose top is unknown. | **SINGLE** (BGA Gamehelp: the Cat only informs the first draw). |
| `tiebreak_stages` | True | `returns()`: equal score → more stages wins; off → draw. | **CONFIRMED** (rulebook). |

Every option except `end_when_no_cards` is read by `GameState` from `state.rules`;
`Environment(rules=...)` passes it through.  Changing a live option changes the game and invalidates
checkpoints trained under another.

---

## 11. Confidence and provenance

Two research passes, both search-engine relays (the official PDF and the BGG attachments were
never fetched).  The first pass (`research/core-rules.md`, `research/two-player-and-faq.md`)
established the rule text; the second pass of 2026-09-13 (`research2/decks.md`,
`research2/wonders.md`, `research2/rules-confirm.md`) re-examined every value in the three JSON
files plus the rule constants and tabulated them as D1–D17 (decks), W1–W15 (wonders), T1–T6
(tokens) and R1–R9 (rules) in `research2/PROPOSAL.md`: **21 CONFIRMED, 12 SINGLE-SOURCE,
14 UNKNOWN** (an item whose *wording* is confirmed but whose value or behaviour is not — D13,
W14, T4, R6, R7 — is counted under the weaker verdict).  Only one game value had to change (Giza:
0 two-horn cards, D12) and one placeholder was re-fitted (the blue split, D16); everything else
was either confirmed as-is or left as an unverifiable placeholder.  The `_status` / `_note`
strings in `decks.json`, `wonders.json` and `tokens.json` repeat these verdicts; the research
notes themselves live outside the repository.

Sources: rulespal.com, officialgamerules.org, manuals.plus and playeraid.net (mirrors of the
official EN rulebook text), the BGA game panel / Gamehelp / Tips pages, BGG threads (2943868
wonder ranking, 2869259 Rhodes, 2767925 Science token, 2757150 military tie, 2758424 Economy,
2773240 / 3517431 Engineering, 2743222 recursivity, 2739970 mandatory build, 2726508 deck info,
2856705 Ephesus, 3263034 Giza), BGG file 233664 (description only), BoostYourPlay strategy guide,
Board Game Family review, Meeple Mountain / Boardgameshots reviews, the official site
7wondersarchitects.com, FR AccessiJeux / videoregles.net / akoatujou.fr / undecent.fr / FR
Wikipedia, DE reviews (brettspielblog.ch, siegpunktsammler.de, hall9000.de, spiele-akademie.de),
IT Balena Ludens, JA bodolog.com, ES studocu rules copy.

### 11.1 CONFIRMED (two or more independent sources, or mirrored rulebook text) — 21 items

Data (`decks.json`):

* D1 — 235 cards = 7 × 25 Wonder decks + 60 central (BGG 2726508, BGA game panel, retailer lists).
* D2 — the 11 non-red kinds: 5 grey resources, yellow coin, blue 3 VP / 2 VP + Cat, 3 green symbols
  (rulebook mirrors, BGA Gamehelp).
* D3 — every red card = exactly 1 shield with 0, 1 or 2 horns; no 2-shield card exists (Board Game
  Family, FR reviews, BGA Gamehelp, rulebook "1 or 2 Horn icons are shown on certain Red cards").
* D4 — the standard Wonder deck holds 3 coins (BoostYourPlay "one more Gold … than normal" in
  Alexandria + BGG file 233664 "Alexandria contains 4 gold").
* D10 — Alexandria holds 4 coins (same two sources).

Data (`wonders.json`):

* W1 — 7 Wonders × 5 stages (rulebook).
* W3 — the effect wording of all seven Wonders (rulebook text mirrored by officialgamerules /
  rulespal / manuals.plus / playeraid; FR relay for Halicarnassus).  Alexandria and Halicarnassus
  were tagged SINGLE before this pass.
* W4 — Rhodes: 2 effect stages (BGG 2943868 "2 shields", Board Game Family, BGG 2869259).
* W5 — Babylon: 2 effect stages (BGG 2943868 "2 tokens", Board Game Family).
* W6 — Ephesus: 3 effect stages (BGG 2943868 "3 cards", Board Game Family "the 3 columns"); one
  dissenting review says "twice".
* W9 — Giza: no effect, more VP (rulebook; DE hall9000 "every wonder except the pyramids").

Data (`tokens.json`):

* T1 — 15 tokens = 14 types with Culture × 2 (rulebook setup; every name found; "There are 2
  copies of this token").
* T2 — Science: "When you take a Green card, choose 1 extra card from the 3 available and place it
  in front of you" (rulebook mirrors + BGG 2767925).  Was SINGLE.
* T3 — Urbanism = Wood / Clay, Crafts = Papyrus / Glass, Jewellery = Stone / Yellow (rulespal +
  BoostYourPlay).
* T5 — Tactics = +2 shields (rulebook mirrors; BGA Pythia "two shields").
* T6 — Architecture, Propaganda, Engineering, Strategy, Education, Decor, Politics, Culture texts
  and values (rulebook mirrors).

Rules (`rules.py` / `state.py`):

* R1 — 3 Conflict tokens at 2 players, table 3/3/4/5/6/6 (DE + IT).
* R2 — 2-player Battle: 1 vs 0 → 1 token, ≥ 2 vs 0 → 2 tokens, otherwise "more → 1, at least twice
  → 2" (BGA game panel + Gamehelp; FR rules page; EN mirrors give only the "twice" rule).  Was
  SINGLE.
* R3 — Cat pawn = 2 VP (Board Game Family, BGA, FR).
* R4 — Military Victory token = 3 VP (rulebook, BGA).
* R8 — the game ends at the end of the completing player's turn, after the Battle (EN mirrors + FR
  videoregles / AccessiJeux).

First-pass rule text that remains CONFIRMED (unchanged): setup (own deck between you and your left
neighbour, discard pile never reshuffled, 3 face-up tokens); the turn (one card from left / right /
central, actions in any order, tokens once per turn, extra cards only from tokens and effects,
empty decks stay empty); the card rules of §3; construction (bottom to top, cost 2–4 identical /
different, mandatory — also several stages per turn, also with yellow cards only —, cards
discarded, effects immediate); the military rules of §7 (horn overflow, Battle at end of turn for
all players, ties give nothing, horn cards discarded, Peace reset, Rhodes / Tactics count); the Cat
rules of §8 (stealing, peek at the start of every turn); scoring components; tie-break by stages
then shared victory.

### 11.2 SINGLE-SOURCE items adopted as defaults — 12 items

* D5 / D6 / D7 — standard glass ≥ 2, gear ≥ 1, one-horn ≥ 1 (BoostYourPlay's Alexandria sentence);
  Giza has exactly 2 one-horn cards, so the standard 2 is consistent.  Encoded: 2 / 1 / 2.
* D11 — Alexandria: gear 2, glass 1, one-horn 1 (BoostYourPlay).
* D12 — Giza: 2 one-horn, 0 two-horn (BoostYourPlay; the same page's Alexandria statement was
  independently confirmed, and the change is one card inside the confirmed "2-3 cards different"
  envelope).  Applied 2026-09-13.
* D15 — game-wide proportions "> 1/3 grey, ~12 % gold, ~17 % each blue / green / red"
  (BoostYourPlay) — satisfied by the table (36.2 / 12.8 / 17.0 / 17.4 / 16.6 %).
* D17 — more gold + stone than glass + papyrus or wood + clay (BoostYourPlay) — satisfied (48 vs 33
  vs 34).
* W2 — total VP 30 / 26 / 25 / 24 / 22 / 22 / 20 (BGG thread 2943868, one post by a 300+-play BGA
  player relayed identically twice; no contradicting figure in 11 languages; BGA prints no totals).
  Was mislabelled "CONFIRMED (BGG/BGA)".
* W7 — Olympia: 2 effect stages = "4 cards in all the game" (BGG 2943868).
* W8 — Alexandria 2 and Halicarnassus 2 effect stages (BGG 2943868).
* W10 — Rhodes effects on stages 2 and 4 (BGG 2869259: the shield "goes away" while the 3rd piece
  is built and returns with the 4th).
* W11 — Olympia's first effect is "early" (stage 1 or 2; BGG 2943868) — stage 2 encoded.

First-pass SINGLE items still in force: BGA "the Cat only informs the first draw"
(`cat_peek_main_draw_only`); the youngest-player start (not modelled); the Cat's start-position
wording.  Note that BGA "Economy must be used" is no longer a default adopted on one source alone:
it is exposed as `economy_forces_build` and counted UNKNOWN below.

### 11.3 UNKNOWN → ASSUMED — 14 items, and exactly which JSON fields to edit

* D8 — standard two-horn count (1 encoded; only "≥ 1" is inferable from Giza being singled out).
* D9 — standard wood / stone / clay / papyrus (2 each), hornless red (1), tablet / compass (2 / 1).
* D13 — which cards make up the "2-3 different" of Babylon, Ephesus, Halicarnassus, Olympia and
  Rhodes (science-rotation placeholders; Rhodes = standard); whether Giza deviates in any other
  card; which card replaces Giza's two-horn card (hornless red chosen).
* D14 — the whole 60-card central deck (16 grey / 8 coin / 6 + 6 blue / 4 + 4 + 4 green / 3 + 6 + 3
  red placeholder; only the aggregate percentages constrain it).
* D16 — the blue split per Wonder deck (1 civ3 / 3 civ2cat, re-fitted to satisfy BGA's "over 11 %
  cat cards", itself a single source).
* W12 — the effect stages of Alexandria (2 & 4), Halicarnassus (2 & 4), Olympia (2 & 4), Babylon
  (2 & 4), Ephesus (1, 3, 5).
* W13 — the per-stage VP split of all seven Wonders (totals kept).
* W14 — the per-stage cost order (2≠ 2= 3≠ 3= 4≠ for every Wonder; the set is confirmed).
* W15 — Giza's cost pattern ("build with fewer matching resource cards", official site).
* T4 / R5 — whether Economy's doubled coin counts towards the mandatory build
  (`economy_forces_build`; BGA yes, BGG 2758424 no).
* R6 — Halicarnassus "shuffle the other cards back": whole-deck reshuffle (engine, `env.py`) vs.
  "on top in any order".
* R7 — all-decks-empty: the engine ends the game; the rulebook only says decks stay empty and a
  player with no action ends the turn.
* R9 — same-turn use of a just-acquired token (`token_usable_same_turn`).

| Item | Where | What to edit once the physical components / the BGG table are available |
|---|---|---|
| Standard Wonder deck (D8, D9, D16, and the exact D5 / D6 values) | `data/decks.json` `standard_wonder_deck`: `wood`, `stone`, `clay`, `papyrus`, `shield`, `tablet`, `compass`, `shield_h2`, `civ3`, `civ2cat` are pure placeholders; `glass` (≥ 2) and `gear` (≥ 1) are bounded; `coin` = 3 and `shield_h1` = 2 are source-backed | Set each kind's count; the deck must sum to 25 (`TestData::test_deck_compositions`).  New kinds (e.g. a 2-shield red card, should one exist after all) are added to `kinds` with the next id; `NUM_KINDS` and the action space grow automatically (`HALI_BASE + kind`), so checkpoints must be retrained. |
| Per-Wonder deviations (D13) | `data/decks.json` `wonder_deck_overrides.Babylon` / `Ephesus` / `Halicarnassus` / `Olympia` / `Rhodes` (entire entries are placeholders); `Giza.shield` (the replacement card) and any further Giza deviation; `Alexandria` is source-backed | List only the kinds that differ from the standard deck; each Wonder deck must sum to 25. |
| Central deck (D14) | `data/decks.json` `central_deck` (all 14 counts) | Must sum to 60. |
| Wonder stage tables (W12, W13, W14, W15; W6 if Ephesus turns out to have 2) | `data/wonders.json` `wonders[*].stages[*]`: `vp` (all), `cost` / `kind` order (all; Giza probably fewer `identical`), `effect` placement for Alexandria / Halicarnassus / Olympia / Ephesus / Babylon (Rhodes `[1, 3]` is SINGLE) | See §5.2; update `TestData::test_wonder_stage_tables` together with the data. |
| `tokens.json` | nothing remains ASSUMED (Science wording and the three resource splits are CONFIRMED); the Economy question is a rule flag | — |
| Behavioural (code) | `rules.py`: `economy_forces_build` (T4 / R5), `token_usable_same_turn` (R9; partial, §14 item 7), `wonder_effect_optional`; `state.py` / `env.py`: all-decks-empty ending (R7), whole-deck Halicarnassus reshuffle (R6), Halicarnassus reveals public, Military-token / Progress-token supply, Cat start wording, youngest player | Flags where they exist; otherwise code. |

Primary sources that would settle the ASSUMED items but were unreachable offline (every fetch was
denied by the proxy; search engines index none of them as text) — anyone with normal internet
access can fill the table in from these:

* The per-deck card table: BGG file 233664 "7 Wonders Architets cards distribution"
  (https://boardgamegeek.com/filepage/233664/7-wonders-architets-cards-distribution, 2022-01-07),
  very likely identical to BGG image 6620199
  (https://boardgamegeek.com/image/6620199/7-wonders-architects), mirrored at
  https://bgmanual.com/boardgames/1092/7-wonders-architects; related threads BGG 2726508 and the
  BGA forum "Card breakdown" (viewtopic.php?f=510&p=106592) / "Card distribution"
  (viewtopic.php?t=32189).
* The Wonder trays (per-stage cost, VP, effect stages): the official EN rulebook PDF, page 6 —
  https://cdn.svc.asmodee.net/production-rprod/storage/downloads/games/7wonders-architects/Rules/rules-architects-en-1639490431DeqhF.pdf
  (mirror: https://cdn.1j1ju.com/medias/00/2d/90-7-wonders-architects-rulebook.pdf); BGG images
  6571587 / 6571595; the BGA client.
* The rule questions: BGG threads 2758424 (Economy), 3429296 (same-turn token), 3102322 / 3256390 /
  2781153 (decks run out), 2835875 / 3004888 (Wonder effect optional) — only the questions were
  indexed.

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
2. **Olympia's two cards are no longer separated by a check** (resolved 2026-09-13).  The first
   card is placed with reason `olympia_nocheck` (no `("check",)` queued), the second with
   `olympia`; the mandatory science / build check runs once both cards have landed, so a payment
   may use either card (`tests/test_review_fixes.py::test_olympia_takes_both_cards_before_the_mandatory_build`).
   The earlier approximation — a build between the two cards whose payment could not see the
   second — is gone.  (The `state.py` docstring still describes the old behaviour, §14 item 8.)
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
8. **Economy counts towards affordability** by default (`economy_forces_build = True`, BGA
   behaviour; §4.2); `False` gives the BGG "optional" reading.  Which is right is **UNKNOWN**.
9. **Canonical payment order** (§4.3): payments name cards in code order; this changes only the
   *path*, never the set of reachable multisets.
10. **Player 0 always starts** (§1.3); the rulebook's "youngest player" is a seat assignment
    handled outside the engine.
11. **Unlimited Military Victory tokens** and **science sets only while a token remains** (§6.2).
12. **All decks empty ⇒ game over at the next `turn_start`** (§9.1).

---

## 13. Worked example: one full turn

Real trace from `Environment(wonders=(2, 6), seed=5)` (player 0 = Alexandria, player 1 = Babylon)
with uniformly random actions (`random.Random(5)` over `legal_actions()`), turn 36, player 0 to
move.  `describe()` output has been condensed; the queue shown is `state.queue` at each decision
node (the popped item being handled is not in it).  Re-verified on 2026-09-13 against the current
`decks.json` (Giza edit, blue re-fit) and engine: the same seed reproduces this state at turn 36
and every step below — the two re-fitted blue cards of the Alexandria and Babylon decks sit
below the cards drawn by turn 36, so only the number of *distinct* kinds left in deck 0 changed
(step 3).

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
3. **Chance `reveal` deck 0** — 6 distinct kinds possible (one card of each: deck 0's 6 unseen
   cards are all different), probabilities = unseen counts / 6; the environment resolves it with
   the true next card: **civ3** (now visible to both).
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

Status re-checked on 2026-09-13 against `state.py` / `env.py` / `rules.py` and the test suite
(`tests/test_engine.py`, `test_env.py`, `test_review_fixes.py`: 183 passed, no `xfail` marker
left).  Of the original six items, 1, 3, 4 and 6 are **fixed**, 2 is mostly fixed, 5 stands;
items 7 and 8 are new.  Reproductions are kept for the record.

1. **FIXED — Architecture could fire twice in one turn** (rule: once per turn).  Original repro
   (conftest helpers): P0 = Alexandria at stage 1 with Architecture, tableau wood×2 + stone + clay,
   opponent's deck top papyrus; run `("check",)`: stage 2 (2 identical) is built → queue = `pick
   alexandria`, `pick token:0`, `check`; the Alexandria pick takes papyrus → stone+clay+papyrus →
   stage 3 (3 different) is built **before** the first Architecture pick executes → `_build`
   queued a *second* `pick token:0` (the `tokens_used` bit is only set when the extra card is
   *placed*).  Now `_handle("pick")` drops any `token:<t>` item whose bit in `tokens_used` is
   already set, so the second pick is discarded when its turn comes.  Regression test:
   `tests/test_review_fixes.py::test_architecture_cannot_fire_twice_through_a_stage_effect_card`
   (Ephesus variant).
2. **MOSTLY FIXED — dead `RulesConfig` flags.**  Three of the four are now consulted:
   `economy_once_per_build` (`_build` clears `econ_used`; its default changed from True to
   **False** = once per turn, matching the rulebook and the actual behaviour documented before),
   `token_usable_same_turn` (`tokens_new` bitmask read by `_token_active` — with the limitation of
   item 7) and `cat_peek_main_draw_only` (`_handle("pick")` peeks before extra central picks when
   False).  `end_when_no_cards` still has **no effect** — by design (with 2 players "the mover
   cannot draw" ≡ "all decks are empty"; `rules.py` comment;
   `test_all_decks_empty_ends_the_game_even_with_the_rule_disabled`).  Either delete it or keep it
   documented as inert (§10); do not describe it as live.
3. **FIXED — Economy mandatory for affordability with no alternative.**  `economy_forces_build`
   (§4.2, §10) exposes the BGG "optional" reading; verified: with the flag off, 1 coin + Economy
   does not force a "2 identical" build, and 2 coins + Economy still offers `PAY_COIN2`.
4. **FIXED — docstring claim** "does not change what a player can achieve" was too strong; the
   `state.py` docstring now names the deviations from a free ordering — but one of the two it
   names has since been removed, see item 8.
5. **Data-dependent test expectations** (unchanged): `TestData::test_wonder_stage_tables` pins
   the VP totals, the effect type per Wonder, the effect counts (Ephesus 3, others 2, Giza 0) and
   Rhodes's effect stages `[1, 3]`; `test_deck_compositions` pins only the sums (25 / 60 / 110)
   and `NUM_KINDS == 14` — no per-kind count is pinned, which is why the 2026-09-13 data edits
   (Giza's horned cards, the blue re-fit) needed no test change.  Correcting the ASSUMED Wonder
   data (§5.2, §11.3) may still require updating those expectations together with the JSON, and
   any checkpoint trained on the previous data is invalidated.
6. **FIXED — identical-stage payment could dead-end and silently skip a mandatory build.**
   Original repro: Giza at stage 3 (next: 3 identical), tableau wood×2 + coin; `check` → the
   first `pay_wood` auto-resolved, then `ctx=(0,)` offered `pay_wood` **and** `pay_coin`; choosing
   `pay_coin` gave `ctx=(0, 5)`, need 1, `_pay_options` returned `[]`, `_pay_decision` returned
   `False` and the turn passed with the stage unbuilt.  Now `_max_value` in mode 0 with `used_res`
   returns `(grey_avail[used_res[0]] if used_res[0] >= min_code else 0) + coin_cap`, so `pay_coin`
   is not offered where only a grey card could complete the payment, and `_pay_decision` raises
   `RuntimeError("payment dead end …")` instead of returning `False`, so such a state can never
   pass silently.  `test_identical_stage_coin_after_grey_never_dead_ends` passes and is no longer
   marked xfail.
7. **NEW — `token_usable_same_turn = False` is only partially implemented.**  `_token_active`
   (used by `_placed` for the extra-card triggers and by `_build` for Architecture) honours
   `tokens_new`, so the six extra-card tokens gained this turn stay silent until the next turn;
   but `_mode`, `_max_value` and `_pay_options` read `self.tokens[mover]` directly, so Engineering
   and Economy gained from a science set still apply to the build of the same check.  Verified on
   the current engine: Giza at stage 1 ("2 identical") with wood + stone + tablet×2 and Engineering
   face up is built in the same turn under the flag; likewise coin + tablet×2 with Economy face up.
   Fix: gate the Engineering / Economy lookups on `tokens_new` as `_token_active` does (and reset
   `tokens_new` bits when the flag is True to keep `key()` stable), or document the flag as
   "extra-card tokens only" in `rules.py`.  The default (True) is unaffected.
8. **NEW — stale comments.**  (a) The `state.py` module docstring still says "Olympia's two cards
   are placed one after the other with a check in between"; since the `olympia_nocheck` reason both
   cards land before the check (§5.1, §12.1 item 2,
   `test_olympia_takes_both_cards_before_the_mandatory_build`).  (b) The docstring of
   `tests/test_engine.py::test_economy_is_once_per_turn_not_once_per_build` says
   `RulesConfig.economy_once_per_build` "is not consulted by the engine" — it is (item 2).
   Documentation only; no behavioural impact.
