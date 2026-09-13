# Engine data files

These JSON files are the *only* place where card counts, Wonder stages and Progress tokens
are defined.  Edit them to correct the game data without touching code.

* `decks.json`   – composition of the 7 Wonder decks (25 cards each) and the central deck (60).
* `wonders.json` – the 5 stages of each Wonder: cost, VP, effect.
* `tokens.json`  – the 15 Progress tokens.

Provenance is recorded in `docs/RULES.md` (§1.2, §5.2, §6.1, §11) and repeated in the `_status` /
`_note` strings of each file (keys starting with `_` are ignored by the loaders).  Two research
passes were made from search-engine relays only — the rulebook PDF and the BGG card-distribution
attachment were never fetchable (last pass 2026-09-13; verdicts: 21 CONFIRMED, 12 SINGLE-SOURCE,
14 UNKNOWN):

* **CONFIRMED** (two independent sources, or mirrored rulebook text): deck sizes; the 14 card
  kinds (every red card = 1 shield + 0/1/2 horns); standard Wonder deck 3 coins / Alexandria 4;
  every Wonder's effect wording; effect counts Rhodes 2, Babylon 2, Ephesus 3 (one dissent), Giza 0;
  all 15 tokens' names, texts and values (Science wording and the Urbanism / Crafts / Jewellery
  resource splits included).
* **SINGLE-SOURCE** (one page or one BGG post): Alexandria +1 gear −1 glass −1 one-horn and Giza
  2 one-horn / 0 two-horn (BoostYourPlay); the Wonder VP totals 30/26/25/24/22/22/20 and the
  effect counts of Olympia / Alexandria / Halicarnassus (BGG thread 2943868); Rhodes effects on
  stages 2 and 4 (BGG thread 2869259); game-wide proportions incl. BGA "over 11 % cat cards".
* **ASSUMED** (placeholders, internally consistent): all other per-deck counts — including the
  blue split 1 × 3 VP / 3 × 2 VP + cat per Wonder deck, re-fitted so that 27 of 235 cards (11.5 %)
  carry a cat icon — and the whole central deck; every per-stage cost order, VP split and effect
  placement in `wonders.json` (Giza may have fewer "identical" stages).  Nothing in `tokens.json`
  remains assumed.

To fill the gaps with normal internet access: BGG file 233664 / image 6620199 (the per-deck card
table, mirrored at bgmanual.com) and page 6 of the official EN rulebook PDF on
cdn.svc.asmodee.net (the Wonder trays).  Each Wonder deck must still sum to 25 and the central
deck to 60; changing any value invalidates trained checkpoints.
