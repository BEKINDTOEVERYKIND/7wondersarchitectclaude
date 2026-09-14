# Engine data files

These JSON files are the *only* place where card counts, Wonder stages and Progress tokens
are defined.  Edit them to correct the game data without touching code.

* `decks.json`   – composition of the 7 Wonder decks (25 cards each) and the central deck (60).
* `wonders.json` – the 5 stages of each Wonder: cost, VP, effect and construction prerequisites.
* `tokens.json`  – the 15 Progress tokens.

Provenance is recorded in `docs/RULES.md` (§1.2, §5.2, §6.1, §11) and repeated in the `_status` /
`_note` strings of each file (keys starting with `_` are ignored by the loaders).

**Status (2026-09-14): every value in the three files is CONFIRMED.**  Two research passes from
search-engine relays (2026-09-13) confirmed the deck sizes, the 14 card kinds, the token texts and
values, the Wonder effect wordings and effect counts; the exact per-deck card counts and the
per-stage Wonder data were then supplied from the physical components (the printed card-distribution
table and the seven Wonder boards, read from the publisher's component image for Alexandria and
component photographs for the other six).

* `decks.json` — `standard_wonder_deck` is the *modal* count of each kind over the seven decks;
  `wonder_deck_overrides[<Wonder>]` lists only the kinds where that deck differs (the `_note`
  spells out every full deck).  Totals: 235 cards; 17 of each resource; 27 coins; 24 two-VP cat
  cards; 16 three-VP cards; 14 of each science symbol; red cards 18 / 15 / 8 with 0 / 1 / 2 horns.
* `wonders.json` — stages are listed in a fixed **cost order** S1..S5 = 2 different, 2 identical,
  3 different, 3 identical, 4 different (the printed costs are the same for all seven Wonders).
  This is *not* the construction order: each stage's `requires` lists the stage indexes (0-based)
  that must already be built, exactly as the tray diagrams show — Alexandria and Giza are linear,
  Babylon `S1>S2>S3>{S4,S5}`, Ephesus `S1>{S2,S3,S4}>S5`, Halicarnassus `S1>S2>{S3,S4}>S5`,
  Olympia `S1>{S2,S3}>S4>S5`, Rhodes `{S1,S2}>S3>S4>S5`.  A stage with the effect carries the
  Wonder's effect id; totals 30 / 26 / 25 / 24 / 22 / 22 / 20.  Babylon's 2-identical stage prints
  0 VP; Rhodes' first shield is on its 2-different foundation.
* `tokens.json` — unchanged since the second research pass (all 14 types, Culture × 2).

Each Wonder deck must still sum to 25 and the central deck to 60 (`tests/test_engine.py::TestData`
pins the sums and the Wonder tables); changing any value invalidates trained checkpoints.
