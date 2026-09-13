# Engine data files

These JSON files are the *only* place where card counts, Wonder stages and Progress tokens
are defined.  Edit them to correct the game data without touching code.

* `decks.json`   – composition of the 7 Wonder decks (25 cards each) and the central deck (60).
* `wonders.json` – the 5 stages of each Wonder: cost, VP, effect.
* `tokens.json`  – the 15 Progress tokens.

Provenance is recorded in `docs/RULES.md`.  Fields marked `"_status": "ASSUMED"` could not be
verified against the physical components during development (network access to the rulebook
and BGG card-distribution files was blocked); they are internally consistent placeholders.
Everything marked `"CONFIRMED"` was cross-checked against at least two independent sources.
