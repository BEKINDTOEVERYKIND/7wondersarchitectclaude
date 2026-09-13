"""Wonder definitions (loaded from ``data/wonders.json``)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Tuple

_DATA = os.path.join(os.path.dirname(__file__), "data", "wonders.json")

IDENTICAL, DIFFERENT = 0, 1

# Stage effects
E_NONE = "none"
E_SHIELD = "shield"                      # Rhodes: +1 permanent shield
E_ANY_DECK = "any_deck_card"             # Alexandria: take top card from any deck
E_LOOK5 = "look_5_choose_1"              # Halicarnassus: top 5 of left/right deck, keep 1
E_LEFT_RIGHT = "left_and_right_cards"    # Olympia: top card of left AND right decks
E_CENTRAL = "central_card"               # Ephesus: top card of the central deck
E_TOKEN = "progress_token"               # Babylon: take a Progress token
EFFECTS = (E_NONE, E_SHIELD, E_ANY_DECK, E_LOOK5, E_LEFT_RIGHT, E_CENTRAL, E_TOKEN)


@dataclass(frozen=True)
class Stage:
    cost: int
    kind: int  # IDENTICAL / DIFFERENT
    vp: int
    effect: str


@dataclass(frozen=True)
class Wonder:
    id: int
    name: str
    stages: Tuple[Stage, ...]

    @property
    def total_vp(self) -> int:
        return sum(s.vp for s in self.stages)


def _load() -> List[Wonder]:
    with open(_DATA) as f:
        raw = json.load(f)
    out = []
    for i, w in enumerate(raw["wonders"]):
        stages = tuple(Stage(int(s["cost"]), IDENTICAL if s["kind"] == "identical" else DIFFERENT, int(s["vp"]),
                             s.get("effect", E_NONE)) for s in w["stages"])
        assert len(stages) == 5, w["name"]
        for s in stages:
            assert s.effect in EFFECTS, (w["name"], s.effect)
        out.append(Wonder(i, w["name"], stages))
    return out


WONDERS: List[Wonder] = _load()
WONDER_BY_NAME = {w.name: w for w in WONDERS}
NUM_WONDERS = len(WONDERS)
