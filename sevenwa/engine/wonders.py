"""Wonder definitions (loaded from ``data/wonders.json``).

Stages are indexed 0..4 in a fixed *cost* order (2 different, 2 identical, 3 different,
3 identical, 4 different).  This is **not** the construction order: each stage lists the stages
it ``requires`` (the printed tray diagram), and a player may construct any *available* stage —
unbuilt with all prerequisites built.  Sets of built stages are bitmasks (bit ``i`` = stage ``i``).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Tuple

_DATA = os.path.join(os.path.dirname(__file__), "data", "wonders.json")

IDENTICAL, DIFFERENT = 0, 1
NUM_STAGES = 5
ALL_BUILT = (1 << NUM_STAGES) - 1

# Stage effects
E_NONE = "none"
E_SHIELD = "shield"                      # Rhodes: +1 permanent shield
E_ANY_DECK = "any_deck_card"             # Alexandria: take top card from any deck
E_LOOK5 = "look_5_choose_1"              # Halicarnassus: top 5 of left/right deck, keep 1
E_LEFT_RIGHT = "left_and_right_cards"    # Olympia: top card of left AND right decks
E_CENTRAL = "central_card"               # Ephesus: top card of the central deck
E_TOKEN = "progress_token"               # Babylon: take a Progress token
EFFECTS = (E_NONE, E_SHIELD, E_ANY_DECK, E_LOOK5, E_LEFT_RIGHT, E_CENTRAL, E_TOKEN)


def popcount(mask: int) -> int:
    return bin(mask).count("1")


@dataclass(frozen=True)
class Stage:
    index: int
    cost: int
    kind: int  # IDENTICAL / DIFFERENT
    vp: int
    effect: str
    requires: Tuple[int, ...]   # stage indexes that must be built first
    prereq_mask: int            # the same as a bitmask

    def label(self) -> str:
        return f"S{self.index + 1}({self.cost}{'=' if self.kind == IDENTICAL else '≠'})"


@dataclass(frozen=True)
class Wonder:
    id: int
    name: str
    stages: Tuple[Stage, ...]
    # per built-mask tables (32 entries each)
    _available: Tuple[Tuple[int, ...], ...]
    _vp_of: Tuple[int, ...]
    _cost_left: Tuple[int, ...]
    _plan: Tuple[Tuple[int, ...], ...]

    @property
    def total_vp(self) -> int:
        return sum(s.vp for s in self.stages)

    def available(self, built: int) -> Tuple[int, ...]:
        """Stage indexes that may be constructed next given the built mask."""
        return self._available[built]

    def vp_of_built(self, built: int) -> int:
        return self._vp_of[built]

    def cost_remaining(self, built: int) -> int:
        """Total number of cards still needed to complete the Wonder."""
        return self._cost_left[built]

    def plan(self, built: int) -> Tuple[int, ...]:
        """A greedy construction order of the unbuilt stages: repeatedly the available stage with the
        lowest cost (ties: higher VP, then lower index).  Used by evaluators that need a notion of
        "the next stage" / "the stage after that"."""
        return self._plan[built]


def _available_of(stages: Tuple[Stage, ...], built: int) -> Tuple[int, ...]:
    return tuple(s.index for s in stages if not (built >> s.index) & 1 and (built & s.prereq_mask) == s.prereq_mask)


def _make_wonder(i: int, name: str, stages: Tuple[Stage, ...]) -> Wonder:
    avail = []
    vp_of = []
    cost_left = []
    plans = []
    for mask in range(1 << NUM_STAGES):
        avail.append(_available_of(stages, mask))
        vp_of.append(sum(s.vp for s in stages if (mask >> s.index) & 1))
        cost_left.append(sum(s.cost for s in stages if not (mask >> s.index) & 1))
        plan = []
        m = mask
        while True:
            av = _available_of(stages, m)
            if not av:
                break
            nxt = min(av, key=lambda j: (stages[j].cost, -stages[j].vp, j))
            plan.append(nxt)
            m |= 1 << nxt
        plans.append(tuple(plan))
    return Wonder(i, name, stages, tuple(avail), tuple(vp_of), tuple(cost_left), tuple(plans))


def _load() -> List[Wonder]:
    with open(_DATA) as f:
        raw = json.load(f)
    out = []
    for i, w in enumerate(raw["wonders"]):
        stages = []
        for j, s in enumerate(w["stages"]):
            req = tuple(int(r) for r in s.get("requires", ([j - 1] if j > 0 else [])))
            assert all(0 <= r < j for r in req), (w["name"], j, req)  # prerequisites precede in cost order
            mask = 0
            for r in req:
                mask |= 1 << r
            stages.append(Stage(j, int(s["cost"]), IDENTICAL if s["kind"] == "identical" else DIFFERENT, int(s["vp"]),
                                s.get("effect", E_NONE), req, mask))
        stages_t = tuple(stages)
        assert len(stages_t) == NUM_STAGES, w["name"]
        for s in stages_t:
            assert s.effect in EFFECTS, (w["name"], s.effect)
        wonder = _make_wonder(i, w["name"], stages_t)
        assert wonder.total_vp == int(w.get("total_vp", wonder.total_vp)), w["name"]
        assert len(wonder.plan(0)) == NUM_STAGES, w["name"]  # every stage is reachable
        out.append(wonder)
    return out


WONDERS: List[Wonder] = _load()
WONDER_BY_NAME = {w.name: w for w in WONDERS}
NUM_WONDERS = len(WONDERS)
