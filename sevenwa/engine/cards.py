"""Card kinds and deck compositions (loaded from ``data/decks.json``).

Cards of the same *kind* are indistinguishable, so the engine represents every collection
of cards (a player's tableau, the unseen part of a deck, the discard) as a count vector
indexed by kind id.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

_DATA = os.path.join(os.path.dirname(__file__), "data", "decks.json")

RESOURCES: List[str] = ["wood", "stone", "clay", "papyrus", "glass"]
SYMBOLS: List[str] = ["tablet", "gear", "compass"]

GREY, YELLOW, BLUE, GREEN, RED = 0, 1, 2, 3, 4
TYPE_NAMES = ["grey", "yellow", "blue", "green", "red"]
_TYPE_IDS = {n: i for i, n in enumerate(TYPE_NAMES)}


@dataclass(frozen=True)
class Kind:
    id: int
    name: str
    type: int  # GREY..RED
    resource: int = -1  # index into RESOURCES for grey cards
    vp: int = 0
    cat: bool = False
    symbol: int = -1  # index into SYMBOLS for green cards
    shields: int = 0
    horns: int = 0

    @property
    def type_name(self) -> str:
        return TYPE_NAMES[self.type]


def _load():
    with open(_DATA) as f:
        raw = json.load(f)
    kinds = []
    for k in raw["kinds"]:
        kinds.append(Kind(
            id=k["id"], name=k["name"], type=_TYPE_IDS[k["type"]],
            resource=RESOURCES.index(k["resource"]) if "resource" in k else -1,
            vp=k.get("vp", 0), cat=bool(k.get("cat", False)),
            symbol=SYMBOLS.index(k["symbol"]) if "symbol" in k else -1,
            shields=k.get("shields", 0), horns=k.get("horns", 0)))
    kinds.sort(key=lambda k: k.id)
    assert [k.id for k in kinds] == list(range(len(kinds))), "kind ids must be 0..n-1"
    return raw, kinds


_RAW, KINDS = _load()
NUM_KINDS = len(KINDS)
KIND_BY_NAME: Dict[str, Kind] = {k.name: k for k in KINDS}


def _counts_from_map(m: Dict[str, int]) -> List[int]:
    v = [0] * NUM_KINDS
    for name, c in m.items():
        if name.startswith("_"):
            continue
        v[KIND_BY_NAME[name].id] = int(c)
    return v


def wonder_deck_counts(wonder_name: str) -> List[int]:
    base = dict(_RAW["standard_wonder_deck"])
    base.update({k: v for k, v in _RAW["wonder_deck_overrides"].get(wonder_name, {}).items() if not k.startswith("_")})
    return _counts_from_map(base)


def central_deck_counts() -> List[int]:
    return _counts_from_map(_RAW["central_deck"])


# Convenience index lists
GREY_KINDS = [k.id for k in KINDS if k.type == GREY]
YELLOW_KINDS = [k.id for k in KINDS if k.type == YELLOW]
BLUE_KINDS = [k.id for k in KINDS if k.type == BLUE]
GREEN_KINDS = [k.id for k in KINDS if k.type == GREEN]
RED_KINDS = [k.id for k in KINDS if k.type == RED]
KIND_OF_RESOURCE = {k.resource: k.id for k in KINDS if k.type == GREY}
KIND_OF_SYMBOL = {k.symbol: k.id for k in KINDS if k.type == GREEN}
COIN_KIND: int = YELLOW_KINDS[0]


def describe_counts(counts: List[int]) -> str:
    return ", ".join(f"{KINDS[i].name}x{c}" for i, c in enumerate(counts) if c)
