"""Progress tokens (loaded from ``data/tokens.json``)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Tuple

from .cards import RESOURCES

_DATA = os.path.join(os.path.dirname(__file__), "data", "tokens.json")

K_EXTRA_ON_BUILD = "extra_card_on_build"
K_EXTRA_ON_HORN = "extra_card_on_horn"
K_EXTRA_ON_RESOURCE = "extra_card_on_resource"
K_EXTRA_ON_GREEN = "extra_card_on_green"
K_ENGINEERING = "engineering"
K_ECONOMY = "economy"
K_SHIELDS = "shields"
K_VP_MIL = "vp_per_military_token"
K_VP_PROG = "vp_per_progress_token"
K_VP_WONDER = "vp_wonder"
K_VP_CAT = "vp_per_cat_icon"
K_CULTURE = "culture"


@dataclass(frozen=True)
class Token:
    id: int
    name: str
    kind: str
    copies: int = 1
    resources: Tuple[int, ...] = ()  # resource indices (or -1 for coin) for extra_card_on_resource
    triggers_on_coin: bool = False
    shields: int = 0
    vp: int = 0
    vp_incomplete: int = 0
    vp_complete: int = 0
    vp_one: int = 0
    vp_both: int = 0
    text: str = ""

    @property
    def is_endgame_vp(self) -> bool:
        return self.kind in (K_VP_MIL, K_VP_PROG, K_VP_WONDER, K_VP_CAT, K_CULTURE)

    @property
    def is_extra_card(self) -> bool:
        return self.kind in (K_EXTRA_ON_BUILD, K_EXTRA_ON_HORN, K_EXTRA_ON_RESOURCE, K_EXTRA_ON_GREEN)


def _load() -> List[Token]:
    with open(_DATA) as f:
        raw = json.load(f)
    out = []
    for t in raw["tokens"]:
        res = tuple(RESOURCES.index(r) for r in t.get("resources", []) if r != "coin")
        out.append(Token(
            id=int(t["id"]), name=t["name"], kind=t["kind"], copies=int(t.get("copies", 1)), resources=res,
            triggers_on_coin="coin" in t.get("resources", []), shields=int(t.get("shields", 0)),
            vp=int(t.get("vp", 0)), vp_incomplete=int(t.get("vp_incomplete", 0)), vp_complete=int(t.get("vp_complete", 0)),
            vp_one=int(t.get("vp_one", 0)), vp_both=int(t.get("vp_both", 0)), text=t.get("text", "")))
    out.sort(key=lambda t: t.id)
    assert [t.id for t in out] == list(range(len(out)))
    return out


TOKENS: List[Token] = _load()
NUM_TOKEN_TYPES = len(TOKENS)
TOTAL_TOKEN_COPIES = sum(t.copies for t in TOKENS)
TOKEN_BY_NAME = {t.name: t for t in TOKENS}
