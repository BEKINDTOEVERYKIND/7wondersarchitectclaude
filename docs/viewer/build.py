#!/usr/bin/env python3
"""Inline the recorded games into the viewer template -> docs/viewer/index.html (a single self-contained page)."""
import glob
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
games = []
for p in sorted(glob.glob(os.path.join(here, "games", "game_*.json"))):
    with open(p) as f:
        games.append(json.load(f))
tpl = open(os.path.join(here, "template.html")).read()
data = json.dumps(games, separators=(",", ":")).replace("</", "<\\/")
out = tpl.replace("__GAMES_JSON__", data)
with open(os.path.join(here, "index.html"), "w") as f:
    f.write(out)
print(f"index.html: {len(games)} games, {len(out) / 1e6:.2f} MB")
