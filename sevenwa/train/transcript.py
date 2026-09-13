"""Play one game between two agents and write an annotated, human-readable transcript."""
from __future__ import annotations

import time
from typing import List, Optional

from ..agents.base import Agent
from ..engine.actions import Actions as A
from ..engine.cards import KINDS, describe_counts
from ..engine.env import Environment
from ..engine.state import CENTRAL, D_HALI_CHOOSE, D_HALI_DECK, D_PAY, D_PICK, D_SCIENCE, D_TOKEN, GameState
from ..engine.tokens import TOKENS
from ..engine.wonders import WONDERS


def _card(k: int) -> str:
    return "?" if k < 0 else KINDS[k].name


def _table(s: GameState) -> str:
    tops = []
    for d in range(3):
        top = s.deck_top[d]
        label = ["P0 deck", "P1 deck", "center"][d]
        if d == CENTRAL and top >= 0:
            knowers = [p for p in (0, 1) if s.knows_central(p)]
            tops.append(f"{label}[{s.deck_size[d]}]: {_card(top)} (known to {knowers})")
        else:
            tops.append(f"{label}[{s.deck_size[d]}]: {_card(top) if d != CENTRAL else '?'}")
    return " | ".join(tops)


def _player(s: GameState, p: int) -> str:
    w = WONDERS[s.wonder[p]]
    toks = [TOKENS[i].name for i, c in enumerate(s.tokens[p]) for _ in range(c)]
    return (f"P{p} {w.name} stage {s.stages[p]}/5  score {s.score_of(p):2d}  shields {s.shields_of(p)}  "
            f"mil {s.mil_tokens[p]}  cat {'yes' if s.cat == p else 'no'}  tokens {toks}\n"
            f"       cards: {describe_counts(s.cards[p]) or '-'}")


def _decision_context(s: GameState) -> str:
    k = s.dkind
    if k == D_PICK:
        reason, optional, avail = s.dctx
        return f"pick a card ({reason}{', optional' if optional else ''})"
    if k == D_PAY:
        st = WONDERS[s.wonder[s.mover]].stages[s.stages[s.mover]]
        return f"pay for stage {s.stages[s.mover] + 1} ({st.cost} {'identical' if st.kind == 0 else 'different'}), paid so far {list(s.dctx)}"
    if k == D_SCIENCE:
        return "choose which science set to convert"
    if k == D_TOKEN:
        return f"choose a Progress token ({s.dctx}); face-up: {[TOKENS[t].name for t in s.faceup]}"
    if k == D_HALI_DECK:
        return "Halicarnassus: choose which deck to look at"
    if k == D_HALI_CHOOSE:
        return f"Halicarnassus: keep one of {[_card(k) for k in s.dctx[1]]}"
    return "?"


def play_transcript(agents, seed: int = 0, wonders=None, max_lines: Optional[int] = None) -> str:
    """Return a markdown transcript of one game (``agents[i]`` plays seat ``i``)."""
    env = Environment(seed=seed, wonders=wonders)
    for a in agents:
        a.new_game()
    lines: List[str] = []
    s = env.state
    lines.append(f"# Self-play game (seed {seed})\n")
    lines.append(f"**P0: {agents[0].name}** plays {WONDERS[s.wonder[0]].name}; **P1: {agents[1].name}** plays {WONDERS[s.wonder[1]].name}.  ")
    lines.append(f"Face-up Progress tokens: {[TOKENS[t].name for t in s.faceup]}.  Conflict tokens: {s.rules.conflict_tokens}.\n")
    ply = 0
    last_turn = -1
    t0 = time.time()
    while not env.is_terminal():
        p = env.to_move()
        obs = env.observe(p)
        if obs.turn != last_turn:
            last_turn = obs.turn
            lines.append(f"\n## Turn {obs.turn + 1} — P{p} to move\n")
            lines.append(f"Table: {_table(obs)}  ")
            lines.append(f"{_player(obs, 0)}  ")
            lines.append(f"{_player(obs, 1)}  ")
            lines.append(f"Conflict: {obs.conflict}/{obs.rules.conflict_tokens}, face-up tokens {[TOKENS[t].name for t in obs.faceup]}\n")
        legal = list(obs.legal_actions())
        action = agents[p].select_action(obs)
        res = getattr(agents[p], "last_result", None)
        stats = ""
        if res is not None:
            q = res.extra.get("q", {})
            parts = []
            for a in legal:
                n = res.visit_counts.get(a, 0)
                qa = q.get(a)
                parts.append(f"{A.name(a)} n={n}" + (f" q={qa:+.2f}" if qa is not None else ""))
            stats = f"  search: root value {res.root_value:+.2f}; " + ", ".join(parts)
        lines.append(f"- P{p} {_decision_context(obs)} → **{A.name(action)}**{stats}")
        env.step(action)
        s = env.state
        ply += 1
        # report consequences visible right after the step
        if s.battle_pending or (s.conflict == 0 and obs.conflict > 0 and s.turn != obs.turn):
            pass
        if max_lines and len(lines) > max_lines:
            lines.append("... (truncated)")
            break
    s = env.state
    lines.append("\n## Final position\n")
    lines.append(f"{_player(s, 0)}  ")
    lines.append(f"{_player(s, 1)}  ")
    r = env.returns()
    winner = "draw" if r[0] == r[1] else ("P0" if r[0] > r[1] else "P1")
    lines.append(f"\n**Result: {winner}** — scores {s.scores()[0]}–{s.scores()[1]} after {s.turn + 1} turns, {ply} decisions "
                 f"({time.time() - t0:.0f}s).")
    return "\n".join(lines)
