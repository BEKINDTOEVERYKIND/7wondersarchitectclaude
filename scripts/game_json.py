#!/usr/bin/env python3
"""Play one game between two agent specs and dump a structured JSON record for the game viewer.

Per decision: the belief state the mover saw (both tableaus, stages, tokens, scores, decks), the
legal options with the agent's search statistics (visits, Q, prior) and the chosen action; plus the
cards that arrived from chance draws.  Usage:

    python scripts/game_json.py --a hybrid:models/imitation_v7.pt:300:0.5 --b <same> --seed 7 --out games/g1.json
"""
import argparse
import json
import time

from sevenwa.agents.factory import make_agent
from sevenwa.engine.actions import Actions as A
from sevenwa.engine.cards import KINDS
from sevenwa.engine.env import Environment
from sevenwa.engine.state import (CENTRAL, D_HALI_CHOOSE, D_HALI_DECK, D_PAY, D_PICK, D_SCIENCE, D_STAGE, D_TOKEN,
                                  DECISION_NAMES)
from sevenwa.engine.tokens import TOKENS
from sevenwa.engine.wonders import WONDERS


def player_snapshot(s, p):
    w = WONDERS[s.wonder[p]]
    built = s.built[p]
    avail = w.available(built)
    return {
        "wonder": w.name,
        "stages": [{"idx": st.index, "cost": st.cost, "kind": "identical" if st.kind == 0 else "different", "vp": st.vp,
                    "effect": st.effect, "requires": list(st.requires),
                    "built": bool((built >> st.index) & 1), "available": st.index in avail} for st in w.stages],
        "cards": {KINDS[k].name: c for k, c in enumerate(s.cards[p]) if c},
        "tokens": [TOKENS[i].name for i, c in enumerate(s.tokens[p]) for _ in range(c)],
        "score": s.score_of(p),
        "stage_vp": w.vp_of_built(built),
        "shields": s.shields_of(p),
        "wonder_shields": s.wonder_shields[p],
        "mil_tokens": s.mil_tokens[p],
        "cat": s.cat == p,
    }


def table_snapshot(s):
    decks = []
    for d in range(3):
        top = s.deck_top[d]
        known = top >= 0 and (d != CENTRAL or s.central_known_to)
        decks.append({"size": s.deck_size[d], "top": KINDS[top].name if known else None,
                      "known_to": [p for p in (0, 1) if s.knows_central(p)] if d == CENTRAL else [0, 1]})
    return {"decks": decks, "conflict": s.conflict, "conflict_max": s.rules.conflict_tokens, "cat": s.cat,
            "faceup": [TOKENS[t].name for t in s.faceup], "stack": s.prog_stack,
            "discard": {KINDS[k].name: c for k, c in enumerate(s.discard) if c}}


def decision_text(s):
    k = s.dkind
    w = WONDERS[s.wonder[s.mover]]
    if k == D_PICK:
        reason, optional, _ = s.dctx
        r = reason.split(":")
        if r[0] == "token":
            reason = f"extra card from {TOKENS[int(r[1])].name}"
        elif r[0] == "main":
            reason = "main pick"
        else:
            reason = f"{reason} effect"
        return f"Pick a card ({reason}{', optional' if optional else ''})"
    if k == D_PAY:
        si, chosen = s.dctx
        st = w.stages[si]
        paid = [("coin×2" if c == 6 else ("coin" if c == 5 else KINDS[c].name)) for c in chosen]
        return f"Pay for {st.label()} ({st.cost} {'identical' if st.kind == 0 else 'different'}, {st.vp} VP)" + (f", paid so far {paid}" if paid else "")
    if k == D_STAGE:
        return "Choose which affordable stage to build: " + ", ".join(w.stages[i].label() for i in s.dctx)
    if k == D_SCIENCE:
        return "Choose which science set to convert into a Progress token"
    if k == D_TOKEN:
        return f"Choose a Progress token ({s.dctx})"
    if k == D_HALI_DECK:
        return "Halicarnassus: choose which deck to look at"
    if k == D_HALI_CHOOSE:
        return "Halicarnassus: keep one of " + ", ".join(KINDS[x].name for x in s.dctx[1])
    return DECISION_NAMES[k]


def action_text(s, a):
    if a == A.PICK_LEFT:
        return "take own deck top" + (f" ({KINDS[s.deck_top[s.mover]].name})" if s.deck_top[s.mover] >= 0 else "")
    if a == A.PICK_RIGHT:
        o = 1 - s.mover
        return "take opponent deck top" + (f" ({KINDS[s.deck_top[o]].name})" if s.deck_top[o] >= 0 else "")
    if a == A.PICK_CENTER:
        t = s.deck_top[CENTRAL]
        return "draw from the central deck" + (f" (peeked: {KINDS[t].name})" if t >= 0 and s.knows_central(s.mover) else " (blind)")
    if a == A.SKIP:
        return "skip"
    if A.PAY_BASE <= a < A.PAY_BASE + 5:
        return f"pay {KINDS[a - A.PAY_BASE].name}"
    if a == A.PAY_COIN:
        return "pay a coin"
    if a == A.PAY_COIN2:
        return "pay a coin doubled (Economy)"
    if A.SCI_PAIR_BASE <= a < A.SCI_PAIR_BASE + 3:
        return f"convert 2× {['tablet', 'gear', 'compass'][a - A.SCI_PAIR_BASE]}"
    if a == A.SCI_TRIPLE:
        return "convert tablet + gear + compass"
    if A.TOKEN_BASE <= a < A.TOKEN_BLIND:
        return f"take {TOKENS[a - A.TOKEN_BASE].name}"
    if a == A.TOKEN_BLIND:
        return "take the top face-down token"
    if A.HALI_BASE <= a < A.STAGE_BASE:
        return f"keep {KINDS[a - A.HALI_BASE].name}"
    if A.STAGE_BASE <= a < A.NUM:
        return f"build {WONDERS[s.wonder[s.mover]].stages[a - A.STAGE_BASE].label()}"
    return A.name(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wonders", default=None, help="two wonder ids, e.g. 1,5 (default: random for the seed)")
    args = ap.parse_args()
    wonders = tuple(int(x) for x in args.wonders.split(",")) if args.wonders else None
    agents = [make_agent(args.a, seed=args.seed * 2), make_agent(args.b, seed=args.seed * 2 + 1)]
    for ag in agents:
        ag.new_game()
    env = Environment(seed=args.seed, wonders=wonders)
    s = env.state
    game = {"seed": args.seed, "agents": [agents[0].name, agents[1].name],
            "wonders": [WONDERS[s.wonder[0]].name, WONDERS[s.wonder[1]].name],
            "faceup_start": [TOKENS[t].name for t in s.faceup], "turns": []}
    t0 = time.time()
    last_turn = -1
    cur = None
    while not env.is_terminal():
        p = env.to_move()
        obs = env.observe(p)
        if obs.turn != last_turn:
            last_turn = obs.turn
            cur = {"turn": obs.turn + 1, "mover": p, "start": {"players": [player_snapshot(obs, 0), player_snapshot(obs, 1)],
                                                              "table": table_snapshot(obs)}, "decisions": [], "events": []}
            game["turns"].append(cur)
        legal = list(obs.legal_actions())
        before_cards = [list(env.state.cards[0]), list(env.state.cards[1])]
        t1 = time.time()
        action = agents[p].select_action(obs)
        res = getattr(agents[p], "last_result", None)
        options = []
        q = res.extra.get("q", {}) if res is not None else {}
        prior = res.extra.get("prior", {}) if res is not None else {}
        for a in legal:
            opt = {"action": int(a), "text": action_text(obs, a)}
            if res is not None:
                opt["visits"] = int(res.visit_counts.get(a, 0))
                if a in q:
                    opt["q"] = round(float(q[a]), 3)
                if a in prior:
                    opt["prior"] = round(float(prior[a]), 3)
            options.append(opt)
        dec = {"player": p, "kind": DECISION_NAMES[obs.dkind], "text": decision_text(obs), "options": options,
               "chosen": int(action), "chosen_text": action_text(obs, action),
               "root_value": round(float(res.root_value), 3) if res is not None else None,
               "seconds": round(time.time() - t1, 2)}
        env.step(action)
        s = env.state
        gained = []
        for pp in (0, 1):
            for k in range(len(KINDS)):
                d = s.cards[pp][k] - before_cards[pp][k]
                if d > 0:
                    gained.extend([KINDS[k].name] * d)
        dec["cards_gained"] = gained
        cur["decisions"].append(dec)
        cur["end"] = {"players": [player_snapshot(s, 0), player_snapshot(s, 1)], "table": table_snapshot(s)}
    s = env.state
    r = env.returns()
    game["final"] = {"players": [player_snapshot(s, 0), player_snapshot(s, 1)], "table": table_snapshot(s),
                     "scores": list(s.scores()), "winner": -1 if r[0] == r[1] else (0 if r[0] > r[1] else 1),
                     "turns": s.turn + 1, "seconds": round(time.time() - t0, 1)}
    with open(args.out, "w") as f:
        json.dump(game, f)
    print(f"wrote {args.out}: {game['wonders']} scores {game['final']['scores']} in {game['final']['turns']} turns ({game['final']['seconds']}s)")


if __name__ == "__main__":
    main()
