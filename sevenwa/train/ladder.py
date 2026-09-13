"""Round-robin ladder between agent specs with Elo fitting."""
from __future__ import annotations

import itertools
import json
from typing import Dict, List, Optional

from ..agents.factory import make_agent
from ..engine.env import make_env
from ..search.mcts import MCTSConfig
from .arena import play_match
from .elo import fit_elo


def run_ladder(specs: List[str], games_per_pair: int = 20, seed: int = 0, sims: int = 100, log=print,
               anchor: Optional[str] = None) -> Dict:
    results = []
    table = {}
    for a, b in itertools.combinations(specs, 2):
        agent_a = make_agent(a, seed=seed, mcts_config=MCTSConfig(num_simulations=sims))
        agent_b = make_agent(b, seed=seed + 1, mcts_config=MCTSConfig(num_simulations=sims))
        res = play_match(make_env, (agent_a, agent_b), games_per_pair, seed=seed)
        log(res.summary((a, b)))
        table[f"{a} vs {b}"] = {"score_a": res.score(0), "wins": res.wins, "draws": res.draws}
        for score in res.outcomes:  # game returns (stage tie-break included), not raw scores
            results.append((a, b, score))
    anchor = anchor or (specs[0] if specs else "random")
    elo = fit_elo(results, anchor=anchor, anchor_rating=0.0)
    log("Elo (anchor %s = 0): %s" % (anchor, json.dumps({k: round(v) for k, v in sorted(elo.items(), key=lambda kv: -kv[1])})))
    return {"pairs": table, "elo": elo}
