"""sevenwa — a self-play reinforcement-learning AI for 2-player *7 Wonders: Architects*.

Package layout
--------------
- ``sevenwa.game``      : abstract game-state contract used by the search (decision + chance nodes).
- ``sevenwa.engine``    : the 7 Wonders: Architects rules engine (cards, wonders, tokens, state, env).
- ``sevenwa.search``    : chance-aware PUCT / Gumbel MCTS with batched neural evaluation.
- ``sevenwa.nn``        : feature encoding and the policy/value network.
- ``sevenwa.agents``    : agents (random, heuristic, rollout-MCTS, neural MCTS).
- ``sevenwa.train``     : self-play, replay buffer, trainer, arena, Elo, pipeline.
"""

__version__ = "0.1.0"
