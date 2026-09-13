"""Abstract game contract consumed by the search and training code.

The search never touches engine internals; it relies only on this protocol.  The
contract models a *two-player, sequential, stochastic* game as an explicit tree of

* **decision nodes** – ``to_move()`` is 0 or 1, ``legal_actions()`` is non-empty,
  ``apply_action(a)`` returns a **new** state; and
* **chance nodes** – ``to_move()`` is :data:`CHANCE`, ``chance_outcomes()`` lists
  ``(outcome_id, probability)`` pairs, ``apply_chance(o)`` returns a new state.

States must be *immutable from the caller's point of view*: ``apply_*`` must not
mutate ``self``.  This lets the search hold many states cheaply and lets the
environment and the search share one implementation.

Hidden information
------------------
The 7 Wonders: Architects engine represents decks as ``(visible top card, multiset of
unseen cards)``.  Revealing a card is therefore a chance node whose distribution is
derived from the unseen multiset, which is exactly the belief of a player who has seen
everything public.  The only private information in the game (the Cat holder peeking
at the central deck) is handled by the engine via a ``visible_to`` field; see
``sevenwa/engine/state.py``.
"""
from __future__ import annotations

from typing import Protocol, Sequence, Tuple, runtime_checkable

#: ``to_move()`` value that marks a chance node.
CHANCE = -1
#: Number of players (the engine is 2-player only; the search assumes zero-sum 2p).
NUM_PLAYERS = 2


@runtime_checkable
class GameState(Protocol):
    """Minimal interface a game must implement to be searchable/trainable."""

    # --- structure -------------------------------------------------------
    def to_move(self) -> int:
        """Player index (0/1) at a decision node, or :data:`CHANCE`."""

    def is_terminal(self) -> bool: ...

    def is_chance(self) -> bool: ...

    # --- decisions -------------------------------------------------------
    def legal_actions(self) -> Sequence[int]:
        """Global action ids that are legal now (decision nodes only, non-empty)."""

    def apply_action(self, action: int) -> "GameState":
        """Return the successor state (never mutate ``self``)."""

    # --- chance ----------------------------------------------------------
    def chance_outcomes(self) -> Sequence[Tuple[int, float]]:
        """``(outcome_id, probability)`` pairs summing to 1 (chance nodes only)."""

    def apply_chance(self, outcome: int) -> "GameState":
        """Return the successor state after the chance outcome (never mutate)."""

    # --- evaluation ------------------------------------------------------
    def returns(self) -> Tuple[float, float]:
        """Terminal payoff for (player 0, player 1) in {-1, 0, +1}; zero-sum."""

    def score_diff(self) -> float:
        """(score of player 0) - (score of player 1) using the current partial scoring.

        Used as an auxiliary training target and by heuristic rollouts.  Must be
        defined at every node, not only terminal ones.
        """

    def key(self) -> bytes:
        """Hashable identity of the *observable* state, for caches/transpositions."""

    # --- action-space metadata ------------------------------------------
    @staticmethod
    def num_actions() -> int:
        """Size of the global action space (policy-head width)."""


class Environment(Protocol):
    """The *real* game: owns the hidden information (shuffled deck orders, token stack).

    Agents never see the environment; they receive observer-specific belief states from
    :meth:`observe` and act with :meth:`step`, which applies the action and resolves every
    chance node with the true hidden state until the next decision node or the end.
    """

    def to_move(self) -> int:
        """Player to move (never :data:`CHANCE`) or -2 if terminal."""

    def is_terminal(self) -> bool: ...

    def observe(self, player: int) -> GameState:
        """Belief state as seen by ``player`` (private info of the other player stripped)."""

    def step(self, action: int) -> None: ...

    def returns(self) -> Tuple[float, float]: ...

    def scores(self) -> Tuple[int, int]: ...
