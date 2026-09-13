from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..game import GameState


@runtime_checkable
class Agent(Protocol):
    name: str

    def new_game(self) -> None:
        """Reset per-game memory (search trees etc.)."""

    def select_action(self, state: GameState) -> int:
        """Return a legal action for the player to move in ``state``."""
