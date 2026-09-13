"""Rule options.  Defaults follow the official rulebook (2-player) as researched in docs/RULES.md.

Ambiguous or unverifiable points are exposed as flags rather than silently chosen.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RulesConfig:
    conflict_tokens: int = 3          # CONFIRMED (DE/IT sources): 3 tokens in a 2-player game
    faceup_progress_tokens: int = 3   # CONFIRMED
    military_token_vp: int = 3        # CONFIRMED
    cat_vp: int = 2                   # CONFIRMED (BGA/FR/DE): the Cat pawn is worth 2 VP to its holder
    # 2p battle: ≥2× shields => 2 tokens.  BGA: 1 vs 0 gives 1 token, 2+ vs 0 gives 2 tokens.
    double_vs_zero_requires_two: bool = True
    # Improvement tokens are optional ("you can use"); when False extra draws are forced.
    extra_card_optional: bool = True
    # Wonder-stage card effects (Alexandria/Ephesus/Olympia/Halicarnassus): mandatory (default) or may be skipped.
    wonder_effect_optional: bool = False
    # Economy: the doubled coin may be used once per turn (False, rulebook: "each Progress token can only
    # be used once per turn") or once per stage construction (True, house rule).
    economy_once_per_build: bool = False
    # Economy's doubled coin counts towards the *mandatory* construction check (BGA reading, True) or
    # only becomes usable once the stage is affordable without it (BGG "you can use" reading, False).
    economy_forces_build: bool = True
    # Coins may replace any resource even when a grey card is available (free payment choice).
    coins_free_choice: bool = True
    # Kept for configuration compatibility: in a 2-player game every player draws from the same three
    # decks, so "the player to move cannot draw" == "all decks are empty", and the engine always ends
    # the game then (otherwise no progress is possible).  The flag therefore has no effect.
    end_when_no_cards: bool = True
    # Progress tokens taken this turn may be used this same turn (rulebook: "whenever you want").
    token_usable_same_turn: bool = True
    # Cat peek applies only to the first (main) draw of the turn (BGA ruling).
    cat_peek_main_draw_only: bool = True
    # Tie-break on equal score: most constructed stages wins; otherwise a draw.
    tiebreak_stages: bool = True
