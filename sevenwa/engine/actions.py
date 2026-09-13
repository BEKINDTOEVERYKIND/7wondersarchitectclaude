"""Global action space (fixed indices so the policy head has a constant width)."""
from __future__ import annotations

from .cards import NUM_KINDS, KINDS, RESOURCES, SYMBOLS
from .tokens import NUM_TOKEN_TYPES, TOKENS


class Actions:
    PICK_LEFT = 0      # take the top card of the deck to the mover's left (= the mover's own Wonder deck)
    PICK_RIGHT = 1     # take the top card of the deck to the mover's right (= the opponent's Wonder deck)
    PICK_CENTER = 2    # take the top card of the face-down central deck
    SKIP = 3           # decline an optional extra card
    PAY_BASE = 4       # PAY_BASE + resource index (0..4): spend one grey card of that resource
    PAY_COIN = 9       # spend one yellow card as 1 coin
    PAY_COIN2 = 10     # spend one yellow card as 2 coins (Economy token)
    SCI_PAIR_BASE = 11 # SCI_PAIR_BASE + symbol index: discard 2 identical symbols
    SCI_TRIPLE = 14    # discard 3 different symbols
    TOKEN_BASE = 15    # TOKEN_BASE + token type id: take that face-up Progress token
    TOKEN_BLIND = TOKEN_BASE + NUM_TOKEN_TYPES  # take the top (face-down) token of the stack
    HALI_BASE = TOKEN_BLIND + 1                 # HALI_BASE + kind id: keep that card (Halicarnassus)
    NUM = HALI_BASE + NUM_KINDS

    @staticmethod
    def name(a: int) -> str:
        A = Actions
        if a == A.PICK_LEFT:
            return "pick_left(own deck)"
        if a == A.PICK_RIGHT:
            return "pick_right(opp deck)"
        if a == A.PICK_CENTER:
            return "pick_center"
        if a == A.SKIP:
            return "skip"
        if A.PAY_BASE <= a < A.PAY_BASE + 5:
            return f"pay_{RESOURCES[a - A.PAY_BASE]}"
        if a == A.PAY_COIN:
            return "pay_coin"
        if a == A.PAY_COIN2:
            return "pay_coin_x2(economy)"
        if A.SCI_PAIR_BASE <= a < A.SCI_PAIR_BASE + 3:
            return f"science_pair_{SYMBOLS[a - A.SCI_PAIR_BASE]}"
        if a == A.SCI_TRIPLE:
            return "science_triple"
        if A.TOKEN_BASE <= a < A.TOKEN_BLIND:
            return f"token_{TOKENS[a - A.TOKEN_BASE].name}"
        if a == A.TOKEN_BLIND:
            return "token_blind"
        if A.HALI_BASE <= a < A.NUM:
            return f"keep_{KINDS[a - A.HALI_BASE].name}"
        return f"?{a}"
