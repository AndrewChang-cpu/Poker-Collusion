"""
Game state for 3-player NLHE: 20 BB, action history with indices + DEAL.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import numpy as np

from poker_collusion.config import (
    NUM_PLAYERS,
    STARTING_STACK_BB,
    SMALL_BLIND_BB,
    BIG_BLIND_BB,
)

# Sentinel for "community cards dealt" in action_history (must be hashable for info set key)
DEAL: str = "DEAL"

UndoEntry = Union[
    Tuple[str, int, List[float], int, float],
    Dict[str, Any],
]


class NLHEState:
    """
    Mutable state for one hand.
    action_history: list of action indices (0..9) or DEAL.
    undo_stack: list of snapshots for step_back (one per apply_action or sample_chance).
    """

    __slots__ = (
        "deck",
        "deck_idx",
        "hole_cards",
        "board",
        "round_idx",
        "stacks",
        "pot",
        "bets",
        "active",
        "all_in",
        "current_player",
        "action_history",
        "last_raiser",
        "last_raise_amount",
        "done",
        "chance_pending",
        "undo_stack",
    )

    def __init__(self) -> None:
        self.deck: List[int] = []
        self.deck_idx: int = 0
        self.hole_cards: List[List[int]] = [[] for _ in range(NUM_PLAYERS)]
        self.board: List[int] = []
        self.round_idx: int = 0  # 0=preflop, 1=flop, 2=turn, 3=river
        self.stacks: List[float] = [STARTING_STACK_BB] * NUM_PLAYERS
        self.pot: float = 0.0
        self.bets: List[float] = [0.0] * NUM_PLAYERS  # current street bets
        self.active: List[bool] = [True] * NUM_PLAYERS
        self.all_in: List[bool] = [False] * NUM_PLAYERS
        self.current_player: int = 0
        self.action_history: List[Union[int, str]] = []  # int (action index) or DEAL
        self.last_raiser: int = -1
        self.last_raise_amount: float = 0.0  # min raise size for next raiser
        self.done: bool = False
        self.chance_pending: bool = False  # True when street ended, need to deal
        self.undo_stack: List[UndoEntry] = []  # for step_back


def deal_new_hand() -> NLHEState:
    """Deal a fresh 3-player hand. P0=Button, P1=SB, P2=BB. Preflop order 0,1,2."""
    state = NLHEState()
    state.deck = list(np.random.permutation(52))
    state.deck_idx = 0
    # Hole cards
    for p in range(NUM_PLAYERS):
        state.hole_cards[p] = [
            state.deck[state.deck_idx],
            state.deck[state.deck_idx + 1],
        ]
        state.deck_idx += 2
    # Blinds
    state.stacks[1] -= SMALL_BLIND_BB
    state.bets[1] = SMALL_BLIND_BB
    state.stacks[2] -= BIG_BLIND_BB
    state.bets[2] = BIG_BLIND_BB
    state.pot = SMALL_BLIND_BB + BIG_BLIND_BB
    state.current_player = 0
    state.last_raiser = 2  # BB counts as "raiser" for preflop min-raise
    state.last_raise_amount = BIG_BLIND_BB
    return state


def get_payoffs(state: NLHEState) -> List[float]:
    """Net profit in BB for each player (stacks - 20). Only valid when state.done."""
    return [state.stacks[p] - STARTING_STACK_BB for p in range(NUM_PLAYERS)]
