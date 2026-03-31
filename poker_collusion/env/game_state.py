"""
Game state for 3-player NLHE: 20 BB, action history with indices + DEAL.
"""

import numpy as np

from poker_collusion.config import (
    NUM_PLAYERS,
    STARTING_STACK_BB,
    SMALL_BLIND_BB,
    BIG_BLIND_BB,
)

# Sentinel for "community cards dealt" in action_history (must be hashable for info set key)
DEAL = "DEAL"


class NLHEState:
    """
    Immutable-by-convention state for one hand. Do not mutate directly —
    use apply_action() and sample_chance() which return new copies.

    action_history: action indices (0..9) or DEAL sentinels, in order applied.
    actor_history:  player index who took each action in action_history (no entry for DEAL).
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
        "actor_history",
        "last_raiser",
        "last_raise_amount",
        "done",
        "chance_pending",
    )

    def __init__(self):
        self.deck = []
        self.deck_idx = 0
        self.hole_cards = [[] for _ in range(NUM_PLAYERS)]
        self.board = []
        self.round_idx = 0  # 0=preflop, 1=flop, 2=turn, 3=river
        self.stacks = (STARTING_STACK_BB,) * NUM_PLAYERS
        self.pot = 0.0
        self.bets = (0.0,) * NUM_PLAYERS  # current street bets
        self.active = (True,) * NUM_PLAYERS
        self.all_in = (False,) * NUM_PLAYERS
        self.current_player = 0
        self.action_history = ()  # int (action index) or DEAL
        self.actor_history = ()   # player who took each action (no entry for DEAL)
        self.last_raiser = -1
        self.last_raise_amount = 0.0  # min raise size for next raiser
        self.done = False
        self.chance_pending = False  # True when street ended, need to deal

    def copy(self):
        """Return a shallow copy of this state. Tuples are shared by reference (immutable).
        Only hole_cards, board, and deck need real copying since they are mutable lists."""
        s = NLHEState.__new__(NLHEState)
        s.deck = self.deck          # immutable after deal — shared ref is safe
        s.deck_idx = self.deck_idx
        s.hole_cards = [list(h) for h in self.hole_cards]
        s.board = list(self.board)
        s.round_idx = self.round_idx
        s.stacks = self.stacks       # tuple — immutable, share ref
        s.pot = self.pot
        s.bets = self.bets           # tuple — immutable, share ref
        s.active = self.active       # tuple — immutable, share ref
        s.all_in = self.all_in       # tuple — immutable, share ref
        s.current_player = self.current_player
        s.action_history = self.action_history  # tuple — immutable, share ref
        s.actor_history = self.actor_history    # tuple — immutable, share ref
        s.last_raiser = self.last_raiser
        s.last_raise_amount = self.last_raise_amount
        s.done = self.done
        s.chance_pending = self.chance_pending
        return s


def deal_new_hand():
    """Deal a fresh 3-player hand. P0=Button, P1=SB, P2=BB. Preflop order: P0, P1, P2."""
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
    stacks = list(state.stacks)
    bets = list(state.bets)
    stacks[1] -= SMALL_BLIND_BB
    bets[1] = SMALL_BLIND_BB
    stacks[2] -= BIG_BLIND_BB
    bets[2] = BIG_BLIND_BB
    state.stacks = tuple(stacks)
    state.bets = tuple(bets)
    state.pot = SMALL_BLIND_BB + BIG_BLIND_BB
    state.current_player = 0
    state.last_raiser = 2  # BB counts as "raiser" for preflop min-raise
    state.last_raise_amount = BIG_BLIND_BB
    return state


def get_payoffs(state):
    """Net profit in BB for each player (stacks - starting stack). Only valid when done."""
    return [state.stacks[p] - STARTING_STACK_BB for p in range(NUM_PLAYERS)]


def reconstruct_actor_history(state):
    """
    Return actor_history as a list. For the copy-on-write state, actor_history
    is already stored as a tuple on the state. This function exists for
    compatibility with the debug module.
    """
    return list(state.actor_history)
