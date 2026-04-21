"""
Game state for 3-player Leduc Hold'em: 12-card deck, 1 hole card per player.
"""

import numpy as np
from poker_collusion.config import (
    NUM_PLAYERS,
    STARTING_STACK_BB,
    SMALL_BLIND_BB,
    BIG_BLIND_BB,
    INITIAL_POT_BB
)

DEAL = "DEAL"

class NLHEState:
    __slots__ = (
        "deck", "deck_idx", "hole_cards", "board", "round_idx",
        "stacks", "pot", "bets", "active", "all_in",
        "current_player", "action_history", "actor_history",
        "last_raiser", "last_raise_amount", "done", "chance_pending",
    )

    def __init__(self):
        self.deck = []
        self.deck_idx = 0
        self.hole_cards = [[] for _ in range(NUM_PLAYERS)]
        self.board = []
        self.round_idx = 0  # 0=preflop, 1=flop
        self.stacks = (STARTING_STACK_BB,) * NUM_PLAYERS
        self.pot = 0.0
        self.bets = (0.0,) * NUM_PLAYERS
        self.active = (True,) * NUM_PLAYERS
        self.all_in = (False,) * NUM_PLAYERS
        self.current_player = 0
        self.action_history = ()
        self.actor_history = ()
        self.last_raiser = -1
        self.last_raise_amount = 0.0
        self.done = False
        self.chance_pending = False

    def copy(self):
        s = NLHEState.__new__(NLHEState)
        s.deck = self.deck
        s.deck_idx = self.deck_idx
        s.hole_cards = [list(h) for h in self.hole_cards]
        s.board = list(self.board)
        s.round_idx = self.round_idx
        s.stacks = self.stacks
        s.pot = self.pot
        s.bets = self.bets
        s.active = self.active
        s.all_in = self.all_in
        s.current_player = self.current_player
        s.action_history = self.action_history
        s.actor_history = self.actor_history
        s.last_raiser = self.last_raiser
        s.last_raise_amount = self.last_raise_amount
        s.done = self.done
        s.chance_pending = self.chance_pending
        return s

def deal_new_hand():
    """Deal a 3-player Leduc hand: 1 hole card each from a 12-card deck."""
    state = NLHEState()
    # 4 ranks (J=0, Q=1, K=2, A=3) * 3 suits = 12 cards
    ranks = [0, 1, 2, 3] * 3
    state.deck = list(np.random.permutation(ranks))
    state.deck_idx = 0
    
    for p in range(NUM_PLAYERS):
        state.hole_cards[p] = [state.deck[state.deck_idx]]
        state.deck_idx += 1
        
    stacks = list(state.stacks)
    bets = list(state.bets)
    stacks[1] -= SMALL_BLIND_BB
    bets[1] = SMALL_BLIND_BB
    stacks[2] -= BIG_BLIND_BB
    bets[2] = BIG_BLIND_BB
    state.stacks = tuple(stacks)
    state.bets = tuple(bets)
    state.pot = INITIAL_POT_BB
    state.current_player = 0
    state.last_raiser = 2
    state.last_raise_amount = BIG_BLIND_BB
    return state

def get_payoffs(state):
    return [state.stacks[p] - STARTING_STACK_BB for p in range(NUM_PLAYERS)]