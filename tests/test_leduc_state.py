import numpy as np
from poker_collusion.env.game_state import deal_new_hand
from poker_collusion.config import STARTING_STACK_BB, SMALL_BLIND_BB, BIG_BLIND_BB

def test_deal_leduc_hand():
    state = deal_new_hand()
    # Deck check (12 cards total, 3 dealt as hole cards)
    assert len(state.deck) == 12
    assert state.deck_idx == 3
    
    # Hole card check (1 card per player)
    assert len(state.hole_cards) == 3
    for p in range(3):
        assert len(state.hole_cards[p]) == 1
        assert 0 <= state.hole_cards[p][0] <= 3

def test_initial_blinds():
    state = deal_new_hand()
    # P1 is SB, P2 is BB
    assert state.stacks[1] == STARTING_STACK_BB - SMALL_BLIND_BB
    assert state.stacks[2] == STARTING_STACK_BB - BIG_BLIND_BB
    assert state.bets[1] == SMALL_BLIND_BB
    assert state.bets[2] == BIG_BLIND_BB
    assert state.pot == SMALL_BLIND_BB + BIG_BLIND_BB

def test_state_copy_isolation():
    state = deal_new_hand()
    scopy = state.copy()
    scopy.board.append(1)
    assert len(state.board) == 0
    assert len(scopy.board) == 1