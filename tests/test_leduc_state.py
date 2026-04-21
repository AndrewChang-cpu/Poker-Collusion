"""
Tests for Leduc Hold'em game state and initialization.
"""
from poker_collusion.env.game_state import deal_new_hand
from poker_collusion.config import STARTING_STACK_BB, SMALL_BLIND_BB, BIG_BLIND_BB, INITIAL_POT_BB

def test_deal_leduc_hand():
    state = deal_new_hand()
    # Verify 12-card deck (4 ranks * 3 suits)
    assert len(state.deck) == 12
    assert state.deck_idx == 3  # 3 hole cards dealt
    
    # Verify hole cards (1 per player)
    assert len(state.hole_cards) == 3
    for p in range(3):
        assert len(state.hole_cards[p]) == 1

def test_initialization_fix():
    state = deal_new_hand()
    # Verify Step 1 Fix: last_raiser must be -1
    assert state.last_raiser == -1
    # Verify Step 3 Fix: blinds are rounded
    assert state.bets[1] == SMALL_BLIND_BB
    assert state.bets[2] == BIG_BLIND_BB
    assert state.pot == INITIAL_POT_BB