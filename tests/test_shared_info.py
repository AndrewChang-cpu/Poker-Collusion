"""
Tests for standardized Info Set keys and Psychic variant isolation.
"""
from poker_collusion.env.game_state import deal_new_hand
from poker_collusion.abstraction.info_set import get_info_key

def test_standardized_keys():
    state = deal_new_hand()
    p0_hole = state.hole_cards[0][0]
    
    # Verify Step 2 Fix: bucket must always be a tuple
    key = get_info_key(state, 0)
    assert isinstance(key[1], tuple)
    assert key[1] == (p0_hole,)

def test_psychic_key_integrity():
    state = deal_new_hand()
    p0_hole = state.hole_cards[0][0]
    p1_hole = state.hole_cards[1][0]
    p2_hole = state.hole_cards[2][0]
    
    # P0 and P1 are teammates
    team = [0, 1]
    p0_key = get_info_key(state, 0, team_seats=team)
    p2_key = get_info_key(state, 2, team_seats=team)
    
    # Teammate sees other's card
    assert p0_key[1] == (p0_hole, (p1_hole,))
    # Opponent (P2) is isolated and sees only their own
    assert p2_key[1] == (p2_hole,)