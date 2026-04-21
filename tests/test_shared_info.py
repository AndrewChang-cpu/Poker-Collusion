from poker_collusion.env.game_state import deal_new_hand
from poker_collusion.abstraction.info_set import get_info_key

def test_psychic_key_generation():
    state = deal_new_hand()
    p0_hole = state.hole_cards[0][0]
    p1_hole = state.hole_cards[1][0]
    
    # Standard Key (No team)
    standard_key = get_info_key(state, 0)
    assert standard_key[1] == p0_hole
    
    # Psychic Key (Team [0, 1])
    psychic_key = get_info_key(state, 0, team_seats=[0, 1])
    # Bucket should be (my_card, (teammate_card,))
    assert psychic_key[1] == (p0_hole, (p1_hole,))

def test_opponent_isolation():
    state = deal_new_hand()
    p2_hole = state.hole_cards[2][0]
    
    # P2 is the victim (not in team [0, 1])
    # Their key should be standard even if team_seats are passed
    key = get_info_key(state, 2, team_seats=[0, 1])
    assert key[1] == p2_hole
    assert not isinstance(key[1], tuple)