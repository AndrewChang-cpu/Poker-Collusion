from poker_collusion.env.game_state import deal_new_hand
from poker_collusion.abstraction.info_set import get_info_key

def test_psychic_key_generation():
    state = deal_new_hand()
    p0_hole = state.hole_cards[0][0]
    p1_hole = state.hole_cards[1][0]
    
    # Standard Key (No team) -> must be a tuple
    standard_key = get_info_key(state, 0)
    assert standard_key[1] == (p0_hole,)
    assert isinstance(standard_key[1], tuple)
    
    # Psychic Key (Team [0, 1])
    psychic_key = get_info_key(state, 0, team_seats=[0, 1])
    # Bucket is (my_card, teammate_cards_tuple)
    assert psychic_key[1] == (p0_hole, (p1_hole,))
    assert isinstance(psychic_key[1], tuple)

def test_opponent_isolation():
    state = deal_new_hand()
    p2_hole = state.hole_cards[2][0]
    
    # P2 is the victim (not in team [0, 1])
    # Their key should follow the 1-element tuple standard
    key = get_info_key(state, 2, team_seats=[0, 1])
    assert key[1] == (p2_hole,)
    assert len(key[1]) == 1