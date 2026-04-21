from poker_collusion.env.game_state import deal_new_hand
from poker_collusion.env.game_logic import apply_action, sample_chance
from poker_collusion.abstraction.info_set import get_info_key

def test_preflop_info_key():
    state = deal_new_hand()
    hole = state.hole_cards[0][0]
    key = get_info_key(state, 0)
    
    # (round_idx, bucket, history)
    assert key[0] == 0
    assert key[1] == hole
    assert isinstance(key[2], tuple)

def test_flop_info_key():
    state = deal_new_hand()
    # Advance to Flop
    state = apply_action(state, 1)
    state = apply_action(state, 1)
    state = apply_action(state, 1)
    state = sample_chance(state)
    
    hole = state.hole_cards[0][0]
    board = state.board[0]
    key = get_info_key(state, 0)
    
    assert key[0] == 1
    # bucket = hole * 4 + board
    assert key[1] == (hole * 4) + board
    # History should contain "DEAL"
    assert "DEAL" in key[2]

def test_info_key_determinism():
    state = deal_new_hand()
    key1 = get_info_key(state, 0)
    key2 = get_info_key(state, 0)
    assert key1 == key2