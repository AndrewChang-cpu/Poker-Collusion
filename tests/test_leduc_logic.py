"""
Tests for Leduc Hold'em betting logic and street progression.
"""
from poker_collusion.env.game_state import deal_new_hand
from poker_collusion.env.game_logic import apply_action, sample_chance, is_terminal

def test_leduc_preflop_check_around():
    state = deal_new_hand()
    # P0 calls (1.0), P1 calls (1.0), P2 checks (1.0)
    state = apply_action(state, 1) # P0 calls
    state = apply_action(state, 1) # P1 calls
    state = apply_action(state, 1) # P2 checks/calls
    
    # Verify street is complete and pending chance node
    assert state.chance_pending
    assert state.round_idx == 0
    assert not state.done

def test_leduc_full_game_resolution():
    state = deal_new_hand()
    # Pre-flop action
    state = apply_action(state, 1)
    state = apply_action(state, 1)
    state = apply_action(state, 1)
    
    # Transition to Flop
    state = sample_chance(state)
    assert state.round_idx == 1
    assert len(state.board) == 1
    
    # Flop action: everyone checks
    state = apply_action(state, 1) # P1
    state = apply_action(state, 1) # P2
    state = apply_action(state, 1) # P0
    
    # Verify game terminal after 2nd street
    assert is_terminal(state)
    assert state.done