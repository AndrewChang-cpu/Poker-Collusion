from poker_collusion.env.game_state import deal_new_hand
from poker_collusion.env.game_logic import apply_action, sample_chance, is_terminal

def test_leduc_full_game_flow():
    state = deal_new_hand()
    
    # Preflop: P0 calls (1), P1 calls (1), P2 checks (1)
    state = apply_action(state, 1) # P0
    state = apply_action(state, 1) # P1
    state = apply_action(state, 1) # P2
    
    assert state.chance_pending
    assert state.round_idx == 0
    
    # Deal Flop
    state = sample_chance(state)
    assert len(state.board) == 1
    assert state.round_idx == 1
    
    # Flop: P1 checks, P2 checks, P0 checks
    state = apply_action(state, 1)
    state = apply_action(state, 1)
    state = apply_action(state, 1)
    
    # Game must end here
    assert is_terminal(state)
    assert state.done

def test_fold_termination():
    state = deal_new_hand()
    # P0 raises to 3BB (action 4)
    state = apply_action(state, 4)
    # P1 folds (0), P2 folds (0)
    state = apply_action(state, 0)
    state = apply_action(state, 0)
    
    assert is_terminal(state)
    # P0 should win the pot
    assert state.stacks[0] == STARTING_STACK_BB + SMALL_BLIND_BB + BIG_BLIND_BB