"""
Leduc Info Set: (round_idx, direct_rank_bucket, history).
"""

_DEAL = "DEAL"

def get_info_key(state, player):
    """
    Generate hashable keys using direct card ranks for Leduc Hold'em.
    Round 0 (Preflop): bucket = rank (0-3).
    Round 1 (Flop): bucket = (hole_rank * 4) + board_rank.
    """
    hole_rank = state.hole_cards[player][0]
    round_idx = state.round_idx

    if round_idx == 0:
        bucket = hole_rank
    else:
        board_rank = state.board[0]
        bucket = (hole_rank * 4) + board_rank

    history = []
    actor_idx = 0
    for a in state.action_history:
        if a == _DEAL:
            history.append(_DEAL)
        else:
            history.append((state.actor_history[actor_idx], a))
            actor_idx += 1
            
    return (round_idx, bucket, tuple(history))