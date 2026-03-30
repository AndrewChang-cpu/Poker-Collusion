"""
Info set key: (card_bucket, action_history) with action indices and DEAL.
"""

from poker_collusion.abstraction.bucketing import get_bucket


def get_info_key(state, player):
    """
    Return hashable info set key: (bucket, tuple(action_history)).
    """
    hole = tuple(state.hole_cards[player])
    board = tuple(state.board)
    round_idx = state.round_idx
    
    # Ensure bucket is a hashable integer
    bucket = int(get_bucket(hole, board, round_idx))
    
    # Ensure history is a hashable tuple and contains no nested lists
    history = []
    for a in state.action_history:
        if isinstance(a, list):
            history.append(tuple(a))
        else:
            history.append(a)
            
    return (bucket, tuple(history))