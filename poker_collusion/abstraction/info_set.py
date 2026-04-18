"""
Info set key generation with canonicalization.
"""

from poker_collusion.abstraction.bucketing import get_bucket


def get_info_key(state, player):
    """
    Return hashable info set key: (bucket, tuple(action_history)).
    Canonicalizes hole cards by sorting.
    """
    # Canonicalize: (A, K) and (K, A) result in identical info sets
    hole = tuple(sorted(state.hole_cards[player]))
    board = tuple(state.board)
    round_idx = state.round_idx
    bucket = get_bucket(hole, board, round_idx)
    history = tuple(state.action_history)
    return (bucket, history)