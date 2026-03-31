"""
Info set key: (card_bucket, action_history) with action indices and DEAL.

Preflop: uses the canonical 169-hand ID directly (no information abstraction).
Postflop: uses equity-based bucket from precomputed tables.
"""

from poker_collusion.abstraction.bucketing import get_bucket, hole_to_canonical


def get_info_key(state, player):
    """
    Return hashable info set key: (bucket, tuple(action_history)).

    On preflop (round_idx == 0) the bucket component is the canonical hand ID
    in [0, 168] — full resolution, no bucketing.  On postflop streets the bucket
    comes from the equity-based abstraction tables.
    """
    hole = tuple(state.hole_cards[player])
    round_idx = state.round_idx

    if round_idx == 0:
        bucket = int(hole_to_canonical(hole))
    else:
        board = tuple(state.board)
        bucket = int(get_bucket(hole, board, round_idx))

    history = []
    for a in state.action_history:
        if isinstance(a, list):
            history.append(tuple(a))
        else:
            history.append(a)

    return (bucket, tuple(history))