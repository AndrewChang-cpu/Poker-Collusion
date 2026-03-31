"""
Info set key: (card_bucket, action_history) with action indices and DEAL.

Preflop: uses the canonical 169-hand ID directly (no information abstraction).
Postflop: uses equity-based bucket from precomputed tables.
"""

from poker_collusion.abstraction.bucketing import get_bucket, hole_to_canonical

# Sentinel that matches game_state.DEAL — defined here to avoid a circular import
# (env.__init__ imports get_info_key; game_state is part of env).
_DEAL = "DEAL"


def get_info_key(state, player):
    """
    Return hashable info set key: (round_idx, bucket, within_street_action_history).

    On preflop (round_idx == 0) the bucket component is the canonical hand ID
    in [0, 168] — full resolution, no bucketing.  On postflop streets the bucket
    comes from the equity-based abstraction tables.

    The history component contains only the actions taken on the current street
    (since the last DEAL sentinel). Prior-street actions are dropped to keep the
    infoset space tractable.
    """
    hole = tuple(state.hole_cards[player])
    round_idx = state.round_idx

    if round_idx == 0:
        bucket = int(hole_to_canonical(hole))
    else:
        board = tuple(state.board)
        bucket = int(get_bucket(hole, board, round_idx))

    # Find the start of the current street: the position after the last DEAL.
    last_deal = -1
    for i, a in enumerate(state.action_history):
        if a == _DEAL:
            last_deal = i
    street_history = state.action_history[last_deal + 1:]

    return (round_idx, bucket, tuple(street_history))