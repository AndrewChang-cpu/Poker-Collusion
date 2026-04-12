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
    Return hashable info set key: (round_idx, bucket, actor_action_pairs).

    On preflop (round_idx == 0) the bucket component is the canonical hand ID
    in [0, 168] — full resolution, no bucketing.  On postflop streets the bucket
    comes from the equity-based abstraction tables.

    The history component contains (actor, action) pairs for the current street
    only (since the last DEAL sentinel). Including the actor disambiguates
    game states where different players have folded on prior streets — without
    this, two different active-player configurations can produce identical keys.
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
    street_actions = state.action_history[last_deal + 1:]

    # actor_history has no DEAL entries — offset by count of non-DEAL actions
    # before the current street to get the matching slice.
    num_prior_actions = sum(1 for a in state.action_history[:last_deal + 1] if a != _DEAL)
    street_actors = state.actor_history[num_prior_actions:]

    return (round_idx, bucket, tuple(zip(street_actors, street_actions)))