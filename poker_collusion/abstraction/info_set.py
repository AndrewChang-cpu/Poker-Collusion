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
    Return hashable info set key: (round_idx, bucket, history).

    On preflop (round_idx == 0) the bucket component is the canonical hand ID
    in [0, 168] — full resolution, no bucketing.  On postflop streets the bucket
    comes from the equity-based abstraction tables.

    The history component contains all (actor, action) pairs from every street,
    with 'DEAL' sentinels marking street boundaries. This enables multi-street
    collusion strategies (e.g. conditioning river play on a teammate's flop
    aggression).

    Old strategies trained without full history are handled transparently via
    key translation in CFRTrainer.get_average_strategy().
    """
    hole = tuple(state.hole_cards[player])
    round_idx = state.round_idx

    if round_idx == 0:
        bucket = int(hole_to_canonical(hole))
    else:
        board = tuple(state.board)
        bucket = int(get_bucket(hole, board, round_idx))

    # All streets: (actor, action) pairs with DEAL markers as street separators.
    history = []
    actor_idx = 0
    for a in state.action_history:
        if a == _DEAL:
            history.append(_DEAL)
        else:
            history.append((state.actor_history[actor_idx], a))
            actor_idx += 1
    assert actor_idx == len(state.actor_history), (
        f"actor_history length {len(state.actor_history)} does not match "
        f"non-DEAL action count {actor_idx}. State history is inconsistent."
    )
    return (round_idx, bucket, tuple(history))