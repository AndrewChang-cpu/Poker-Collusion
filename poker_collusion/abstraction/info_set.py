"""
Info set key: (card_bucket, action_history) with action indices and DEAL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from poker_collusion.abstraction.bucketing import get_bucket
from poker_collusion.typing_defs import InfoSetKey

if TYPE_CHECKING:
    from poker_collusion.env.game_state import NLHEState


def get_info_key(state: NLHEState, player: int) -> InfoSetKey:
    """
    Return hashable info set key: (bucket, tuple(action_history)).
    state must have: hole_cards, board, round_idx, action_history.
    """
    hole = tuple(state.hole_cards[player])
    board = tuple(state.board)
    round_idx = state.round_idx
    bucket = get_bucket(hole, board, round_idx)
    return (bucket, state.action_history)
