"""
Shared type aliases and Protocols for the poker_collusion package.
"""

from __future__ import annotations

from typing import Any, List, Literal, Protocol, Tuple, Union

# Action index 0..9 or community-card deal marker in history
ActionHistoryEntry = Union[int, Literal["DEAL"]]

# CFR info set: (card_bucket, immutable action sequence)
InfoSetKey = Tuple[int, Tuple[ActionHistoryEntry, ...]]

# Comparable 5-card hand score from hand_eval
HandScore = Tuple[int, ...]


class AbstractionState(Protocol):
    """Minimum state surface required by abstraction.actions."""

    current_player: int
    round_idx: int
    stacks: Tuple[float, ...]
    pot: float
    bets: Tuple[float, ...]
    active: Tuple[bool, ...]
    all_in: Tuple[bool, ...]
    last_raiser: int
    last_raise_amount: float


class CFRGame(Protocol):
    """Game module interface expected by CFRTrainer."""

    def deal_new_hand(self) -> Any: ...
    def get_current_player(self, state: Any) -> int: ...
    def get_legal_actions(self, state: Any) -> List[int]: ...
    def get_info_key(self, state: Any, player: int) -> Any: ...
    def is_terminal(self, state: Any) -> bool: ...
    def get_payoffs(self, state: Any) -> List[float]: ...
    def apply_action(self, state: Any, action_index: int) -> Any: ...
    def is_chance_node(self, state: Any) -> bool: ...
    def sample_chance(self, state: Any) -> Any: ...
