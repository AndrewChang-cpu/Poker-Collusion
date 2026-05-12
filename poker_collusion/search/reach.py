"""
Reach probability tracker for online search.

Maintains a probability vector over all 1326 possible hole-card pairs,
updated after each action the bot takes.  Used by the subgame solver to
weight the chance node at the root of the depth-limited subgame.

Encoding: pairs are indexed by the *combination index* of two cards drawn
from a 52-card deck.  For cards c0 < c1 (both in 0..51) the index is
    idx = c0 * (103 - c0) // 2 + c1 - c0 - 1
which bijects to [0, 1325].  Helper functions `pair_index` and
`index_to_pair` convert between the two representations.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Set

import numpy as np

NUM_PAIRS = 1326  # C(52, 2)


def pair_index(c0: int, c1: int) -> int:
    """Return the combination index in [0, 1325] for two card ids (order-free)."""
    lo, hi = (c0, c1) if c0 < c1 else (c1, c0)
    return lo * (103 - lo) // 2 + hi - lo - 1


def index_to_pair(idx: int) -> tuple[int, int]:
    """Inverse of pair_index: return (c0, c1) with c0 < c1."""
    lo = 0
    while True:
        span = 51 - lo
        if idx < span:
            return lo, lo + 1 + idx
        idx -= span
        lo += 1


_PAIR_TABLE: Optional[np.ndarray] = None


def _build_pair_table() -> np.ndarray:
    """Precompute a (1326, 2) array mapping index -> (c0, c1)."""
    global _PAIR_TABLE
    if _PAIR_TABLE is not None:
        return _PAIR_TABLE
    table = np.empty((NUM_PAIRS, 2), dtype=np.int32)
    i = 0
    for c0 in range(52):
        for c1 in range(c0 + 1, 52):
            table[i] = (c0, c1)
            i += 1
    _PAIR_TABLE = table
    return _PAIR_TABLE


class ReachTracker:
    """
    Tracks reach probabilities for one bot over all 1326 hole-card pairs.

    Usage per hand:
        tracker.reset(visible_cards)       # at hand start
        tracker.update(action, strategy_fn) # after each of the bot's actions
        probs = tracker.probs              # current reach distribution
    """

    __slots__ = ("_probs",)

    def __init__(self) -> None:
        self._probs = np.zeros(NUM_PAIRS, dtype=np.float64)

    @property
    def probs(self) -> np.ndarray:
        """Current reach probability vector (length 1326, sums to 1)."""
        return self._probs

    def reset(self, visible_cards: Set[int]) -> None:
        """
        Initialise to uniform over hands that don't contain any visible card.

        visible_cards: set of card indices the bot can see (its own hole cards
        + board cards at the time of initialisation).  For the bot's opponents'
        perspective during subgame construction this includes only board cards.
        """
        table = _build_pair_table()
        mask = np.ones(NUM_PAIRS, dtype=np.float64)
        for card in visible_cards:
            mask[(table[:, 0] == card) | (table[:, 1] == card)] = 0.0
        total = mask.sum()
        if total > 0:
            self._probs = mask / total
        else:
            self._probs = np.zeros(NUM_PAIRS, dtype=np.float64)

    def update(
        self,
        action_index: int,
        action_probs_by_pair: np.ndarray,
    ) -> None:
        """
        Update reach probs after the bot takes *action_index*.

        action_probs_by_pair: shape (1326,) giving P(action_index | pair i)
            for each hole-card pair *i*.  Zero for infeasible pairs is fine.
        """
        self._probs *= action_probs_by_pair
        total = self._probs.sum()
        if total > 0:
            self._probs /= total

    def zero_out(self, dead_cards: Set[int]) -> None:
        """Zero out pairs that contain any of *dead_cards* and renormalize."""
        table = _build_pair_table()
        for card in dead_cards:
            self._probs[(table[:, 0] == card) | (table[:, 1] == card)] = 0.0
        total = self._probs.sum()
        if total > 0:
            self._probs /= total

    def feasible_pairs(self) -> np.ndarray:
        """Return indices of pairs with non-zero reach probability."""
        return np.nonzero(self._probs > 0)[0]

    def feasible_hands(self) -> List[tuple[int, int]]:
        """Return list of (c0, c1) pairs with non-zero reach probability."""
        table = _build_pair_table()
        idxs = self.feasible_pairs()
        return [(int(table[i, 0]), int(table[i, 1])) for i in idxs]
