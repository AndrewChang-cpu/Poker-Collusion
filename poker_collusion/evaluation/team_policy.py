"""
CentralizedToDecentralizedPolicy: wraps a full-comm trained CFRTrainer for
deployment without teammate communication.

At inference, the player does not know their teammate's card rank. We average
the centralized strategy over all possible teammate ranks, weighted by the
card-removal prior P(teammate_rank | my_rank, community_rank).

Full-comm key format:  (round_idx, my_rank, teammate_rank, community_rank, history)
Standard key format:   (round_idx, my_rank, -1,            community_rank, history)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

# Leduc deck constants (must match game_state.py)
_NUM_RANKS = 4
_NUM_SUITS = 3


class CentralizedToDecentralizedPolicy:
    """
    Implements the get_average_strategy interface expected by mbbg._get_policy_probs.
    Receives a STANDARD info key (my_rank, no teammate rank) from the game module
    and returns an action distribution averaged over all possible teammate ranks.
    """

    def __init__(self, trainer, team_seat: int) -> None:
        """
        trainer   : CFRTrainer loaded from a full-comm checkpoint
        team_seat : seat index (0=BTN, 1=SB) — used for validation only
        """
        self.trainer = trainer
        self.team_seat = team_seat

    def get_average_strategy(
        self, standard_key: tuple, legal_actions: List[int]
    ) -> np.ndarray:
        """
        standard_key = (round_idx, my_rank, community_rank, history)
        Returns probability distribution over legal_actions.
        """
        round_idx, my_rank, community_rank, history = standard_key
        prior = self._prior(my_rank, community_rank)

        weighted = np.zeros(len(legal_actions))
        total_weight = 0.0

        for r in range(_NUM_RANKS):
            w = prior[r]
            if w <= 0.0:
                continue
            full_key = (round_idx, my_rank, r, community_rank, history)
            strat = self.trainer.get_average_strategy(full_key, legal_actions)
            if strat is None:
                strat = np.ones(len(legal_actions)) / len(legal_actions)
            weighted += w * strat
            total_weight += w

        if total_weight > 0.0:
            return weighted / total_weight
        return np.ones(len(legal_actions)) / len(legal_actions)

    def _prior(self, my_rank: int, community_rank: int) -> np.ndarray:
        """
        Uniform prior over teammate rank, adjusted for card removal.
        P(teammate_rank = r) ∝ number of cards of rank r still in deck
        after removing my card and (if dealt) the community card.
        """
        counts = np.full(_NUM_RANKS, float(_NUM_SUITS))
        counts[my_rank] -= 1.0                       # my hole card
        if community_rank >= 0:
            counts[community_rank] -= 1.0            # community card
        counts = np.maximum(counts, 0.0)
        total = counts.sum()
        return counts / total if total > 0.0 else np.ones(_NUM_RANKS) / _NUM_RANKS
