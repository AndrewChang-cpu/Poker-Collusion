"""
Regret matching and average strategy extraction for CFR.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np

RegretLike = Union[Sequence[float], np.ndarray]


def regret_matching(regret_sum: RegretLike, num_actions: int) -> np.ndarray:
    """
    Convert cumulative regrets to strategy probabilities.
    positive_regret[a] = max(regret_sum[a], 0); then normalize.
    If sum(positive) == 0, return uniform.
    """
    if num_actions <= 0:
        return np.array([])
    regrets = np.asarray(regret_sum) if len(regret_sum) >= num_actions else np.zeros(num_actions)
    if len(regrets) < num_actions:
        regrets = np.resize(regrets, num_actions)
    positive = np.maximum(regrets[:num_actions], 0)
    total = positive.sum()
    if total > 0:
        return positive / total
    return np.ones(num_actions) / num_actions


def get_average_strategy(strategy_sum: RegretLike, num_actions: int) -> np.ndarray:
    """
    Normalized cumulative strategy = blueprint.
    If sum is 0, return uniform.
    """
    s = np.asarray(strategy_sum)
    if len(s) < num_actions:
        s = np.resize(s, num_actions)
    s = s[:num_actions]
    total = s.sum()
    if total > 0:
        return s / total
    return np.ones(num_actions) / num_actions
