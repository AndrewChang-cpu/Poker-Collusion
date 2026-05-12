"""
Regret matching and average strategy extraction for CFR.
"""

import os
import pickle
import numpy as np
from poker_collusion.config import NUM_ACTIONS


def regret_matching(regret_sum, num_actions):
    if num_actions <= 0: return np.array([])
    regrets = np.asarray(regret_sum) if len(regret_sum) >= num_actions else np.zeros(num_actions)
    positive = np.maximum(regrets[:num_actions], 0)
    total = positive.sum()
    return positive / total if total > 0 else np.ones(num_actions) / num_actions


def get_average_strategy(strategy_sum, num_actions):
    s = np.asarray(strategy_sum)[:num_actions]
    total = s.sum()
    return s / total if total > 0 else np.ones(num_actions) / num_actions


class Strategy:
    def __init__(self, strategy_sum=None, action_map=None):
        self.strategy_sum = strategy_sum or {}
        self.action_map = action_map or {}

    @classmethod
    def load(cls, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Strategy file not found: {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(strategy_sum=data.get("strategy_sum", {}), action_map=data.get("action_map", {}))

    def get_action_probabilities(self, info_key, legal_actions):
        if isinstance(info_key, list):
            info_key = tuple(info_key)

        if info_key not in self.strategy_sum:
            return np.ones(len(legal_actions)) / len(legal_actions)

        s = self.strategy_sum[info_key]
        s_sub = np.array([s[a] if a < len(s) else 0.0 for a in legal_actions])
        return get_average_strategy(s_sub, len(legal_actions))

    def get_average_strategy(self, info_key, legal_actions):
        """Alias for get_action_probabilities (matches CFRTrainer interface)."""
        return self.get_action_probabilities(info_key, legal_actions)

    def sample_action(self, info_key, legal_actions):
        probs = self.get_action_probabilities(info_key, legal_actions)
        return np.random.choice(legal_actions, p=probs)