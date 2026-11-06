"""
Compact, read-only average strategy extracted from an MCCFR checkpoint.

The full checkpoint holds regrets too (~1 GB resident for the 5M baseline);
the RL opponent only needs normalized average-strategy rows.
"""

from __future__ import annotations

import pickle
from typing import List

import numpy as np

from poker_collusion.abstraction.info_set import get_info_key


def _plain(x):
    if isinstance(x, tuple):
        return tuple(_plain(v) for v in x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _context(info_key) -> tuple:
    """(round, card bucket, facing a bet): replays the public history to get to_call."""
    from poker_collusion.env import game_logic as g
    from poker_collusion.env.game_state import deal_new_hand
    round_idx, bucket, history = info_key
    s = deal_new_hand()
    for entry in history:
        s = g.sample_chance(s) if entry == "DEAL" else g.apply_action(s, entry[1])
    return (round_idx, bucket, max(s.bets) - s.bets[s.current_player] > 0)


def compact_checkpoint(src: str, dst: str) -> int:
    with open(src, "rb") as f:
        data = pickle.load(f)
    keys, rows = [], []
    for k, v in data["strategy_sum"].items():
        total = float(np.sum(v))
        if total > 0:
            keys.append(_plain(k))
            rows.append(np.asarray(v, dtype=np.float64) / total)
    table = np.asarray(rows, dtype=np.float32)
    # Card-aware fallback for unseen infosets: mean strategy over seen infosets
    # with the same round, card bucket and facing-bet context.
    sums: dict = {}
    for k, row in zip(keys, rows):
        c = _context(k)
        acc = sums.setdefault(c, [np.zeros(len(row)), 0])
        acc[0] += row
        acc[1] += 1
    fallback = {c: (v / n).astype(np.float32) for c, (v, n) in sums.items()}
    with open(dst, "wb") as f:
        pickle.dump({"keys": keys, "table": table, "source": src, "fallback": fallback,
                     "iteration": data["iteration"]}, f)
    return len(keys)


class FrozenPolicy:
    """Callable opponent: (state, seat, legal) -> probs over legal actions."""

    def __init__(self, path: str, fallback: str = "context"):
        """fallback: 'context' (card-aware mean, default) or 'uniform' (CFRTrainer behavior)."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.index = {k: i for i, k in enumerate(data["keys"])}
        self.table = data["table"]
        self.fallback_mode = fallback
        self.fallback = data.get("fallback", {})
        self.iteration = data["iteration"]
        self.lookups = 0
        self.misses = 0

    def probs(self, info_key, legal: List[int], facing: bool = False) -> np.ndarray:
        self.lookups += 1
        key = _plain(info_key)
        i = self.index.get(key)
        if i is not None:
            p = self.table[i, legal].astype(np.float64)
            if p.sum() > 0:
                return p / p.sum()
        self.misses += 1
        if self.fallback_mode == "context":
            row = self.fallback.get((key[0], key[1], facing))
            if row is not None and row[legal].sum() > 0:
                p = row[legal].astype(np.float64)
                return p / p.sum()
        return np.full(len(legal), 1.0 / len(legal))  # CFRTrainer's behavior

    def __call__(self, state, seat: int, legal: List[int]) -> np.ndarray:
        facing = max(state.bets) - state.bets[seat] > 0
        return self.probs(get_info_key(state, seat), legal, facing)

