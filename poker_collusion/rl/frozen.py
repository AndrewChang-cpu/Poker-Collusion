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
    with open(dst, "wb") as f:
        pickle.dump({"keys": keys, "table": table, "source": src,
                     "iteration": data["iteration"]}, f)
    return len(keys)


class FrozenPolicy:
    """Callable opponent: (state, seat, legal) -> probs over legal actions."""

    def __init__(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.index = {k: i for i, k in enumerate(data["keys"])}
        self.table = data["table"]
        self.iteration = data["iteration"]
        self.lookups = 0
        self.misses = 0

    def probs(self, info_key, legal: List[int]) -> np.ndarray:
        self.lookups += 1
        i = self.index.get(_plain(info_key))
        if i is None:
            self.misses += 1  # unseen infoset: same uniform fallback as CFRTrainer
            return np.full(len(legal), 1.0 / len(legal))
        p = self.table[i, legal].astype(np.float64)
        s = p.sum()
        if s <= 0:
            self.misses += 1
            return np.full(len(legal), 1.0 / len(legal))
        return p / s

    def __call__(self, state, seat: int, legal: List[int]) -> np.ndarray:
        return self.probs(get_info_key(state, seat), legal)

