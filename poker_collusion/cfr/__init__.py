"""
MCCFR trainer: external sampling, Linear CFR, regret pruning.
"""

from poker_collusion.cfr.trainer import CFRTrainer
from poker_collusion.cfr.strategy import regret_matching, get_average_strategy
from poker_collusion.cfr.debug import CFRDebugger

__all__ = [
    "CFRTrainer",
    "regret_matching",
    "get_average_strategy",
    "CFRDebugger",
]
