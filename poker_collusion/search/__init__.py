"""
Online real-time search: reach probabilities, subgame construction,
continuation strategies, depth-limited CFR solver, and PluribusBot agent.
"""

from poker_collusion.search.reach import ReachTracker, pair_index, index_to_pair
from poker_collusion.search.continuation import get_continuation_strategy
from poker_collusion.search.solver import SubgameSolver
from poker_collusion.search.bot import PluribusBot
from poker_collusion.search.play import play_hand, run_match, print_results

__all__ = [
    "ReachTracker",
    "pair_index",
    "index_to_pair",
    "get_continuation_strategy",
    "SubgameSolver",
    "PluribusBot",
    "play_hand",
    "run_match",
    "print_results",
]
