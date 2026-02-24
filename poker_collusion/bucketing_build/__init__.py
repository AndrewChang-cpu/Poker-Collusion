"""
One-off precomputation of bucket tables (preflop and postflop).
Run scripts/build_buckets.py to generate data/*.pkl.
"""

from poker_collusion.bucketing_build.preflop_table import build_preflop_table
from poker_collusion.bucketing_build.postflop_table import (
    build_flop_table,
    build_turn_table,
    build_river_table,
)

__all__ = [
    "build_preflop_table",
    "build_flop_table",
    "build_turn_table",
    "build_river_table",
]
