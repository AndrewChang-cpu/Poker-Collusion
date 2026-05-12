"""
Bucket lookup: (hole_cards, board, round) -> bucket id.
Updated for 12-card Leduc Hold'em logic.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

from poker_collusion.config import (
    PREFLOP_BUCKETS,
    FLOP_BUCKETS,
    TURN_BUCKETS,
    RIVER_BUCKETS,
    DEFAULT_BUCKET_DIR,
    PREFLOP_BUCKETS_FILE,
    FLOP_BUCKETS_FILE,
    TURN_BUCKETS_FILE,
    RIVER_BUCKETS_FILE,
)

_preflop_table: Optional[Dict[int, int]] = None
_flop_centers: Optional[List[float]] = None
_turn_centers: Optional[List[float]] = None
_river_centers: Optional[List[float]] = None
_equity_cache: Dict[Tuple, float] = {}


def _path(filename: str) -> str:
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, DEFAULT_BUCKET_DIR, filename)


def _load_tables() -> None:
    global _preflop_table, _flop_centers, _turn_centers, _river_centers
    if _preflop_table is not None:
        return
    try:
        import pickle
        p = _path(PREFLOP_BUCKETS_FILE)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                _preflop_table = pickle.load(f)
        p = _path(FLOP_BUCKETS_FILE)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                _flop_centers = pickle.load(f)
        # Turn/River placeholders kept for import compatibility
    except Exception:
        pass


def get_bucket(
    hole_cards: Sequence[int], board: Sequence[int], round_idx: int
) -> int:
    """
    Return bucket id in [0, n_buckets-1] for Leduc (hole_cards, board).
    hole_cards: tuple of 1 rank (0-3).
    board: tuple of 0 or 1 rank.
    round_idx: 0=preflop, 1=flop.
    """
    _load_tables()
    if round_idx == 0:
        if _preflop_table is not None:
            canonical = hole_to_canonical(hole_cards)
            return _preflop_table.get(canonical, 0) % PREFLOP_BUCKETS
        return _preflop_fallback(hole_cards) % PREFLOP_BUCKETS
    
    if round_idx == 1:
        if _flop_centers is not None:
            # For Leduc, board_len is always 1 postflop
            return _equity_to_bucket(hole_cards, board, 1, _flop_centers, FLOP_BUCKETS)
        return _postflop_fallback(hole_cards, board, FLOP_BUCKETS)
    
    return 0


def hole_to_canonical(hole_cards: Sequence[int]) -> int:
    """Leduc mapping: The rank of the single card is its canonical ID (0-3)."""
    return hole_cards[0]


def _preflop_fallback(hole_cards: Sequence[int], num_buckets: int = PREFLOP_BUCKETS) -> int:
    """Leduc fallback: Direct rank mapping."""
    return hole_cards[0] % num_buckets


def _postflop_fallback(
    hole_cards: Sequence[int], board: Sequence[int], num_buckets: int
) -> int:
    """Leduc fallback: (hole * 4) + board."""
    if not board:
        return 0
    return ((hole_cards[0] * 4) + board[0]) % num_buckets


def _equity_to_bucket(
    hole_cards: Sequence[int],
    board: Sequence[int],
    board_len: int,
    centers: Optional[List[float]],
    num_buckets: int,
) -> int:
    """Assign bucket by nearest cluster center (equity)."""
    eq = _estimate_equity(hole_cards, board, board_len)
    if centers is None or len(centers) == 0:
        return int(eq * num_buckets) % num_buckets
    best = 0
    best_dist = abs(eq - centers[0])
    for i, c in enumerate(centers):
        d = abs(eq - c)
        if d < best_dist:
            best_dist = d
            best = i
    return best % num_buckets


def _estimate_equity(
    hole_cards: Sequence[int],
    board: Sequence[int],
    board_len: int,
    n_rollouts: int = 1000,
) -> float:
    """
    Deterministic Leduc MC equity estimate vs random opponent (0..1).
    Uses 12-card deck and Leduc 2-card hand rules.
    """
    from poker_collusion.env.hand_eval import evaluate_hand
    cache_key: Tuple = (tuple(sorted(hole_cards)), tuple(sorted(board[:board_len])))
    if cache_key in _equity_cache:
        return _equity_cache[cache_key]

    seed = 0
    for v in cache_key[0] + cache_key[1]:
        seed = seed * 53 + int(v) + 1
    seed &= 0xFFFFFFFF
    rng = random.Random(seed)

    # Leduc specific: 4 ranks * 3 suits = 12 cards
    used = set(hole_cards) | set(board[:board_len])
    deck = [c for c in [0, 1, 2, 3]*3 if c not in used] # Simplified deck for equity logic
    
    cards_needed = 1 - board_len
    wins = 0.0
    for _ in range(n_rollouts):
        rest = list(deck)
        rng.shuffle(rest)
        
        # Opponent gets 1 hole card
        opp = [rest[0]]
        # Board gets remaining needed cards
        runout = rest[1:1 + cards_needed]
        full_board = list(board[:board_len]) + list(runout)
        
        if not full_board:
            continue
            
        my_hand = evaluate_hand(list(hole_cards) + full_board)
        opp_hand = evaluate_hand(list(opp) + full_board)
        
        if my_hand > opp_hand:
            wins += 1.0
        elif my_hand == opp_hand:
            wins += 0.5
            
    result = wins / n_rollouts
    _equity_cache[cache_key] = result
    return result