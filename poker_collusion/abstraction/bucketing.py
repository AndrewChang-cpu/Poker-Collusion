"""
Bucket lookup: (hole_cards, board, round) -> bucket id.
Loads precomputed tables from data/ when available; fallback to 0.
"""

import os
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

_preflop_table = None
_flop_centers = None
_turn_centers = None
_river_centers = None


def _path(filename):
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, DEFAULT_BUCKET_DIR, filename)


def _load_tables():
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
        p = _path(TURN_BUCKETS_FILE)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                _turn_centers = pickle.load(f)
        p = _path(RIVER_BUCKETS_FILE)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                _river_centers = pickle.load(f)
    except Exception:
        pass


def get_bucket(hole_cards, board, round_idx):
    """
    Return bucket id in [0, n_buckets-1] for (hole_cards, board) at given round.
    hole_cards: tuple of 2 ints (card indices 0-51).
    board: tuple of 0, 3, 4, or 5 ints.
    round_idx: 0=preflop, 1=flop, 2=turn, 3=river.
    """
    _load_tables()
    if round_idx == 0:
        if _preflop_table is not None:
            canonical = _hole_to_canonical(hole_cards)
            return _preflop_table.get(canonical, 0) % PREFLOP_BUCKETS
        return _preflop_fallback(hole_cards) % PREFLOP_BUCKETS
    if round_idx == 1 and len(board) >= 3:
        if _flop_centers is not None:
            return _equity_to_bucket(hole_cards, board, 3, _flop_centers, FLOP_BUCKETS)
        return _postflop_fallback(hole_cards, board, FLOP_BUCKETS)
    if round_idx == 2 and len(board) >= 4:
        if _turn_centers is not None:
            return _equity_to_bucket(hole_cards, board, 4, _turn_centers, TURN_BUCKETS)
        return _postflop_fallback(hole_cards, board, TURN_BUCKETS)
    if round_idx == 3 and len(board) >= 5:
        if _river_centers is not None:
            return _equity_to_bucket(hole_cards, board, 5, _river_centers, RIVER_BUCKETS)
        return _postflop_fallback(hole_cards, board, RIVER_BUCKETS)
    return 0


def _hole_to_canonical(hole_cards):
    """Map 2 cards to 169 canonical hand id (0..168). Matches bucketing_build.preflop_table."""
    r0, r1 = hole_cards[0] % 13, hole_cards[1] % 13
    s0, s1 = hole_cards[0] // 13, hole_cards[1] // 13
    high, low = max(r0, r1), min(r0, r1)
    if high == low:
        return high
    suited = 1 if s0 == s1 else 0
    return 13 + (high - 1) * high + 2 * low + (0 if suited else 1)


def _preflop_fallback(hole_cards, num_buckets=PREFLOP_BUCKETS):
    """Simple rank-based fallback when no table loaded."""
    r0, r1 = hole_cards[0] % 13, hole_cards[1] % 13
    high, low = max(r0, r1), min(r0, r1)
    suited = (hole_cards[0] // 13) == (hole_cards[1] // 13)
    score = high * 13 + low
    if high == low:
        score += 100
    if suited:
        score += 20
    return int((score / (12 * 13 + 12 + 100 + 20 + 1)) * num_buckets) % num_buckets


def _postflop_fallback(hole_cards, board, num_buckets):
    """Fallback: use hand category. Requires hand_eval."""
    from poker_collusion.env.hand_eval import evaluate_hand
    cards = list(hole_cards) + list(board)
    if len(cards) < 5:
        return 0
    score = evaluate_hand(cards)
    category = score[0]
    return int((category / 9.0) * num_buckets) % num_buckets


def _equity_to_bucket(hole_cards, board, board_len, centers, num_buckets):
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


def _estimate_equity(hole_cards, board, board_len, n_rollouts=100):
    """Monte Carlo equity estimate vs random opponent (0..1)."""
    import random
    from poker_collusion.env.hand_eval import evaluate_hand
    used = set(hole_cards) | set(board[:board_len])
    deck = [c for c in range(52) if c not in used]
    wins = 0
    for _ in range(n_rollouts):
        rest = list(deck)
        random.shuffle(rest)
        opp = tuple(rest[:2])
        if board_len == 3:
            runout = rest[2:7]
        elif board_len == 4:
            runout = rest[2:6]
        else:
            runout = []
        full_board = list(board[:board_len]) + list(runout)
        if len(full_board) < 5:
            continue
        my_hand = evaluate_hand(list(hole_cards) + full_board)
        opp_hand = evaluate_hand(list(opp) + full_board)
        if my_hand > opp_hand:
            wins += 1
        elif my_hand == opp_hand:
            wins += 0.5
    return wins / n_rollouts
