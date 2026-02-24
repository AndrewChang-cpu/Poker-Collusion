"""
Build preflop bucket table: 169 canonical hands -> 15 buckets by equity vs random.
"""

import random
import numpy as np
from poker_collusion.env.hand_eval import evaluate_hand
from poker_collusion.config import PREFLOP_BUCKETS


def enumerate_canonical_hands():
    """
    Yield (canonical_id, card0, card1) for each of 169 types.
    canonical_id: 0..12 for pairs (2-2 .. A-A), then suited/offsuit combos.
    """
    # Pairs: 13. Use same suit.
    for r in range(13):
        c0, c1 = r * 13, r * 13 + 1  # same suit
        yield r, c0, c1
    # Non-pairs: high > low. Suited then offsuit for each (high, low).
    idx = 13
    for high in range(1, 13):
        for low in range(high):
            # Suited
            c0, c1 = high * 13, low * 13
            yield idx, c0, c1
            idx += 1
            # Offsuit
            c0, c1 = high * 13, low * 13 + 1
            yield idx, c0, c1
            idx += 1


def equity_vs_random(hole0, hole1, n_rollouts=1000):
    """All-in equity vs one random opponent hand (0..1)."""
    used = {hole0, hole1}
    deck = [c for c in range(52) if c not in used]
    wins = 0
    for _ in range(n_rollouts):
        random.shuffle(deck)
        opp = (deck[0], deck[1])
        board = tuple(deck[2:7])
        my_hand = evaluate_hand([hole0, hole1] + list(board))
        opp_hand = evaluate_hand(list(opp) + list(board))
        if my_hand > opp_hand:
            wins += 1
        elif my_hand == opp_hand:
            wins += 0.5
    return wins / n_rollouts


def build_preflop_table(n_rollouts=1000, num_buckets=None):
    """
    Build mapping canonical_id -> bucket in [0, num_buckets-1].
    Equal-frequency binning by equity.
    """
    num_buckets = num_buckets or PREFLOP_BUCKETS
    pairs = []  # (canonical_id, equity)
    for cid, c0, c1 in enumerate_canonical_hands():
        eq = equity_vs_random(c0, c1, n_rollouts)
        pairs.append((cid, eq))
    pairs.sort(key=lambda x: x[1])
    n = len(pairs)
    table = {}
    for i, (cid, _) in enumerate(pairs):
        table[cid] = min(i * num_buckets // n, num_buckets - 1)
    return table


def canonical_from_hole(hole_cards):
    """Map 2 cards to canonical id 0..168 (same as enumerate order)."""
    r0, r1 = hole_cards[0] % 13, hole_cards[1] % 13
    s0, s1 = hole_cards[0] // 13, hole_cards[1] // 13
    high, low = max(r0, r1), min(r0, r1)
    if high == low:
        return high
    suited = 1 if s0 == s1 else 0
    # Non-pair index: 13 + (high-1)*high//2*2 + (high-1-low)*2? Actually enumerate: pairs 0..12, then for high=1..12, low=0..high-1, each (high,low) gives suited then offsuit. So (1,0) -> 13, 14; (2,0) -> 15, 16; (2,1) -> 17, 18; ...
    # Number of non-pairs before (high, low): sum for h=1 to high-1 of 2*h = (high-1)*high. Then for this high, low gives 2*low (suited) and 2*low+1 (offsuit). So index = 13 + (high-1)*high + 2*low + (0 if suited else 1)
    base = 13 + (high - 1) * high
    return base + 2 * low + (0 if suited else 1)
