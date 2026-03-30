"""
5-card hand evaluator for Texas Hold'em.
Cards are integers 0-51: suit = card // 13, rank = card % 13 (0=2 .. 12=A).
Returns a comparable tuple; higher = better hand.
"""

from itertools import combinations
from collections import Counter

RANK_NAMES = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
HAND_TYPE_NAMES = [
    "High Card", "Pair", "Two Pair", "Three of a Kind", 
    "Straight", "Flush", "Full House", "Four of a Kind", "Straight Flush"
]

def card_rank(card):
    return card % 13

def card_suit(card):
    return card // 13

def evaluate_hand(cards):
    """Evaluate best 5-card hand from 5-7 cards. Returns comparable tuple."""
    best = None
    card_list = list(cards)
    for combo in combinations(card_list, 5):
        score = _score_5(combo)
        if best is None or score > best:
            best = score
    return best

def get_hand_description(score):
    """Translates a score tuple into a human-readable string."""
    if not score: return "Unknown"
    type_idx = score[0]
    name = HAND_TYPE_NAMES[type_idx]
    
    if type_idx == 8: # Straight Flush
        return f"Straight Flush, {RANK_NAMES[score[1]]} high"
    if type_idx == 7: # Quads
        return f"Four of a Kind, {RANK_NAMES[score[1]]}s"
    if type_idx == 6: # Full House
        return f"Full House, {RANK_NAMES[score[1]]}s over {RANK_NAMES[score[2]]}s"
    if type_idx == 5: # Flush
        return f"Flush, {RANK_NAMES[score[1]]} high"
    if type_idx == 4: # Straight
        return f"Straight, {RANK_NAMES[score[1]]} high"
    if type_idx == 3: # Trips
        return f"Three of a Kind, {RANK_NAMES[score[1]]}s"
    if type_idx == 2: # Two Pair
        return f"Two Pair, {RANK_NAMES[score[1]]}s and {RANK_NAMES[score[2]]}s"
    if type_idx == 1: # Pair
        return f"Pair of {RANK_NAMES[score[1]]}s"
    return f"High Card, {RANK_NAMES[score[1]]}"

def _score_5(cards):
    """Score a 5-card hand. Returns comparable tuple."""
    ranks = sorted([card_rank(c) for c in cards], reverse=True)
    suits = [card_suit(c) for c in cards]
    is_flush = len(set(suits)) == 1

    unique_ranks = sorted(set(ranks), reverse=True)
    is_straight = False
    straight_high = 0
    if len(unique_ranks) == 5:
        if unique_ranks[0] - unique_ranks[4] == 4:
            is_straight = True
            straight_high = unique_ranks[0]
        if unique_ranks == [12, 3, 2, 1, 0]:  # A-2-3-4-5
            is_straight = True
            straight_high = 3

    counts = Counter(ranks)
    freq = sorted(counts.values(), reverse=True)
    ranks_by_freq = sorted(counts.keys(), key=lambda r: (counts[r], r), reverse=True)

    if is_straight and is_flush:
        return (8, straight_high)
    if freq == [4, 1]:
        return (7, *ranks_by_freq)
    if freq == [3, 2]:
        return (6, *ranks_by_freq)
    if is_flush:
        return (5, *ranks)
    if is_straight:
        return (4, straight_high)
    if freq == [3, 1, 1]:
        return (3, *ranks_by_freq)
    if freq == [2, 2, 1]:
        return (2, *ranks_by_freq)
    if freq == [2, 1, 1, 1]:
        return (1, *ranks_by_freq)
    return (0, *ranks)