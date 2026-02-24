"""
Build postflop bucket tables: sample (hand, board), MC equity, k-means -> 50 clusters.
Saves cluster centers for flop, turn, river.
"""

import random
import numpy as np
from poker_collusion.env.hand_eval import evaluate_hand
from poker_collusion.config import FLOP_BUCKETS, TURN_BUCKETS, RIVER_BUCKETS


def sample_hand_board(board_len, n_samples, rng=None):
    """
    Yield (hole_cards, board) as tuples. board_len in (3, 4, 5).
    """
    rng = rng or random.Random()
    deck = list(range(52))
    for _ in range(n_samples):
        rng.shuffle(deck)
        hole = (deck[0], deck[1])
        board = tuple(deck[2 : 2 + board_len])
        yield hole, board


def equity_flop(hole, board, n_rollouts=500):
    """Equity vs random opponent with random turn/river."""
    used = set(hole) | set(board)
    deck = [c for c in range(52) if c not in used]
    wins = 0
    for _ in range(n_rollouts):
        random.shuffle(deck)
        opp = (deck[0], deck[1])
        turn, river = deck[2], deck[3]
        full = list(board) + [turn, river]
        my_hand = evaluate_hand(list(hole) + full)
        opp_hand = evaluate_hand(list(opp) + full)
        if my_hand > opp_hand:
            wins += 1
        elif my_hand == opp_hand:
            wins += 0.5
    return wins / n_rollouts


def equity_turn(hole, board, n_rollouts=500):
    """Equity vs random opponent with random river."""
    used = set(hole) | set(board)
    deck = [c for c in range(52) if c not in used]
    wins = 0
    for _ in range(n_rollouts):
        random.shuffle(deck)
        opp = (deck[0], deck[1])
        river = deck[2]
        full = list(board) + [river]
        my_hand = evaluate_hand(list(hole) + full)
        opp_hand = evaluate_hand(list(opp) + full)
        if my_hand > opp_hand:
            wins += 1
        elif my_hand == opp_hand:
            wins += 0.5
    return wins / n_rollouts


def equity_river(hole, board, n_rollouts=500):
    """Hand strength on river: win prob vs 2 random opponent hands (2 opponents)."""
    used = set(hole) | set(board)
    deck = [c for c in range(52) if c not in used]
    wins = 0
    for _ in range(n_rollouts):
        random.shuffle(deck)
        opp1 = (deck[0], deck[1])
        opp2 = (deck[2], deck[3])
        full = list(board)
        my_hand = evaluate_hand(list(hole) + full)
        h1 = evaluate_hand(list(opp1) + full)
        h2 = evaluate_hand(list(opp2) + full)
        if my_hand >= h1 and my_hand >= h2:
            if my_hand > h1 and my_hand > h2:
                wins += 1
            else:
                wins += 0.5
        elif my_hand >= h1 or my_hand >= h2:
            wins += 0.33
    return wins / n_rollouts


def build_flop_table(n_samples=50000, n_rollouts=500, n_clusters=50, seed=42):
    """Sample (hand, flop), compute equity, k-means, return cluster centers."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return _build_flop_fallback(n_samples, n_rollouts, n_clusters, seed)
    rng = random.Random(seed)
    equities = []
    for hole, board in sample_hand_board(3, n_samples, rng):
        eq = equity_flop(hole, board, n_rollouts)
        equities.append(eq)
    X = np.array(equities).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(X)
    return kmeans.cluster_centers_.flatten().tolist()


def _build_flop_fallback(n_samples, n_rollouts, n_clusters, seed):
    """Equal-width bins when sklearn not available."""
    rng = random.Random(seed)
    equities = []
    for hole, board in sample_hand_board(3, n_samples, rng):
        eq = equity_flop(hole, board, n_rollouts)
        equities.append(eq)
    equities = np.array(equities)
    edges = np.percentile(equities, np.linspace(0, 100, n_clusters + 1)[1:-1])
    return edges.tolist()


def build_turn_table(n_samples=50000, n_rollouts=500, n_clusters=50, seed=42):
    """Sample (hand, turn), compute equity, k-means, return cluster centers."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return _build_turn_fallback(n_samples, n_rollouts, n_clusters, seed)
    rng = random.Random(seed)
    equities = []
    for hole, board in sample_hand_board(4, n_samples, rng):
        eq = equity_turn(hole, board, n_rollouts)
        equities.append(eq)
    X = np.array(equities).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(X)
    return kmeans.cluster_centers_.flatten().tolist()


def _build_turn_fallback(n_samples, n_rollouts, n_clusters, seed):
    rng = random.Random(seed)
    equities = []
    for hole, board in sample_hand_board(4, n_samples, rng):
        eq = equity_turn(hole, board, n_rollouts)
        equities.append(eq)
    equities = np.array(equities)
    edges = np.percentile(equities, np.linspace(0, 100, n_clusters + 1)[1:-1])
    return edges.tolist()


def build_river_table(n_samples=50000, n_rollouts=500, n_clusters=50, seed=42):
    """Sample (hand, river), compute strength, k-means, return cluster centers."""
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return _build_river_fallback(n_samples, n_rollouts, n_clusters, seed)
    rng = random.Random(seed)
    equities = []
    for hole, board in sample_hand_board(5, n_samples, rng):
        eq = equity_river(hole, board, n_rollouts)
        equities.append(eq)
    X = np.array(equities).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    kmeans.fit(X)
    return kmeans.cluster_centers_.flatten().tolist()


def _build_river_fallback(n_samples, n_rollouts, n_clusters, seed):
    rng = random.Random(seed)
    equities = []
    for hole, board in sample_hand_board(5, n_samples, rng):
        eq = equity_river(hole, board, n_rollouts)
        equities.append(eq)
    equities = np.array(equities)
    edges = np.percentile(equities, np.linspace(0, 100, n_clusters + 1)[1:-1])
    return edges.tolist()
