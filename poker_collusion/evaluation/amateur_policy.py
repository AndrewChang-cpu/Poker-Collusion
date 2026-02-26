"""
Competent-amateur policy: hand strength (preflop 2-card, postflop Monte Carlo) + pot odds.
Outputs a probability distribution over legal actions for evaluation vs CFR.
"""

import numpy as np
from poker_collusion.env.hand_eval import evaluate_hand, card_rank, card_suit

# Default number of random opponent hands for postflop strength
DEFAULT_POSTFLOP_SAMPLES = 100


def _to_call(state, player):
    return max(state.bets) - state.bets[player]


def _pot_after_call(state, player):
    return state.pot + _to_call(state, player)


def _preflop_strength(hole_cards):
    """
    Scalar strength in [0, 1] from two hole cards.
    Uses high rank, pair, suited, connected.
    """
    r0, r1 = sorted([card_rank(c) for c in hole_cards], reverse=True)
    suited = card_suit(hole_cards[0]) == card_suit(hole_cards[1])
    connected = (abs(r0 - r1) <= 1) or (r0 == 12 and r1 == 0)  # A2

    if r0 == r1:
        # Pair: 0.5 + rank component
        base = 0.5 + (r0 / 13) * 0.4
    else:
        # High card: (r0*13 + r1) / 169 normalized to ~[0, 0.5]
        base = (r0 * 13 + r1) / 169 * 0.45
    if suited:
        base += 0.08
    if connected:
        base += 0.05
    return float(np.clip(base, 0.0, 1.0))


def _postflop_strength(hole_cards, board, n_samples=DEFAULT_POSTFLOP_SAMPLES):
    """
    Monte Carlo hand strength: win rate vs n_samples random opponent hands.
    Returns float in [0, 1]. Ties count as 0.5.
    """
    my_cards = list(hole_cards) + list(board)
    my_best = evaluate_hand(my_cards)
    used = set(hole_cards) | set(board)
    deck = [c for c in range(52) if c not in used]
    n = len(deck)
    if n < 2:
        return 0.5
    wins = 0.0
    deck = np.array(deck)
    for _ in range(n_samples):
        idx = np.random.choice(n, size=2, replace=False)
        opp = [int(deck[idx[0]]), int(deck[idx[1]])]
        opp_best = evaluate_hand(list(opp) + list(board))
        if my_best > opp_best:
            wins += 1.0
        elif my_best == opp_best:
            wins += 0.5
    return wins / n_samples


def _fold_call_raise_weights(strength, to_call, pot_after_call, facing_bet):
    """
    Base weights (fold_w, call_w, raise_w) from strength and pot odds.
    facing_bet: True if to_call > 0.
    """
    if not facing_bet:
        # Can check: weak -> check, strong -> raise
        fold_w = 0.0
        call_w = 1.0 - strength * 0.7
        raise_w = 0.1 + strength * 0.7
    else:
        pot_odds = to_call / pot_after_call if pot_after_call > 0 else 1.0
        # Bad pot odds + weak -> fold more
        if strength < 0.33:
            fold_w = 2.0 + (1.0 - pot_odds) * 1.5
            call_w = 0.5 + pot_odds
            raise_w = 0.2
        elif strength < 0.66:
            fold_w = 0.8 + (1.0 - pot_odds) * 0.5
            call_w = 1.2 + pot_odds * 0.5
            raise_w = 0.5
        else:
            fold_w = 0.1
            call_w = 1.0 + pot_odds
            raise_w = 1.5
    return fold_w, call_w, raise_w


def get_action_probs(state, player, legal_actions, n_postflop_samples=DEFAULT_POSTFLOP_SAMPLES):
    """
    Return a probability distribution over legal_actions (same length as legal_actions).
    state: NLHEState; player: int; legal_actions: list of action indices in [0..9].
    """
    hole = state.hole_cards[player]
    to_call = _to_call(state, player)
    pot_after = _pot_after_call(state, player)
    facing_bet = to_call > 0

    if state.round_idx == 0 or len(state.board) < 3:
        strength = _preflop_strength(hole)
    else:
        strength = _postflop_strength(hole, state.board, n_postflop_samples)

    fold_w, call_w, raise_w = _fold_call_raise_weights(strength, to_call, pot_after, facing_bet)

    # Map each legal action to one of fold / call / raise
    raise_indices = [a for a in legal_actions if a >= 2]
    n_raise = len(raise_indices) if raise_indices else 1
    weights = []
    for a in legal_actions:
        if a == 0:
            weights.append(fold_w)
        elif a == 1:
            weights.append(call_w)
        else:
            weights.append(raise_w / n_raise)

    probs = np.array(weights, dtype=float)
    total = probs.sum()
    if total <= 0:
        probs = np.ones(len(legal_actions)) / len(legal_actions)
    else:
        probs /= total
    return probs


class AmateurPolicy:
    """Wrapper so evaluation can detect policy type and call get_action_probs."""

    def __init__(self, n_postflop_samples=DEFAULT_POSTFLOP_SAMPLES):
        self.n_postflop_samples = n_postflop_samples

    def get_action_probs(self, state, player, legal_actions):
        return get_action_probs(
            state, player, legal_actions, n_postflop_samples=self.n_postflop_samples
        )
