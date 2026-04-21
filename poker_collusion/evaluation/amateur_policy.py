"""
Competent-amateur policy for Leduc Hold'em: hand strength (normalized rank / Monte Carlo) + pot odds.
Outputs a probability distribution over legal actions for evaluation vs CFR.
Modified for 3-player Leduc (1 hole card, 12-card deck).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from poker_collusion.env.game_state import NLHEState
from poker_collusion.env.hand_eval import evaluate_hand

# Default number of random opponent hands for postflop strength
DEFAULT_POSTFLOP_SAMPLES = 100


def _to_call(state: NLHEState, player: int) -> float:
    return max(state.bets) - state.bets[player]


def _pot_after_call(state: NLHEState, player: int) -> float:
    return state.pot + _to_call(state, player)


def _preflop_strength(hole_cards: Sequence[int]) -> float:
    """
    Simplified Leduc preflop strength: direct rank normalization.
    Ranks 0 (J), 1 (Q), 2 (K), 3 (A) map to [0, 1] range.
    """
    if not hole_cards:
        return 0.0
    # Normalize rank index 0-3 to 0.0-1.0
    return float(hole_cards[0] / 3.0)


def _postflop_strength(
    hole_cards: Sequence[int],
    board: Sequence[int],
    n_samples: int = DEFAULT_POSTFLOP_SAMPLES,
) -> float:
    """
    Monte Carlo hand strength for Leduc: win rate vs 1-card opponent from 12-card deck.
    Ties count as 0.5.
    """
    my_hand = evaluate_hand(list(hole_cards) + list(board))
    used = set(hole_cards) | set(board)
    
    # Leduc deck: 4 ranks * 3 suits = 12 cards
    deck = [0, 1, 2, 3] * 3
    for c in used:
        deck.remove(c)
        
    n = len(deck)
    if n < 1:
        return 0.5
        
    wins = 0.0
    for _ in range(n_samples):
        # In Leduc, opponent has 1 hole card
        opp_card = np.random.choice(deck)
        opp_hand = evaluate_hand([int(opp_card)] + list(board))
        
        if my_hand > opp_hand:
            wins += 1.0
        elif my_hand == opp_hand:
            wins += 0.5
    return wins / n_samples


def _fold_call_raise_weights(
    strength: float, to_call: float, pot_after_call: float, facing_bet: bool
) -> Tuple[float, float, float]:
    """
    Base weights (fold_w, call_w, raise_w) from strength and pot odds.
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


def get_action_probs(
    state: NLHEState,
    player: int,
    legal_actions: List[int],
    n_postflop_samples: int = DEFAULT_POSTFLOP_SAMPLES,
) -> np.ndarray:
    """
    Return a probability distribution over legal_actions.
    Modified for Leduc street indices.
    """
    hole = state.hole_cards[player]
    to_call = _to_call(state, player)
    pot_after = _pot_after_call(state, player)
    facing_bet = to_call > 0

    # Leduc street check: round_idx 0 is preflop, 1 is flop
    if state.round_idx == 0:
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

    def __init__(self, n_postflop_samples: int = DEFAULT_POSTFLOP_SAMPLES) -> None:
        self.n_postflop_samples = n_postflop_samples

    def get_action_probs(
        self, state: NLHEState, player: int, legal_actions: List[int]
    ) -> np.ndarray:
        return get_action_probs(
            state, player, legal_actions, n_postflop_samples=self.n_postflop_samples
        )