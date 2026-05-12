"""
Continuation strategies for subgame leaf-node evaluation.

At each non-terminal leaf of the depth-limited subgame each player
independently selects one of K=4 continuation strategies.  The leaf payoff
is estimated by rolling out the remainder of the game under the chosen
strategies and averaging over multiple Monte Carlo rollouts.

The four strategies (per player):
  0  Blueprint    — the trained average strategy
  1  Fold-biased  — blueprint with fold probability * BIAS_FACTOR, renorm
  2  Call-biased  — blueprint with call probability * BIAS_FACTOR, renorm
  3  Raise-biased — blueprint with all raise/bet probs * BIAS_FACTOR, renorm
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from poker_collusion.config import BIAS_FACTOR, NUM_ACTIONS

NUM_CONTINUATION_STRATEGIES = 4


def _bias_probs(
    probs: np.ndarray,
    legal_actions: List[int],
    bias_factor: float,
    target_indices: set[int],
) -> np.ndarray:
    """Multiply probs at *target_indices* by bias_factor, then renormalize."""
    biased = probs.copy()
    for i, a in enumerate(legal_actions):
        if a in target_indices:
            biased[i] *= bias_factor
    total = biased.sum()
    if total > 0:
        biased /= total
    else:
        biased = np.ones(len(legal_actions)) / len(legal_actions)
    return biased


def get_continuation_strategy(
    strategy_id: int,
    blueprint_probs: np.ndarray,
    legal_actions: List[int],
    bias_factor: float = BIAS_FACTOR,
) -> np.ndarray:
    """
    Return a probability distribution over *legal_actions* for the given
    continuation strategy id.

    Parameters
    ----------
    strategy_id : int in [0, 3]
    blueprint_probs : array of shape (len(legal_actions),)
        The trained average strategy probabilities.
    legal_actions : list of action indices (0..9)
    bias_factor : multiplier for the biased action category.

    Returns
    -------
    np.ndarray of shape (len(legal_actions),), non-negative, sums to 1.
    """
    if strategy_id == 0:
        return blueprint_probs.copy()

    if strategy_id == 1:
        return _bias_probs(blueprint_probs, legal_actions, bias_factor, {0})

    if strategy_id == 2:
        return _bias_probs(blueprint_probs, legal_actions, bias_factor, {1})

    if strategy_id == 3:
        raise_actions = {a for a in legal_actions if a >= 2}
        return _bias_probs(blueprint_probs, legal_actions, bias_factor, raise_actions)

    raise ValueError(f"Invalid continuation strategy id: {strategy_id}")


def rollout_leaf_value(
    game: Any,
    state: Any,
    blueprint: Any,
    num_players: int,
    continuation_ids: List[int],
    num_rollouts: int,
    bias_factor: float = BIAS_FACTOR,
    bucket_cache: Any = None,
) -> np.ndarray:
    """
    Estimate the expected payoff vector at a non-terminal depth-limit leaf
    by running *num_rollouts* rollouts to the end of the game.

    Each rollout plays out the game from *state* to termination.  At each
    decision point, each player uses their assigned continuation strategy
    (given by continuation_ids[player]).

    Parameters
    ----------
    game : CFRGame-like module
    state : current game state at the leaf
    blueprint : object with get_average_strategy(info_key, legal_actions)
    num_players : int
    continuation_ids : list of int, length num_players
    num_rollouts : int
    bias_factor : float
    bucket_cache : optional dict for memoizing bucket lookups

    Returns
    -------
    np.ndarray of shape (num_players,) — mean payoff per player across rollouts.
    """
    payoff_sum = np.zeros(num_players, dtype=np.float64)

    for _ in range(num_rollouts):
        s = state.copy()
        while not game.is_terminal(s):
            if game.is_chance_node(s):
                s = game.sample_chance(s)
                continue
            player = game.get_current_player(s)
            actions = game.get_legal_actions(s)
            if not actions:
                break

            info_key = _cached_info_key(s, player, bucket_cache)
            bp_probs = blueprint.get_average_strategy(info_key, actions)
            if bp_probs is None or len(bp_probs) != len(actions):
                bp_probs = np.ones(len(actions)) / len(actions)

            strat_id = continuation_ids[player]
            probs = get_continuation_strategy(strat_id, bp_probs, actions, bias_factor)
            action_idx = np.random.choice(len(actions), p=probs)
            s = game.apply_action(s, actions[action_idx])

        payoff_sum += np.array(game.get_payoffs(s))

    return payoff_sum / num_rollouts


def _cached_info_key(state: Any, player: int, bucket_cache: Any = None) -> Any:
    """Compute info key with optional bucket caching for postflop streets."""
    from poker_collusion.abstraction.bucketing import get_bucket, hole_to_canonical

    hole = tuple(sorted(state.hole_cards[player]))
    round_idx = state.round_idx

    if round_idx == 0:
        bucket = int(hole_to_canonical(hole))
    else:
        cache_key = (hole, round_idx)
        if bucket_cache is not None and cache_key in bucket_cache:
            bucket = bucket_cache[cache_key]
        else:
            board = tuple(state.board)
            bucket = int(get_bucket(hole, board, round_idx))
            if bucket_cache is not None:
                bucket_cache[cache_key] = bucket

    history = tuple(state.action_history)
    return (bucket, history)
