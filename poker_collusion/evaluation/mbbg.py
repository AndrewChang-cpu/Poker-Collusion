"""
Self-play evaluation: mbb/g and block bootstrap standard error.
"""

from __future__ import annotations

from typing import Any, List, Optional, Protocol, Tuple, Union

import numpy as np
from tqdm import tqdm

from poker_collusion.config import NUM_PLAYERS, EVAL_BLOCK_SIZE
from poker_collusion.evaluation.amateur_policy import AmateurPolicy
from poker_collusion.typing_defs import CFRGame


class _SupportsAverageStrategy(Protocol):
    def get_average_strategy(
        self, info_key: Any, legal_actions: List[int]
    ) -> Any: ...


class _SupportsActionProbs(Protocol):
    def get_action_probs(
        self, state: Any, player: int, legal_actions: List[int]
    ) -> Any: ...


Policy = Union[_SupportsAverageStrategy, _SupportsActionProbs, AmateurPolicy]


def _get_policy_probs(
    game: CFRGame, state: Any, player: int, actions: List[int], policy: Policy
) -> np.ndarray:
    """Return probability distribution over actions from trainer or amateur policy."""
    if hasattr(policy, "get_average_strategy"):
        info_key = game.get_info_key(state, player)
        probs = policy.get_average_strategy(info_key, actions)
        if probs is None or len(probs) != len(actions):
            probs = np.ones(len(actions)) / len(actions)
    else:
        probs = policy.get_action_probs(state, player, actions)
    return probs


def play_hand_with_policies(
    game: CFRGame, policies: List[Policy], num_players: int = NUM_PLAYERS
) -> List[float]:
    """
    Play one hand with per-player policies. policies[i] is either a CFRTrainer
    (uses get_average_strategy) or an AmateurPolicy (uses get_action_probs).
    Returns list of payoffs (BB) per player.
    """
    state = game.deal_new_hand()
    while not game.is_terminal(state):
        if game.is_chance_node(state):
            state = game.sample_chance(state)
            continue
        player = game.get_current_player(state)
        actions = game.get_legal_actions(state)
        if not actions:
            break
        policy = policies[player]
        probs = _get_policy_probs(game, state, player, actions, policy)
        action_idx = np.random.choice(len(actions), p=probs)
        state = game.apply_action(state, actions[action_idx])
    return game.get_payoffs(state)


def play_hand(
    game: CFRGame, trainer: _SupportsAverageStrategy, num_players: int = NUM_PLAYERS
) -> List[float]:
    """
    Play one hand; all players use the trainer's average strategy.
    Returns list of payoffs (BB) per player.
    """
    return play_hand_with_policies(game, [trainer] * num_players, num_players)


def _collect_block_payoffs(
    game: CFRGame,
    policies: List[Policy],
    num_hands: int,
    num_players: int,
    block_size: int,
    desc: str,
) -> np.ndarray:
    """
    Play num_hands hands and return block-averaged payoffs array of shape (n_blocks, num_players).
    """
    block_payoffs = []
    current_block = np.zeros(num_players)
    hands_in_block = 0

    for _ in tqdm(range(num_hands), desc=desc):
        current_block += np.array(play_hand_with_policies(game, policies, num_players))
        hands_in_block += 1
        if hands_in_block >= block_size:
            block_payoffs.append(current_block / hands_in_block)
            current_block = np.zeros(num_players)
            hands_in_block = 0

    if hands_in_block > 0:
        block_payoffs.append(current_block / hands_in_block)

    return np.array(block_payoffs)


def _summarize_blocks(
    block_payoffs: np.ndarray,
    num_hands: int,
    num_players: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mbb_mean and mbb_se from block-averaged payoffs."""
    mean = block_payoffs.mean(axis=0)
    std_err = block_payoffs.std(axis=0, ddof=1) / np.sqrt(len(block_payoffs))
    return mean * 1000, std_err * 1000


def evaluate_strategies(
    game: CFRGame,
    policies: List[Policy],
    names: Optional[List[str]] = None,
    num_hands: int = 10000,
    num_players: int = NUM_PLAYERS,
    block_size: int = EVAL_BLOCK_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate per-player policies with block bootstrap standard error and 95% CI.
    policies[i] is the strategy for player i. names[i] is an optional display label.
    Returns (mbb_mean, mbb_se) arrays.
    """
    if names is None:
        names = [f"P{p}" for p in range(num_players)]

    block_payoffs = _collect_block_payoffs(
        game, policies, num_hands, num_players, block_size, desc="Evaluating..."
    )
    mbb_mean, mbb_se = _summarize_blocks(block_payoffs, num_hands, num_players)

    print(f"\nEvaluation over {num_hands} hands ({len(block_payoffs)} blocks):")
    print(f"{'Seat':<6} {'Strategy':<28} {'mbb/g':<12} {'± SE':<12} {'95% CI':<20}")
    print("-" * 78)
    seat_names = ["BTN", "SB", "BB"]
    for p in range(num_players):
        ci_low = mbb_mean[p] - 1.96 * mbb_se[p]
        ci_high = mbb_mean[p] + 1.96 * mbb_se[p]
        print(
            f"{seat_names[p]:<6} {names[p]:<28} {mbb_mean[p]:<12.1f} {mbb_se[p]:<12.1f} [{ci_low:.1f}, {ci_high:.1f}]"
        )

    return mbb_mean, mbb_se


def evaluate_with_variance(
    game: CFRGame,
    trainer: _SupportsAverageStrategy,
    num_hands: int = 10000,
    num_players: int = NUM_PLAYERS,
    block_size: int = EVAL_BLOCK_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Self-play evaluation: all players use trainer. Convenience wrapper over evaluate_strategies.
    Returns (mbb_mean, mbb_se) arrays.
    """
    return evaluate_strategies(
        game,
        policies=[trainer] * num_players,
        num_hands=num_hands,
        num_players=num_players,
        block_size=block_size,
    )


def evaluate_vs_amateur(
    game: CFRGame,
    trainer: _SupportsAverageStrategy,
    num_hands: int = 10000,
    num_players: int = NUM_PLAYERS,
    cfr_seat: int = 0,
    block_size: int = EVAL_BLOCK_SIZE,
    amateur: Optional[AmateurPolicy] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate CFR (trainer) vs amateur policy. CFR plays in cfr_seat; others play amateur.
    With rotation, run this for cfr_seat=0,1,2 and average CFR mbb/g.
    Returns (mbb_mean, mbb_se) arrays; prints per-player and, if rotating, CFR average.
    """
    if amateur is None:
        amateur = AmateurPolicy()
    policies = [amateur] * num_players
    policies[cfr_seat] = trainer

    block_payoffs = _collect_block_payoffs(
        game, policies, num_hands, num_players, block_size, desc="Evaluating vs amateur..."
    )
    mbb_mean, mbb_se = _summarize_blocks(block_payoffs, num_hands, num_players)

    seat_names = ["BTN", "SB", "BB"]
    print(f"\nCFR vs Amateur — CFR in seat {cfr_seat} ({seat_names[cfr_seat]}), {num_hands} hands:")
    print(f"{'Seat':<8} {'Role':<8} {'mbb/g':<12} {'± SE':<12} {'95% CI':<20}")
    print("-" * 62)
    for p in range(num_players):
        role = "CFR" if p == cfr_seat else "Amateur"
        ci_low = mbb_mean[p] - 1.96 * mbb_se[p]
        ci_high = mbb_mean[p] + 1.96 * mbb_se[p]
        print(f"{seat_names[p]:<8} {role:<8} {mbb_mean[p]:<12.1f} {mbb_se[p]:<12.1f} [{ci_low:.1f}, {ci_high:.1f}]")
    print(f"\nCFR (seat {cfr_seat}): mbb/g = {mbb_mean[cfr_seat]:.1f} ± {mbb_se[cfr_seat]:.1f}")
    return mbb_mean, mbb_se


def evaluate_rotate(
    game: CFRGame,
    primary: Policy,
    opponent: Policy,
    primary_name: str = "Primary",
    opponent_name: str = "Opponent",
    num_hands_per_seat: int = 10000,
    num_players: int = NUM_PLAYERS,
    block_size: int = EVAL_BLOCK_SIZE,
) -> Tuple[List[float], List[float]]:
    """
    Rotate primary through each seat against opponent filling the remaining seats.
    Calls evaluate_strategies for each rotation and prints a summary.
    Returns (primary_mbb, primary_se) lists indexed by seat.
    """
    seat_names = ["BTN", "SB", "BB"]
    primary_mbb = []
    primary_se = []

    for primary_seat in range(num_players):
        policies = [opponent] * num_players
        policies[primary_seat] = primary
        names = [opponent_name] * num_players
        names[primary_seat] = primary_name

        mbb_mean, mbb_se = evaluate_strategies(
            game, policies, names,
            num_hands=num_hands_per_seat,
            num_players=num_players,
            block_size=block_size,
        )
        primary_mbb.append(float(mbb_mean[primary_seat]))
        primary_se.append(float(mbb_se[primary_seat]))

    print("\n" + "=" * 60)
    print(f"Rotation summary: {primary_name} vs {opponent_name}")
    print("=" * 60)
    for i in range(num_players):
        print(f"  {primary_name} as {seat_names[i]:<4}: mbb/g = {primary_mbb[i]:.1f} ± {primary_se[i]:.1f}")
    avg_mbb = sum(primary_mbb) / num_players
    avg_se = (sum(s ** 2 for s in primary_se) ** 0.5) / num_players  # approximate
    print(f"  {primary_name} average:    mbb/g = {avg_mbb:.1f} ± {avg_se:.1f}")
    return primary_mbb, primary_se


def evaluate_vs_amateur_rotate(
    game: CFRGame,
    trainer: _SupportsAverageStrategy,
    num_hands_per_seat: int = 10000,
    num_players: int = NUM_PLAYERS,
    block_size: int = EVAL_BLOCK_SIZE,
    amateur: Optional[AmateurPolicy] = None,
) -> Tuple[List[float], List[float]]:
    """
    Rotate CFR through BTN/SB/BB against amateur. Wrapper around evaluate_rotate.
    """
    if amateur is None:
        amateur = AmateurPolicy()
    return evaluate_rotate(
        game, trainer, amateur,
        primary_name="CFR", opponent_name="Amateur",
        num_hands_per_seat=num_hands_per_seat,
        num_players=num_players,
        block_size=block_size,
    )
