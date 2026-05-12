"""
Multi-bot play harness for running matches between PluribusBot instances
and/or AmateurPolicy opponents.

Handles the hand lifecycle: deal, decision loop (query bots, notify all),
and payoff collection.  Reports mbb/g with block-bootstrap standard error.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from poker_collusion.config import EVAL_BLOCK_SIZE, NUM_PLAYERS
from poker_collusion.evaluation.amateur_policy import AmateurPolicy
from poker_collusion.search.bot import PluribusBot

BotOrPolicy = Union[PluribusBot, AmateurPolicy]


def play_hand(
    game: Any,
    bots: List[BotOrPolicy],
    num_players: int = NUM_PLAYERS,
) -> List[float]:
    """
    Play one hand with the given bots/policies.

    For PluribusBot instances: uses reset_hand / select_action / observe_action.
    For AmateurPolicy instances: uses get_action_probs + sampling.

    Returns list of payoffs (BB) per player.
    """
    state = game.deal_new_hand()

    for bot in bots:
        if isinstance(bot, PluribusBot):
            bot.reset_hand(state)

    while not game.is_terminal(state):
        if game.is_chance_node(state):
            state = game.sample_chance(state)
            for bot in bots:
                if isinstance(bot, PluribusBot):
                    bot.observe_new_street(state)
            continue

        player = game.get_current_player(state)
        actions = game.get_legal_actions(state)
        if not actions:
            break

        bot = bots[player]
        if isinstance(bot, PluribusBot):
            action = bot.select_action(state)
        else:
            probs = bot.get_action_probs(state, player, actions)
            action_idx = np.random.choice(len(actions), p=probs)
            action = actions[action_idx]

        for b in bots:
            if isinstance(b, PluribusBot):
                b.observe_action(state, action, player)

        state = game.apply_action(state, action)

    return game.get_payoffs(state)


def run_match(
    game: Any,
    bots: List[BotOrPolicy],
    num_hands: int,
    num_players: int = NUM_PLAYERS,
    block_size: int = EVAL_BLOCK_SIZE,
    desc: str = "Playing...",
    silent: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run *num_hands* hands and report mbb/g per seat with standard error.

    Returns
    -------
    (mbb_mean, mbb_se) : tuple of np.ndarray, each of shape (num_players,)
    """
    block_payoffs = []
    current_block = np.zeros(num_players, dtype=np.float64)
    hands_in_block = 0

    iterator = tqdm(range(num_hands), desc=desc, disable=silent)
    for _ in iterator:
        payoffs = play_hand(game, bots, num_players)
        current_block += np.array(payoffs)
        hands_in_block += 1
        if hands_in_block >= block_size:
            block_payoffs.append(current_block / hands_in_block)
            current_block = np.zeros(num_players, dtype=np.float64)
            hands_in_block = 0

    if hands_in_block > 0:
        block_payoffs.append(current_block / hands_in_block)

    blocks = np.array(block_payoffs) if block_payoffs else np.zeros((1, num_players))
    mean = blocks.mean(axis=0) * 1000  # convert to mbb/g
    if len(blocks) > 1:
        se = blocks.std(axis=0, ddof=1) / np.sqrt(len(blocks)) * 1000
    else:
        se = np.zeros(num_players)

    return mean, se


def print_results(
    mbb_mean: np.ndarray,
    mbb_se: np.ndarray,
    bot_names: List[str],
    num_hands: int,
    num_players: int = NUM_PLAYERS,
) -> None:
    """Pretty-print match results."""
    seat_names = ["BTN", "SB", "BB"]
    print(f"\nMatch results ({num_hands} hands):")
    print(f"{'Seat':<6} {'Bot':<28} {'mbb/g':<12} {'± SE':<12} {'95% CI':<20}")
    print("-" * 78)
    for p in range(num_players):
        ci_low = mbb_mean[p] - 1.96 * mbb_se[p]
        ci_high = mbb_mean[p] + 1.96 * mbb_se[p]
        print(
            f"{seat_names[p]:<6} {bot_names[p]:<28} "
            f"{mbb_mean[p]:<12.1f} {mbb_se[p]:<12.1f} "
            f"[{ci_low:.1f}, {ci_high:.1f}]"
        )
