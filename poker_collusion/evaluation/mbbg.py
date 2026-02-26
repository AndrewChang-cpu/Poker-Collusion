"""
Self-play evaluation: mbb/g and block bootstrap standard error.
"""

import numpy as np
from tqdm import tqdm
from poker_collusion.config import NUM_PLAYERS, EVAL_BLOCK_SIZE


def play_hand(game, trainer, num_players=NUM_PLAYERS):
    """
    Play one hand; all players use the trainer's average strategy.
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
        info_key = game.get_info_key(state, player)
        avg_strategy = trainer.get_average_strategy(info_key, actions)
        if avg_strategy is None or len(avg_strategy) != len(actions):
            avg_strategy = np.ones(len(actions)) / len(actions)
        action_idx = np.random.choice(len(actions), p=avg_strategy)
        game.apply_action(state, actions[action_idx])
    return game.get_payoffs(state)


def evaluate(game, trainer, num_hands=10000, num_players=NUM_PLAYERS):
    """
    Run evaluation; return (avg_payoffs, mbb_per_game) per player.
    mbb/g = (avg profit per hand in BB) * 1000.
    """
    total_payoffs = np.zeros(num_players)
    for _ in range(num_hands):
        total_payoffs += play_hand(game, trainer, num_players)
    avg_payoffs = total_payoffs / num_hands
    mbb_per_game = avg_payoffs * 1000
    return avg_payoffs, mbb_per_game


def evaluate_with_variance(
    game,
    trainer,
    num_hands=10000,
    num_players=NUM_PLAYERS,
    block_size=EVAL_BLOCK_SIZE,
):
    """
    Evaluate with block bootstrap standard error and 95% CI.
    Returns (mbb_mean, mbb_se) arrays.
    """
    block_payoffs = []
    current_block = np.zeros(num_players)
    hands_in_block = 0

    for _ in tqdm(range(num_hands),"Evaluating..."):
        current_block += np.array(play_hand(game, trainer, num_players))
        hands_in_block += 1
        if hands_in_block >= block_size:
            block_payoffs.append(current_block / hands_in_block)
            current_block = np.zeros(num_players)
            hands_in_block = 0

    if hands_in_block > 0:
        block_payoffs.append(current_block / hands_in_block)

    block_payoffs = np.array(block_payoffs)
    mean = block_payoffs.mean(axis=0)
    std_err = block_payoffs.std(axis=0) / np.sqrt(len(block_payoffs))
    mbb_mean = mean * 1000
    mbb_se = std_err * 1000

    print(f"\nEvaluation over {num_hands} hands ({len(block_payoffs)} blocks):")
    print(f"{'Player':<10} {'mbb/g':<12} {'± SE':<12} {'95% CI':<20}")
    print("-" * 55)
    for p in range(num_players):
        ci_low = mbb_mean[p] - 1.96 * mbb_se[p]
        ci_high = mbb_mean[p] + 1.96 * mbb_se[p]
        print(f"Player {p:<4} {mbb_mean[p]:<12.1f} {mbb_se[p]:<12.1f} [{ci_low:.1f}, {ci_high:.1f}]")

    return mbb_mean, mbb_se
