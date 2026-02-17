"""
Evaluation: play trained agents against each other and measure mbb/g.
"""

import numpy as np


def play_hand(game, trainer, num_players=3):
    """
    Play a single hand where all players use the trained average strategy.
    Returns payoffs for each player.
    """
    state = game.deal_new_hand()

    while not game.is_terminal(state):
        if game.is_chance_node(state):
            state = game.sample_chance(state)
            continue

        player = game.get_current_player(state)
        actions = game.get_legal_actions(state)

        if len(actions) == 0:
            break

        info_key = game.get_info_key(state, player)
        avg_strategy = trainer.get_average_strategy(info_key)

        if avg_strategy is None or len(avg_strategy) != len(actions):
            # Unseen info set: play uniformly
            avg_strategy = np.ones(len(actions)) / len(actions)

        # Sample action from average strategy
        action_idx = np.random.choice(len(actions), p=avg_strategy)
        state = game.apply_action(state, actions[action_idx])

    return game.get_payoffs(state)


def evaluate(game, trainer, num_hands=10000, num_players=3):
    """
    Run evaluation over many hands. Report mbb/g for each player.

    mbb/g = (avg profit per hand) * 1000, measured in big blinds.
    """
    total_payoffs = np.zeros(num_players)
    hand_count = 0

    for i in range(num_hands):
        payoffs = play_hand(game, trainer, num_players)
        total_payoffs += np.array(payoffs)
        hand_count += 1

    avg_payoffs = total_payoffs / hand_count
    mbb_per_game = avg_payoffs * 1000  # convert BB to mBB

    print(f"\nEvaluation over {num_hands} hands:")
    print(f"{'Player':<10} {'Avg BB/hand':<15} {'mbb/g':<15}")
    print("-" * 40)
    for p in range(num_players):
        print(f"Player {p:<4} {avg_payoffs[p]:<15.4f} {mbb_per_game[p]:<15.1f}")

    # Sanity check: payoffs should sum to ~0 (zero-sum)
    print(f"\nSum of payoffs: {avg_payoffs.sum():.4f} (should be ~0)")

    return avg_payoffs, mbb_per_game


def evaluate_with_variance(game, trainer, num_hands=10000, num_players=3, block_size=500):
    """
    Evaluate with standard error estimation using block bootstrapping.
    Reports mbb/g with confidence intervals.
    """
    block_payoffs = []
    current_block = np.zeros(num_players)
    hands_in_block = 0

    for i in range(num_hands):
        payoffs = play_hand(game, trainer, num_players)
        current_block += np.array(payoffs)
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
