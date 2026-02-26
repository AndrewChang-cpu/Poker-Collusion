"""
Self-play evaluation: mbb/g and block bootstrap standard error.
"""

import numpy as np
from tqdm import tqdm
from poker_collusion.config import NUM_PLAYERS, EVAL_BLOCK_SIZE
from poker_collusion.evaluation.amateur_policy import AmateurPolicy


def _get_policy_probs(game, state, player, actions, policy):
    """Return probability distribution over actions from trainer or amateur policy."""
    if hasattr(policy, "get_average_strategy"):
        info_key = game.get_info_key(state, player)
        probs = policy.get_average_strategy(info_key, actions)
        if probs is None or len(probs) != len(actions):
            probs = np.ones(len(actions)) / len(actions)
    else:
        probs = policy.get_action_probs(state, player, actions)
    return probs


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


def play_hand_with_policies(game, policies, num_players=NUM_PLAYERS):
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


def evaluate_vs_amateur(
    game,
    trainer,
    num_hands=10000,
    num_players=NUM_PLAYERS,
    cfr_seat=0,
    block_size=EVAL_BLOCK_SIZE,
    amateur=None,
):
    """
    Evaluate CFR (trainer) vs amateur policy. CFR plays in cfr_seat; others play amateur.
    With rotation, run this for cfr_seat=0,1,2 and average CFR mbb/g.
    Returns (mbb_mean, mbb_se) arrays; prints per-player and, if rotating, CFR average.
    """
    if amateur is None:
        amateur = AmateurPolicy()
    policies = [amateur] * num_players
    policies[cfr_seat] = trainer

    block_payoffs = []
    current_block = np.zeros(num_players)
    hands_in_block = 0

    for _ in tqdm(range(num_hands), desc="Evaluating vs amateur..."):
        current_block += np.array(play_hand_with_policies(game, policies, num_players))
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


def evaluate_vs_amateur_rotate(
    game,
    trainer,
    num_hands_per_seat=10000,
    num_players=NUM_PLAYERS,
    block_size=EVAL_BLOCK_SIZE,
    amateur=None,
):
    """
    Run evaluate_vs_amateur for cfr_seat=0,1,2 (BTN, SB, BB). Report per-seat and average CFR mbb/g.
    """
    if amateur is None:
        amateur = AmateurPolicy()
    seat_names = ["BTN", "SB", "BB"]
    cfr_mbb = []
    cfr_se = []
    for cfr_seat in range(num_players):
        mbb_mean, mbb_se = evaluate_vs_amateur(
            game, trainer,
            num_hands=num_hands_per_seat,
            num_players=num_players,
            cfr_seat=cfr_seat,
            block_size=block_size,
            amateur=amateur,
        )
        cfr_mbb.append(mbb_mean[cfr_seat])
        cfr_se.append(mbb_se[cfr_seat])

    print("\n" + "=" * 60)
    print("CFR vs Amateur — Rotation summary (button/SB/BB)")
    print("=" * 60)
    for i in range(num_players):
        print(f"  CFR as {seat_names[i]:<4}: mbb/g = {cfr_mbb[i]:.1f} ± {cfr_se[i]:.1f}")
    avg_mbb = sum(cfr_mbb) / num_players
    avg_se = (sum(s**2 for s in cfr_se) ** 0.5) / num_players  # approximate
    print(f"  CFR average:         mbb/g = {avg_mbb:.1f} ± {avg_se:.1f}")
    return cfr_mbb, cfr_se
