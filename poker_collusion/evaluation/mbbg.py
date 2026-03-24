"""
Self-play evaluation: mbb/g and block bootstrap standard error.
Supports parallel execution via multiprocessing.
"""

import numpy as np
import multiprocessing
import os
import time
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


# --- Parallel Worker Helpers ---

def _run_batch_play(args):
    """Worker function for self-play evaluation."""
    game, trainer, num_hands, num_players, seed = args
    np.random.seed(seed)
    batch_results = []
    for _ in range(num_hands):
        batch_results.append(play_hand(game, trainer, num_players))
    return batch_results


def _run_batch_policies(args):
    """Worker function for vs-policy evaluation."""
    game, policies, num_hands, num_players, seed = args
    np.random.seed(seed)
    batch_results = []
    for _ in range(num_hands):
        batch_results.append(play_hand_with_policies(game, policies, num_players))
    return batch_results


def _collect_parallel_results(num_hands, num_processes, worker_fn, worker_args_base):
    """Generic orchestrator for multiprocessing evaluation."""
    if num_processes < 1:
        num_processes = multiprocessing.cpu_count()
    
    # Batch size of 500 prevents excessive IPC overhead
    batch_size = 500
    num_batches = (num_hands + batch_size - 1) // batch_size
    
    tasks = []
    for i in range(num_batches):
        hands_in_batch = min(batch_size, num_hands - i * batch_size)
        # Unique seed for every batch
        seed = int(time.time() * 1000) % 2**32 + i + os.getpid()
        tasks.append(worker_args_base + (hands_in_batch, NUM_PLAYERS, seed))
        
    all_payoffs = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        for batch_result in tqdm(pool.imap_unordered(worker_fn, tasks), 
                                total=num_batches, desc="Parallel Eval"):
            all_payoffs.extend(batch_result)
            
    return np.array(all_payoffs)


# --- Public Evaluation API ---

def evaluate(game, trainer, num_hands=10000, num_players=NUM_PLAYERS, num_processes=1):
    """
    Run basic evaluation; return (avg_payoffs, mbb_per_game) per player.
    mbb/g = (avg profit per hand in BB) * 1000.
    """
    if num_processes > 1:
        all_payoffs = _collect_parallel_results(num_hands, num_processes, _run_batch_play, (game, trainer))
    else:
        all_payoffs = []
        for _ in range(num_hands):
            all_payoffs.append(play_hand(game, trainer, num_players))
        all_payoffs = np.array(all_payoffs)
        
    avg_payoffs = np.mean(all_payoffs, axis=0)
    mbb_per_game = avg_payoffs * 1000
    return avg_payoffs, mbb_per_game


def evaluate_with_variance(
    game,
    trainer,
    num_hands=10000,
    num_players=NUM_PLAYERS,
    block_size=EVAL_BLOCK_SIZE,
    num_processes=1
):
    """
    Evaluate with block bootstrap standard error and 95% CI.
    Supports parallel execution.
    """
    if num_processes > 1:
        all_payoffs = _collect_parallel_results(num_hands, num_processes, _run_batch_play, (game, trainer))
    else:
        all_payoffs = []
        for _ in tqdm(range(num_hands), "Evaluating..."):
            all_payoffs.append(play_hand(game, trainer, num_players))
        all_payoffs = np.array(all_payoffs)

    # Apply block bootstrapping to the results
    block_payoffs = []
    num_blocks = len(all_payoffs) // block_size
    for i in range(num_blocks):
        block = all_payoffs[i*block_size : (i+1)*block_size]
        block_payoffs.append(np.mean(block, axis=0))
    
    # Handle remaining hands
    if len(all_payoffs) % block_size != 0:
        remaining = all_payoffs[num_blocks*block_size:]
        block_payoffs.append(np.mean(remaining, axis=0))

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
    num_processes=1
):
    """
    Evaluate CFR (trainer) vs amateur policy. Supports parallel execution.
    """
    if amateur is None:
        amateur = AmateurPolicy()
    policies = [amateur] * num_players
    policies[cfr_seat] = trainer

    if num_processes > 1:
        all_payoffs = _collect_parallel_results(num_hands, num_processes, _run_batch_policies, (game, policies))
    else:
        all_payoffs = []
        for _ in tqdm(range(num_hands), desc="Evaluating vs amateur..."):
            all_payoffs.append(play_hand_with_policies(game, policies, num_players))
        all_payoffs = np.array(all_payoffs)

    # Apply block bootstrapping
    block_payoffs = []
    num_blocks = len(all_payoffs) // block_size
    for i in range(num_blocks):
        block = all_payoffs[i*block_size : (i+1)*block_size]
        block_payoffs.append(np.mean(block, axis=0))
    
    if len(all_payoffs) % block_size != 0:
        remaining = all_payoffs[num_blocks*block_size:]
        block_payoffs.append(np.mean(remaining, axis=0))

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
    num_processes=1
):
    """
    Run evaluate_vs_amateur for cfr_seat=0,1,2 (BTN, SB, BB). Supports parallel execution.
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
            num_processes=num_processes
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