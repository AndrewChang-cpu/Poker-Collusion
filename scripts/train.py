#!/usr/bin/env python3
"""
Run MCCFR training and save blueprint strategy.
Supports parallelization via --processes with a smooth tqdm progress bar.
"""

import os
import sys
import time
import argparse
import multiprocessing
import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from poker_collusion.env import (
    deal_new_hand,
    get_current_player,
    get_legal_actions,
    get_info_key,
    is_terminal,
    get_payoffs,
    apply_action,
    undo_action,
    is_chance_node,
    sample_chance,
)
from poker_collusion.cfr import CFRTrainer
from poker_collusion.evaluation import evaluate_with_variance
from poker_collusion.config import (
    T_MAX_DEFAULT,
    LOG_INTERVAL,
    USE_LINEAR_CFR,
    PRUNE_THRESHOLD,
    PRUNE_WARM_UP_ITERATIONS,
    PRUNE_SKIP_PROBABILITY,
    EVAL_HANDS_DEFAULT,
    NUM_PLAYERS,
)


class GameModule:
    deal_new_hand = staticmethod(deal_new_hand)
    get_current_player = staticmethod(get_current_player)
    get_legal_actions = staticmethod(get_legal_actions)
    get_info_key = staticmethod(get_info_key)
    is_terminal = staticmethod(is_terminal)
    get_payoffs = staticmethod(get_payoffs)
    apply_action = staticmethod(apply_action)
    undo_action = staticmethod(undo_action)
    is_chance_node = staticmethod(is_chance_node)
    sample_chance = staticmethod(sample_chance)


def _parallel_train_worker(args):
    """Independent worker running a chunk of iterations."""
    num_iterations, load_path, seed = args
    np.random.seed(seed)
    
    game = GameModule()
    trainer = CFRTrainer(
        game,
        num_players=NUM_PLAYERS,
        use_linear_cfr=USE_LINEAR_CFR,
        prune_threshold=PRUNE_THRESHOLD,
        prune_warm_up=PRUNE_WARM_UP_ITERATIONS,
        prune_skip_prob=PRUNE_SKIP_PROBABILITY,
    )

    if load_path:
        trainer.load(load_path)

    # Run the assigned iterations for this chunk
    trainer.train(num_iterations=num_iterations, log_interval=0, show_progress=False)
    
    return {
        "regret_sum": trainer.regret_sum,
        "strategy_sum": trainer.strategy_sum,
        "action_map": trainer.action_map,
        "iteration": num_iterations # Return how many it contributed
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", "-n", type=int, default=T_MAX_DEFAULT)
    ap.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    ap.add_argument("--out", "-o", default="output/blueprint.pkl")
    ap.add_argument("--load", "-l", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--eval-hands", type=int, default=EVAL_HANDS_DEFAULT)
    ap.add_argument("--processes", "-j", type=int, default=1, help="Number of parallel processes")
    args = ap.parse_args()

    game = GameModule()
    main_trainer = CFRTrainer(
        game,
        num_players=NUM_PLAYERS,
        use_linear_cfr=USE_LINEAR_CFR,
        prune_threshold=None if args.no_prune else PRUNE_THRESHOLD,
        prune_warm_up=PRUNE_WARM_UP_ITERATIONS,
        prune_skip_prob=PRUNE_SKIP_PROBABILITY,
    )

    # Prepare initial baseline if loading
    baseline_iter = 0
    if args.load:
        load_full_path = os.path.join(ROOT, args.load)
        main_trainer.load(load_full_path)
        baseline_iter = main_trainer.iteration

    print("=" * 60)
    print(f"3-Player NLHE — Parallel MCCFR Training")
    print(f"Cores: {args.processes} | Total Iterations: {args.iterations}")
    print("=" * 60)

    start_time = time.time()

    if args.processes > 1:
        # Divide total iterations into chunks to show progress
        # 100 chunks provides a smooth bar without too much communication overhead
        num_chunks = max(args.processes, 100)
        iters_per_chunk = args.iterations // num_chunks
        
        worker_tasks = []
        for i in range(num_chunks):
            # The last chunk gets any remainder
            n = iters_per_chunk if i < num_chunks - 1 else args.iterations - (iters_per_chunk * (num_chunks - 1))
            seed = int(time.time() * 1000) % 2**32 + i
            worker_tasks.append((n, args.load, seed))
            
        all_results = []
        with multiprocessing.Pool(processes=args.processes) as pool:
            # imap_unordered allows the tqdm bar to update as soon as ANY chunk finishes
            pbar = tqdm(total=args.iterations, desc="Parallel Training", unit="iter")
            for result in pool.imap_unordered(_parallel_train_worker, worker_tasks):
                all_results.append(result)
                pbar.update(result["iteration"])
            pbar.close()
            
        print("\nMerging and Averaging results...")
        # Reset main trainer sums to aggregate worker data
        main_trainer.regret_sum = {}
        main_trainer.strategy_sum = {}
        
        for res in all_results:
            temp = CFRTrainer(game)
            temp.regret_sum = res["regret_sum"]
            temp.strategy_sum = res["strategy_sum"]
            temp.action_map = res["action_map"]
            main_trainer.merge(temp)
            
        # Divide by num_chunks to get the average across all independent samples
        for key in main_trainer.regret_sum:
            main_trainer.regret_sum[key] /= num_chunks
        for key in main_trainer.strategy_sum:
            main_trainer.strategy_sum[key] /= num_chunks
            
        main_trainer.iteration = baseline_iter + args.iterations
    else:
        # Standard sequential training
        main_trainer.train(
            num_iterations=args.iterations,
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_every,
            checkpoint_path=args.out
        )

    print(f"Total Time: {time.time() - start_time:.1f}s")
    out_path = os.path.join(ROOT, args.out)
    main_trainer.save(out_path)

    if args.eval_hands > 0:
        print("\n--- Final Evaluation ---")
        evaluate_with_variance(game, main_trainer, num_hands=args.eval_hands, num_processes=args.processes)


if __name__ == "__main__":
    main()