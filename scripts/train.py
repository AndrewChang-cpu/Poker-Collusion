#!/usr/bin/env python3
"""
Run MCCFR training and save blueprint strategy.
Supports parallelization via --processes.
"""

import os
import sys
import time
import argparse
import multiprocessing
import numpy as np

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
    PRUNE_WAR_UP_ITERATIONS,
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
    """Independent worker running a subset of iterations."""
    num_iterations, load_path, seed, worker_id = args
    np.random.seed(seed)
    
    game = GameModule()
    trainer = CFRTrainer(
        game,
        num_players=NUM_PLAYERS,
        use_linear_cfr=USE_LINEAR_CFR,
        prune_threshold=PRUNE_THRESHOLD,
        prune_warm_up=PRUNE_WAR_UP_ITERATIONS,
        prune_skip_prob=PRUNE_SKIP_PROBABILITY,
    )

    if load_path:
        trainer.load(load_path)

    # Workers don't show individual progress bars to avoid overlapping output
    trainer.train(num_iterations=num_iterations, log_interval=0, show_progress=False)
    
    return {
        "regret_sum": trainer.regret_sum,
        "strategy_sum": trainer.strategy_sum,
        "action_map": trainer.action_map,
        "iteration": trainer.iteration
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
        prune_warm_up=PRUNE_WAR_UP_ITERATIONS,
        prune_skip_prob=PRUNE_SKIP_PROBABILITY,
    )

    if args.load:
        main_trainer.load(os.path.join(ROOT, args.load))

    print("=" * 60)
    print(f"3-Player NLHE — MCCFR Training (Processes: {args.processes})")
    print("=" * 60)

    start_time = time.time()

    if args.processes > 1:
        iters_per_proc = args.iterations // args.processes
        rem = args.iterations % args.processes
        
        worker_tasks = []
        for i in range(args.processes):
            n = iters_per_proc + (rem if i == 0 else 0)
            seed = int(time.time() * 1000) % 2**32 + i
            worker_tasks.append((n, args.load, seed, i))
            
        with multiprocessing.Pool(processes=args.processes) as pool:
            print(f"Dispatching {args.iterations} iterations across {args.processes} workers...")
            results = pool.map(_parallel_train_worker, worker_tasks)
            
        print("\nMerging results...")
        baseline_iter = main_trainer.iteration
        
        # Reset main trainer sums to start averaging
        main_trainer.regret_sum = {}
        main_trainer.strategy_sum = {}
        
        for res in results:
            temp = CFRTrainer(game)
            temp.regret_sum = res["regret_sum"]
            temp.strategy_sum = res["strategy_sum"]
            temp.action_map = res["action_map"]
            main_trainer.merge(temp)
            
        # Average the sums across workers
        for key in main_trainer.regret_sum:
            main_trainer.regret_sum[key] /= args.processes
        for key in main_trainer.strategy_sum:
            main_trainer.strategy_sum[key] /= args.processes
            
        main_trainer.iteration = baseline_iter + args.iterations
    else:
        main_trainer.train(
            num_iterations=args.iterations,
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_every,
            checkpoint_path=args.out
        )

    print(f"Time: {time.time() - start_time:.1f}s")
    out_path = os.path.join(ROOT, args.out)
    main_trainer.save(out_path)

    if args.eval_hands > 0:
        print("\n--- Evaluation ---")
        evaluate_with_variance(game, main_trainer, num_hands=args.eval_hands, num_processes=args.processes)


if __name__ == "__main__":
    main()