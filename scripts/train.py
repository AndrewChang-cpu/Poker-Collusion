#!/usr/bin/env python3
"""
Run MCCFR training and save blueprint strategy.

Usage:
  python scripts/train.py --iterations 10000 --out output/blueprint.pkl
  python scripts/train.py --load output/blueprint.pkl --iterations 5000   # run 5k more iters, append to same file
  python scripts/train.py --iterations 10000 --checkpoint-every 2000 --out output/blueprint_{iter}.pkl
"""

import os
import sys
import time
import argparse

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


# Game module interface for CFR (env is the module)
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


def main():
    ap = argparse.ArgumentParser(description="MCCFR training; optionally resume from --load and/or checkpoint with --checkpoint-every")
    ap.add_argument("--iterations", "-n", type=int, default=T_MAX_DEFAULT, help="Training iterations (additional if --load)")
    ap.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    ap.add_argument("--out", "-o", default="output/blueprint.pkl", help="Output path for final strategy")
    ap.add_argument("--load", "-l", default=None, help="Load existing strategy and continue training (run --iterations more)")
    ap.add_argument("--checkpoint-every", type=int, default=0, metavar="N", help="Save checkpoint every N iterations; --out can use {iter}")
    ap.add_argument("--no-prune", action="store_true", help="Disable regret pruning")
    ap.add_argument("--eval-hands", type=int, default=EVAL_HANDS_DEFAULT, help="Hands for post-training eval")
    args = ap.parse_args()

    game = GameModule()
    trainer = CFRTrainer(
        game,
        num_players=NUM_PLAYERS,
        use_linear_cfr=USE_LINEAR_CFR,
        prune_threshold=None if args.no_prune else PRUNE_THRESHOLD,
        prune_warm_up=PRUNE_WARM_UP_ITERATIONS,
        prune_skip_prob=PRUNE_SKIP_PROBABILITY,
    )

    if args.load:
        load_path = os.path.join(ROOT, args.load)
        if not os.path.isfile(load_path):
            print(f"Error: not found: {load_path}")
            sys.exit(1)
        trainer.load(load_path)
        print(f"Resuming from iter {trainer.iteration}; will run {args.iterations} more iterations.")
    else:
        print("Starting from scratch.")

    print("=" * 60)
    print("3-Player NLHE — MCCFR Blueprint Training")
    print("=" * 60)

    out_path = os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    checkpoint_path = out_path if args.checkpoint_every else None
    if args.checkpoint_every and "{iter}" not in out_path:
        base, ext = os.path.splitext(out_path)
        checkpoint_path = f"{base}_{{iter}}{ext}"

    start = time.time()
    trainer.train(
        num_iterations=args.iterations,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_every,
        checkpoint_path=checkpoint_path,
    )
    elapsed = time.time() - start
    print(f"Time: {elapsed:.1f}s")

    trainer.save(out_path)

    if args.eval_hands > 0:
        print("\n--- Evaluation ---")
        evaluate_with_variance(game, trainer, num_hands=args.eval_hands)


if __name__ == "__main__":
    main()
