#!/usr/bin/env python3
"""
Load blueprint and evaluate mbb/g with block bootstrap SE.
Usage:
  python scripts/evaluate.py [--strategy output/blueprint.pkl] [--hands 50000]
  python scripts/evaluate.py --vs-amateur --strategy output/blueprint.pkl --hands 10000
  python scripts/evaluate.py --vs-amateur --rotate --hands 10000   # CFR in BTN/SB/BB, report average
"""

import os
import sys
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
from poker_collusion.evaluation import (
    evaluate_with_variance,
    evaluate_vs_amateur,
    evaluate_vs_amateur_rotate,
)
from poker_collusion.config import EVAL_HANDS_DEFAULT, EVAL_BLOCK_SIZE, NUM_PLAYERS


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", "-s", default="output/blueprint.pkl", help="Path to saved strategy")
    ap.add_argument("--hands", type=int, default=EVAL_HANDS_DEFAULT)
    ap.add_argument("--block-size", type=int, default=EVAL_BLOCK_SIZE)
    ap.add_argument("--vs-amateur", action="store_true", help="Evaluate CFR vs amateur policy")
    ap.add_argument("--cfr-seat", type=int, default=0, choices=[0, 1, 2],
                    help="Seat for CFR when using --vs-amateur (0=BTN, 1=SB, 2=BB)")
    ap.add_argument("--rotate", action="store_true",
                    help="With --vs-amateur: run CFR in all three seats and report average (button/SB/BB rotation)")
    args = ap.parse_args()

    path = os.path.join(ROOT, args.strategy)
    if not os.path.isfile(path):
        print(f"Strategy file not found: {path}")
        sys.exit(1)

    game = GameModule()
    trainer = CFRTrainer(game, num_players=NUM_PLAYERS)
    trainer.load(path)

    if args.vs_amateur:
        print("=" * 60)
        print("CFR vs Amateur Evaluation")
        print("=" * 60)
        if args.rotate:
            evaluate_vs_amateur_rotate(
                game,
                trainer,
                num_hands_per_seat=args.hands,
                block_size=args.block_size,
            )
        else:
            evaluate_vs_amateur(
                game,
                trainer,
                num_hands=args.hands,
                cfr_seat=args.cfr_seat,
                block_size=args.block_size,
            )
    else:
        print("=" * 60)
        print("Blueprint Evaluation (self-play)")
        print("=" * 60)
        evaluate_with_variance(
            game,
            trainer,
            num_hands=args.hands,
            block_size=args.block_size,
        )


if __name__ == "__main__":
    main()
