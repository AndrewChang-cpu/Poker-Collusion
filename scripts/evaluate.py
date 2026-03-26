#!/usr/bin/env python3
"""
Load blueprint and evaluate mbb/g with block bootstrap SE.
Usage:
  python scripts/evaluate.py [--strategy output/blueprint.pkl] [--hands 50000]
  python scripts/evaluate.py --vs-amateur --strategy output/blueprint.pkl --hands 10000
  python scripts/evaluate.py --vs-amateur --rotate --hands 10000   # CFR in BTN/SB/BB, report average
  python scripts/evaluate.py --strategies p0.pkl p1.pkl p2.pkl --hands 50000  # per-player strategies
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
    is_chance_node,
    sample_chance,
)
from poker_collusion.cfr import CFRTrainer
from poker_collusion.evaluation import (
    evaluate_strategies,
    evaluate_rotate,
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
    is_chance_node = staticmethod(is_chance_node)
    sample_chance = staticmethod(sample_chance)


def _load_trainer(game: GameModule, path: str) -> CFRTrainer:
    full_path = os.path.join(ROOT, path)
    if not os.path.isfile(full_path):
        print(f"Strategy file not found: {full_path}")
        sys.exit(1)
    trainer = CFRTrainer(game, num_players=NUM_PLAYERS)
    trainer.load(full_path)
    return trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    strategy_group = ap.add_mutually_exclusive_group()
    strategy_group.add_argument(
        "--strategy", "-s", default=None,
        help="Path to a single saved strategy (used by all players or as CFR strategy vs amateur)",
    )
    strategy_group.add_argument(
        "--strategies", nargs="+", metavar="PATH",
        help="Per-player strategy paths: provide 1 (all same) or 3 (one per seat: BTN SB BB)",
    )
    ap.add_argument("--hands", type=int, default=EVAL_HANDS_DEFAULT)
    ap.add_argument("--block-size", type=int, default=EVAL_BLOCK_SIZE)
    ap.add_argument("--vs-amateur", action="store_true", help="Evaluate CFR vs amateur policy")
    ap.add_argument("--cfr-seat", type=int, default=0, choices=[0, 1, 2],
                    help="Seat for CFR when using --vs-amateur (0=BTN, 1=SB, 2=BB)")
    ap.add_argument("--rotate", action="store_true",
                    help="With --vs-amateur: run CFR in all three seats and report average")
    args = ap.parse_args()

    game = GameModule()

    # ── Per-player strategies mode ────────────────────────────────────────────
    if args.strategies is not None:
        if args.vs_amateur:
            print("--vs-amateur is not supported with --strategies; use --strategy instead.")
            sys.exit(1)

        if args.rotate:
            if len(args.strategies) != 2:
                print("--strategies --rotate requires exactly 2 paths (primary opponent).")
                sys.exit(1)
            primary = _load_trainer(game, args.strategies[0])
            opponent = _load_trainer(game, args.strategies[1])
            primary_name = os.path.basename(args.strategies[0])
            opponent_name = os.path.basename(args.strategies[1])
            print("=" * 60)
            print(f"Strategy Rotation: {primary_name} vs {opponent_name}")
            print("=" * 60)
            evaluate_rotate(
                game, primary, opponent,
                primary_name=primary_name,
                opponent_name=opponent_name,
                num_hands_per_seat=args.hands,
                block_size=args.block_size,
            )
        elif len(args.strategies) == 1:
            trainer = _load_trainer(game, args.strategies[0])
            print("=" * 60)
            print("Blueprint Evaluation (self-play)")
            print("=" * 60)
            evaluate_with_variance(
                game, trainer, num_hands=args.hands, block_size=args.block_size
            )
        elif len(args.strategies) == NUM_PLAYERS:
            trainers = [_load_trainer(game, p) for p in args.strategies]
            names = [os.path.basename(p) for p in args.strategies]
            print("=" * 60)
            print("Multi-Strategy Evaluation")
            print("=" * 60)
            evaluate_strategies(
                game,
                policies=trainers,
                names=names,
                num_hands=args.hands,
                block_size=args.block_size,
            )
        else:
            print(f"--strategies requires 1, 2 (with --rotate), or {NUM_PLAYERS} paths.")
            sys.exit(1)
        return

    # ── Single strategy mode (--strategy or default) ─────────────────────────
    strategy_path = args.strategy or "output/blueprint.pkl"
    trainer = _load_trainer(game, strategy_path)

    if args.vs_amateur:
        print("=" * 60)
        print("CFR vs Amateur Evaluation")
        print("=" * 60)
        if args.rotate:
            evaluate_vs_amateur_rotate(
                game, trainer,
                num_hands_per_seat=args.hands,
                block_size=args.block_size,
            )
        else:
            evaluate_vs_amateur(
                game, trainer,
                num_hands=args.hands,
                cfr_seat=args.cfr_seat,
                block_size=args.block_size,
            )
    else:
        print("=" * 60)
        print("Blueprint Evaluation (self-play)")
        print("=" * 60)
        evaluate_with_variance(
            game, trainer,
            num_hands=args.hands,
            block_size=args.block_size,
        )


if __name__ == "__main__":
    main()
