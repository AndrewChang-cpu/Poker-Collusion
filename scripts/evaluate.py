#!/usr/bin/env python3
"""
Load blueprint and evaluate mbb/g with block bootstrap SE.
Modified for 3-player Leduc Hold'em and Psychic Collusion support.
Standardized CLI overrides for Shared Information (Step 3).
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
    summarize_team,
)
from poker_collusion.config import EVAL_HANDS_DEFAULT, EVAL_BLOCK_SIZE, NUM_PLAYERS


class GameModule:
    """Interface for evaluation logic to interact with the Leduc environment."""
    deal_new_hand      = staticmethod(deal_new_hand)
    get_current_player = staticmethod(get_current_player)
    get_legal_actions  = staticmethod(get_legal_actions)
    get_info_key       = staticmethod(get_info_key)
    is_terminal        = staticmethod(is_terminal)
    get_payoffs        = staticmethod(get_payoffs)
    apply_action       = staticmethod(apply_action)
    is_chance_node     = staticmethod(is_chance_node)
    sample_chance      = staticmethod(sample_chance)


def _load_trainer(game: GameModule, path: str, team_seats=None, shared_info=None) -> CFRTrainer:
    """Load a strategy file and optionally override psychic/team metadata."""
    full_path = os.path.join(ROOT, path)
    if not os.path.isfile(full_path):
        print(f"Strategy file not found: {full_path}")
        sys.exit(1)
    trainer = CFRTrainer(game, num_players=NUM_PLAYERS)
    trainer.load(full_path) # Values loaded from .pkl
    
    # Apply command line overrides (Step 3: Standardized Overrides)
    if team_seats is not None:
        trainer.team_seats = set(team_seats)
    if shared_info is not None: # Handles explicit True/False from CLI
        trainer.use_shared_info = shared_info
        
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
    
    # Psychic support flags (Step 3: BooleanOptionalAction for explicit True/False/None)
    ap.add_argument("--shared-info", action=argparse.BooleanOptionalAction, 
                    help="Force Shared Information (Psychic) keys on/off during evaluation")
    ap.add_argument("--team-seats", type=str, default=None, help="Force team seats (comma-separated, e.g. '0,1')")

    ap.add_argument("--team-eval", action="store_true",
                    help="Evaluate a team checkpoint vs a frozen opponent checkpoint")
    ap.add_argument("--team-strategy", type=str, default=None, metavar="PATH",
                    help="Path to team-trained .pkl (required with --team-eval)")
    ap.add_argument("--frozen-strategy", type=str, default=None, metavar="PATH",
                    help="Path to frozen opponent .pkl (required with --team-eval)")
    args = ap.parse_args()

    game = GameModule()
    
    # Parse manual team seats if provided via CLI
    manual_team_seats = [int(s) for s in args.team_seats.split(",")] if args.team_seats else None

    # ── Team evaluation mode ──────────────────────────────────────────────────
    if args.team_eval:
        if not args.team_strategy:
            print("Error: --team-strategy is required with --team-eval.")
            sys.exit(1)
        if not args.frozen_strategy:
            print("Error: --frozen-strategy is required with --team-eval.")
            sys.exit(1)

        # Load strategies, applying overrides directly
        team_trainer = _load_trainer(game, args.team_strategy, 
                                     team_seats=manual_team_seats, 
                                     shared_info=args.shared_info)
        frozen_trainer = _load_trainer(game, args.frozen_strategy)

        team_seats = sorted(team_trainer.team_seats)
        if not team_seats:
            print("Error: team strategy has no team_seats metadata. Use --team-seats to override.")
            sys.exit(1)
            
        frozen_seats = [s for s in range(NUM_PLAYERS) if s not in team_seats]

        seat_labels = ["BTN", "SB", "BB"]
        policies = [None] * NUM_PLAYERS
        names = [None] * NUM_PLAYERS
        for s in team_seats:
            policies[s] = team_trainer
            names[s] = f"Team ({seat_labels[s]})"
        for s in frozen_seats:
            policies[s] = frozen_trainer
            names[s] = f"Frozen ({seat_labels[s]})"

        obj = getattr(team_trainer, "team_objective", "utilitarian")
        print("=" * 60)
        print(f"Team Evaluation: seats {team_seats} vs frozen {frozen_seats} (objective={obj})")
        print("=" * 60)

        mbb_mean, mbb_se = evaluate_strategies(
            game,
            policies=policies,
            names=names,
            num_hands=args.hands,
            block_size=args.block_size,
        )
        summarize_team(mbb_mean, mbb_se, team_seats)
        return

    # ── Per-player strategies mode ────────────────────────────────────────────
    if args.strategies is not None:
        if args.vs_amateur:
            print("--vs-amateur is not supported with --strategies; use --strategy instead.")
            sys.exit(1)

        if args.rotate:
            if len(args.strategies) != 2:
                print("--strategies --rotate requires exactly 2 paths (primary opponent).")
                sys.exit(1)
            primary = _load_trainer(game, args.strategies[0], team_seats=manual_team_seats, shared_info=args.shared_info)
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
            trainer = _load_trainer(game, args.strategies[0], team_seats=manual_team_seats, shared_info=args.shared_info)
            print("=" * 60)
            print("Blueprint Evaluation (self-play)")
            print("=" * 60)
            evaluate_with_variance(
                game, trainer, num_hands=args.hands, block_size=args.block_size
            )
        elif len(args.strategies) == NUM_PLAYERS:
            trainers = [_load_trainer(game, p) for p in args.strategies]
            names = [os.path.basename(p) for p in args.strategies]
            
            # Apply manual overrides to all trainers if specified via CLI
            for t in trainers:
                if manual_team_seats: t.team_seats = set(manual_team_seats)
                if args.shared_info is not None: t.use_shared_info = args.shared_info
                    
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

    # ── Single strategy mode ──────────────────────────────────────────────────
    strategy_path = args.strategy or "output/blueprint.pkl"
    trainer = _load_trainer(game, strategy_path, team_seats=manual_team_seats, shared_info=args.shared_info)

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