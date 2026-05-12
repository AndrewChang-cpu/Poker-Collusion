#!/usr/bin/env python3
"""
Run matches between PluribusBot (search-augmented) and opponents.

Usage:
  # 3 PluribusBot self-play
  python scripts/play.py --strategy output/blueprint.pkl --hands 100

  # 1 PluribusBot vs 2 amateurs
  python scripts/play.py --vs-amateur --strategy output/blueprint.pkl --hands 100

  # Rotate PluribusBot through all seats vs amateurs
  python scripts/play.py --vs-amateur --rotate --strategy output/blueprint.pkl --hands 100

  # Tune search parameters
  python scripts/play.py --vs-amateur --strategy output/blueprint.pkl --hands 100 \
      --cfr-iters 200 --depth-limit 3 --leaf-rollouts 10 --bias-factor 4.0
"""

import os
import sys
import argparse
import time

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
from poker_collusion.cfr.strategy import Strategy
from poker_collusion.evaluation.amateur_policy import AmateurPolicy
from poker_collusion.search.bot import PluribusBot
from poker_collusion.search.play import run_match, print_results
from poker_collusion.config import (
    BIAS_FACTOR,
    EVAL_BLOCK_SIZE,
    NUM_PLAYERS,
    SUBGAME_CFR_ITERATIONS,
    SUBGAME_DEPTH_LIMIT,
    SUBGAME_LEAF_ROLLOUTS,
)


class GameModule:
    deal_new_hand      = staticmethod(deal_new_hand)
    get_current_player = staticmethod(get_current_player)
    get_legal_actions  = staticmethod(get_legal_actions)
    get_info_key       = staticmethod(get_info_key)
    is_terminal        = staticmethod(is_terminal)
    get_payoffs        = staticmethod(get_payoffs)
    apply_action = staticmethod(apply_action)
    is_chance_node = staticmethod(is_chance_node)
    sample_chance = staticmethod(sample_chance)


def _load_blueprint(path: str) -> Strategy:
    full = os.path.join(ROOT, path)
    if not os.path.isfile(full):
        print(f"Strategy file not found: {full}")
        sys.exit(1)
    return Strategy.load(full)


def main() -> None:
    ap = argparse.ArgumentParser(description="PluribusBot match play")
    ap.add_argument("--strategy", "-s", required=True, help="Blueprint strategy file")
    ap.add_argument("--hands", "-n", type=int, default=100, help="Number of hands to play")
    ap.add_argument("--block-size", type=int, default=EVAL_BLOCK_SIZE)
    ap.add_argument("--vs-amateur", action="store_true", help="PluribusBot vs amateur opponents")
    ap.add_argument("--cfr-seat", type=int, default=0, choices=[0, 1, 2],
                    help="Seat for the PluribusBot (with --vs-amateur)")
    ap.add_argument("--rotate", action="store_true",
                    help="Rotate PluribusBot through all seats (with --vs-amateur)")
    ap.add_argument("--cfr-iters", type=int, default=SUBGAME_CFR_ITERATIONS,
                    help=f"Subgame CFR iterations (default: {SUBGAME_CFR_ITERATIONS})")
    ap.add_argument("--depth-limit", type=int, default=SUBGAME_DEPTH_LIMIT,
                    help=f"Subgame depth limit in streets (default: {SUBGAME_DEPTH_LIMIT})")
    ap.add_argument("--leaf-rollouts", type=int, default=SUBGAME_LEAF_ROLLOUTS,
                    help=f"MC rollouts per non-terminal leaf (default: {SUBGAME_LEAF_ROLLOUTS})")
    ap.add_argument("--bias-factor", type=float, default=BIAS_FACTOR,
                    help=f"Continuation strategy bias factor (default: {BIAS_FACTOR})")
    args = ap.parse_args()

    game = GameModule()
    blueprint = _load_blueprint(args.strategy)

    bot_kwargs = dict(
        game=game,
        blueprint=blueprint,
        num_players=NUM_PLAYERS,
        cfr_iterations=args.cfr_iters,
        depth_limit=args.depth_limit,
        leaf_rollouts=args.leaf_rollouts,
        bias_factor=args.bias_factor,
    )

    print("=" * 60)
    print("PluribusBot Match Play")
    print("=" * 60)
    print(f"Blueprint: {args.strategy}")
    print(f"Search params: cfr_iters={args.cfr_iters}, depth={args.depth_limit}, "
          f"rollouts={args.leaf_rollouts}, bias={args.bias_factor}")

    if args.vs_amateur:
        if args.rotate:
            _run_rotation(game, blueprint, bot_kwargs, args)
        else:
            _run_single_seat(game, blueprint, bot_kwargs, args)
    else:
        _run_self_play(game, blueprint, bot_kwargs, args)


def _run_self_play(game, blueprint, bot_kwargs, args):
    """All 3 seats are PluribusBot."""
    bots = [PluribusBot(seat=i, **bot_kwargs) for i in range(NUM_PLAYERS)]
    names = [f"Pluribus-{i}" for i in range(NUM_PLAYERS)]

    print(f"\nSelf-play: 3 × PluribusBot, {args.hands} hands")
    start = time.time()
    mbb, se = run_match(game, bots, args.hands, block_size=args.block_size)
    elapsed = time.time() - start

    print_results(mbb, se, names, args.hands)
    print(f"\nElapsed: {elapsed:.1f}s ({elapsed/args.hands:.2f}s/hand)")


def _run_single_seat(game, blueprint, bot_kwargs, args):
    """PluribusBot in one seat, amateurs in the rest."""
    seat = args.cfr_seat
    bots = []
    names = []
    for i in range(NUM_PLAYERS):
        if i == seat:
            bots.append(PluribusBot(seat=i, **bot_kwargs))
            names.append("Pluribus")
        else:
            bots.append(AmateurPolicy())
            names.append("Amateur")

    print(f"\nPluribusBot (seat {seat}) vs Amateur, {args.hands} hands")
    start = time.time()
    mbb, se = run_match(game, bots, args.hands, block_size=args.block_size)
    elapsed = time.time() - start

    print_results(mbb, se, names, args.hands)
    print(f"\nPluribusBot mbb/g: {mbb[seat]:.1f} ± {se[seat]:.1f}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed/args.hands:.2f}s/hand)")


def _run_rotation(game, blueprint, bot_kwargs, args):
    """Rotate PluribusBot through all 3 seats."""
    seat_names = ["BTN", "SB", "BB"]
    results = []

    for seat in range(NUM_PLAYERS):
        bots = []
        names = []
        for i in range(NUM_PLAYERS):
            if i == seat:
                bots.append(PluribusBot(seat=i, **bot_kwargs))
                names.append("Pluribus")
            else:
                bots.append(AmateurPolicy())
                names.append("Amateur")

        print(f"\n--- Rotation: Pluribus as {seat_names[seat]} ---")
        mbb, se = run_match(
            game, bots, args.hands, block_size=args.block_size,
            desc=f"Pluribus as {seat_names[seat]}",
        )
        print_results(mbb, se, names, args.hands)
        results.append((mbb[seat], se[seat]))

    print("\n" + "=" * 60)
    print("Rotation Summary")
    print("=" * 60)
    for i, (m, s) in enumerate(results):
        print(f"  Pluribus as {seat_names[i]:<4}: mbb/g = {m:.1f} ± {s:.1f}")
    avg_mbb = sum(r[0] for r in results) / NUM_PLAYERS
    avg_se = (sum(r[1] ** 2 for r in results) ** 0.5) / NUM_PLAYERS
    print(f"  Average:           mbb/g = {avg_mbb:.1f} ± {avg_se:.1f}")


if __name__ == "__main__":
    main()
