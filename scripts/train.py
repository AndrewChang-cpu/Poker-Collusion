"""
MCCFR training script.

Usage:
  python scripts/train.py --iterations 10000 --out output/blueprint.pkl

  # Resume training to a target total
  python scripts/train.py --resume-from output/leduc_1m.pkl --iterations 10000000 --out output/leduc_10m.pkl

  # Plot a per-seat mbb/g learning curve (evaluated every 1M iterations, 100k hands each)
  python scripts/train.py --iterations 10000000 --out output/leduc_10m.pkl --plot-every 1000000

  # Team: observable signaling (team seats 0,1 vs frozen BB)
  python scripts/train.py --team-seats 0,1 \
      --frozen-strategy output/leduc_ne.pkl --iterations 500000 --out output/leduc_obs_signal.pkl

  # Team: free communication (full info keys for team members)
  python scripts/train.py --team-seats 0,1 --shared-info \
      --frozen-strategy output/leduc_ne.pkl --iterations 500000 --out output/leduc_free_comm.pkl
"""
import argparse
import contextlib
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from poker_collusion.cfr import CFRTrainer
from poker_collusion.env import (
    deal_new_hand, get_current_player, get_legal_actions,
    get_info_key, is_terminal, get_payoffs, apply_action,
    is_chance_node, sample_chance, evaluate_hand,
)
from poker_collusion.env.game_logic import _resolve_side_pots
from poker_collusion.evaluation.mbbg import evaluate_strategies
from poker_collusion.config import NUM_PLAYERS


class GameModule:
    deal_new_hand      = staticmethod(deal_new_hand)
    get_current_player = staticmethod(get_current_player)
    get_legal_actions  = staticmethod(get_legal_actions)
    get_info_key       = staticmethod(get_info_key)
    is_terminal        = staticmethod(is_terminal)
    get_payoffs        = staticmethod(get_payoffs)
    apply_action       = staticmethod(apply_action)
    is_chance_node     = staticmethod(is_chance_node)
    sample_chance      = staticmethod(sample_chance)
    evaluate_hand      = staticmethod(evaluate_hand)
    _resolve_side_pots = staticmethod(_resolve_side_pots)


def _eval_selfplay(game, trainer, num_hands):
    """Self-play evaluation; returns (btn, sb, bb) mbb/g."""
    silent = io.StringIO()
    with contextlib.redirect_stdout(silent):
        mbb_mean, _ = evaluate_strategies(
            game, [trainer] * NUM_PLAYERS, num_hands=num_hands
        )
    return tuple(float(x) for x in mbb_mean)


def _plot_curve(checkpoints, plot_out):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot. Run: pip install matplotlib")
        return

    iters   = [c[0] for c in checkpoints]
    btn_mbb = [c[1] for c in checkpoints]
    sb_mbb  = [c[2] for c in checkpoints]
    bb_mbb  = [c[3] for c in checkpoints]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, btn_mbb, marker="o", markersize=3, label="BTN (seat 0)")
    ax.plot(iters, sb_mbb,  marker="s", markersize=3, label="SB (seat 1)")
    ax.plot(iters, bb_mbb,  marker="^", markersize=3, label="BB (seat 2)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Training iterations")
    ax.set_ylabel("mbb/g (self-play)")
    ax.set_title("Per-seat mbb/g learning curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_out, dpi=150)
    print(f"Saved learning curve to {plot_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", "-n", type=int, default=1000,
                        help="Target total number of iterations")
    parser.add_argument("--out", "-o", default="output/blueprint.pkl")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to an existing .pkl to resume training from")
    parser.add_argument("--team-seats", type=str, default=None)
    parser.add_argument("--shared-info", action="store_true",
                        help="Full-comm: include teammate card rank in info key (team members only)")
    parser.add_argument("--team-objective", default="utilitarian")
    parser.add_argument("--frozen-strategy", type=str, default=None,
                        help="Path to pre-trained .pkl to freeze non-team seats")
    parser.add_argument("--plot-every", type=int, default=0, metavar="N",
                        help="Evaluate and record a curve point every N iterations (0 = disabled)")
    parser.add_argument("--eval-hands", type=int, default=100_000, metavar="N",
                        help="Hands per curve evaluation (default: 100000)")
    parser.add_argument("--plot-out", type=str, default=None, metavar="PATH",
                        help="Output path for the learning curve PNG (default: <out>.png)")
    args = parser.parse_args()

    team_seats = [int(s) for s in args.team_seats.split(",")] if args.team_seats else None
    game = GameModule()

    frozen_trainer = None
    if args.frozen_strategy:
        if os.path.exists(args.frozen_strategy):
            print(f"Loading frozen strategy from {args.frozen_strategy}...")
            frozen_trainer = CFRTrainer(GameModule(), num_players=NUM_PLAYERS)
            frozen_trainer.load(args.frozen_strategy)
        else:
            print(f"Warning: {args.frozen_strategy} not found. Proceeding without freezing.")

    trainer = CFRTrainer(
        game,
        num_players=NUM_PLAYERS,
        use_shared_info=args.shared_info,
        team_seats=team_seats,
        frozen_trainer=frozen_trainer,
        team_objective=args.team_objective,
    )

    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"Resuming training from {args.resume_from}...")
            trainer.load(args.resume_from)
        else:
            print(f"Error: {args.resume_from} not found.")
            sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────────
    checkpoints = []

    if args.plot_every > 0:
        current = trainer.iteration
        while current < args.iterations:
            next_target = min(current + args.plot_every, args.iterations)
            trainer.train(target_iterations=next_target)
            current = trainer.iteration
            btn, sb, bb = _eval_selfplay(game, trainer, args.eval_hands)
            checkpoints.append((current, btn, sb, bb))
            print(f"  iter {current:>10,}  BTN {btn:+.1f}  SB {sb:+.1f}  BB {bb:+.1f}")
    else:
        trainer.train(target_iterations=args.iterations)

    trainer.save(args.out)
    print(f"Training complete. Final iteration: {trainer.iteration}. Strategy saved to {args.out}")

    if checkpoints:
        plot_out = args.plot_out or os.path.splitext(args.out)[0] + "_curve.png"
        _plot_curve(checkpoints, plot_out)


if __name__ == "__main__":
    main()
