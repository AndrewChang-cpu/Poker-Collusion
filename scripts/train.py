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

# Ensure the project root is in the path for internal imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Bypassing the package-level __init__ to avoid CFRDebugger/terminal_display dependency issues
from poker_collusion.cfr.trainer import CFRTrainer
from poker_collusion.env import (
    deal_new_hand, get_current_player, get_legal_actions,
    get_info_key, is_terminal, get_payoffs, apply_action,
    is_chance_node, sample_chance, evaluate_hand,
)
from poker_collusion.env.game_logic import _resolve_side_pots
from poker_collusion.evaluation.mbbg import evaluate_strategies
from poker_collusion.config import NUM_PLAYERS, LOG_INTERVAL
from tqdm import tqdm


class GameModule:
    """
    Comprehensive interface for the CFRTrainer to interact with the game environment.
    Exposes core CFR methods and auxiliary environment logic.
    """
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


def _eval_selfplay_policies(game, policies, num_hands):
    """Evaluate per-seat policies; returns (btn, sb, bb) mbb/g."""
    silent = io.StringIO()
    with contextlib.redirect_stdout(silent):
        mbb_mean, _ = evaluate_strategies(game, policies, num_hands=num_hands)
    return tuple(float(x) for x in mbb_mean)


def _eval_selfplay(game, trainer, num_hands):
    """Self-play evaluation; returns (btn, sb, bb) mbb/g."""
    return _eval_selfplay_policies(game, [trainer] * NUM_PLAYERS, num_hands)


def _save_curve_data(checkpoints, path):
    file_exists = os.path.isfile(path)
    with open(path, "a") as f:
        if not file_exists:
            f.write("iteration\tBTN\tSB\tBB\n")
        for it, btn, sb, bb in checkpoints:
            f.write(f"{it}\t{btn:.2f}\t{sb:.2f}\t{bb:.2f}\n")
    print(f"Saved curve data to {path}")


def _plot_curve(checkpoints, plot_out):
    """Generate a learning curve plot if matplotlib is available."""
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
                        help="Target total number of iterations (e.g. 200000)")
    parser.add_argument("--out", "-o", default="output/blueprint.pkl")
    
    # Support for victim modeling and resuming from multiple sources
    parser.add_argument("--resume-from", nargs="+", metavar="PATH",
                        help="One or more existing .pkl files to load/merge")
    
    parser.add_argument("--team-seats", type=str, default=None)
    parser.add_argument("--shared-info", action="store_true",
                        help="Full-comm: include teammate card rank in info key (team members only)")
    parser.add_argument("--team-objective", default="utilitarian")
    
    parser.add_argument("--train-seats", type=str, default=None,
                        help="Seats to train (traversers). Default: all. Example: '2' for victim only.")
    
    parser.add_argument("--reset-strategy-sum", action="store_true",
                        help="Clear the accumulated average strategy (S) while keeping regrets (R)")
    
    parser.add_argument("--frozen-strategy", type=str, default=None,
                        help="Path to pre-trained .pkl to act as static strategy for non-training seats")
    
    parser.add_argument("--plot-every", type=int, default=0, metavar="N",
                        help="Evaluate and record a curve point every N iterations (0 = disabled)")
    parser.add_argument("--eval-hands", type=int, default=100_000, metavar="N",
                        help="Hands per curve evaluation (default: 100000)")
    parser.add_argument("--plot-out", type=str, default=None, metavar="PATH",
                        help="Output path for the learning curve PNG (default: <out>.png)")
    # Co-evolution flags
    parser.add_argument("--coevolve", action="store_true",
                        help="Simultaneously train team + victim against each other")
    parser.add_argument("--victim-seat", type=int, default=2, choices=[0, 1, 2],
                        help="Seat index for the victim in co-evolution (default: 2)")
    parser.add_argument("--victim-init", type=str, default=None, metavar="PATH",
                        help="Initial victim strategy pkl for co-evolution")
    parser.add_argument("--victim-out", type=str, default=None, metavar="PATH",
                        help="Output path for co-evolved victim pkl (default: <out>_victim.pkl)")
    args = parser.parse_args()

    team_seats = [int(s) for s in args.team_seats.split(",")] if args.team_seats else None
    train_seats = [int(s) for s in args.train_seats.split(",")] if args.train_seats else None
    game = GameModule()

    # Load frozen strategy for baseline opponents
    frozen_trainer = None
    if args.frozen_strategy:
        if os.path.exists(args.frozen_strategy):
            print(f"Loading frozen strategy from {args.frozen_strategy}...")
            frozen_trainer = CFRTrainer(GameModule(), num_players=NUM_PLAYERS)
            frozen_trainer.load(args.frozen_strategy)
        else:
            print(f"Warning: {args.frozen_strategy} not found. Proceeding without freezing.")

    # Initialize the main trainer
    trainer = CFRTrainer(
        game,
        num_players=NUM_PLAYERS,
        use_shared_info=args.shared_info,
        team_seats=team_seats,
        train_seats=train_seats,
        frozen_trainer=frozen_trainer,
        team_objective=args.team_objective,
    )

    # Resume from existing checkpoints, potentially merging multiple files
    if args.resume_from:
        for i, path in enumerate(args.resume_from):
            if os.path.exists(path):
                is_merge = (i > 0)
                print(f"Loading checkpoint from {path} (merge={is_merge})...")
                trainer.load(path, merge=is_merge)
            else:
                print(f"Error: {path} not found.")
                sys.exit(1)

    # Optionally reset strategy sum to allow pre-trained agents to pivot faster
    if args.reset_strategy_sum:
        print("Resetting accumulated strategy sum for all seats...")
        trainer.reset_strategy_sum()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ── Co-evolution mode ────────────────────────────────────────────────────
    if args.coevolve:
        victim_seat = args.victim_seat
        team_seats_coev = [s for s in range(NUM_PLAYERS) if s != victim_seat]

        team_trainer = CFRTrainer(game, num_players=NUM_PLAYERS,
                                  team_seats=team_seats_coev,
                                  team_objective=args.team_objective)
        victim_trainer = CFRTrainer(game, num_players=NUM_PLAYERS)

        if args.resume_from:
            team_trainer.load(args.resume_from[0])
        if args.victim_init and os.path.exists(args.victim_init):
            victim_trainer.load(args.victim_init)

        # Circular live references
        team_trainer.frozen_trainer = victim_trainer
        victim_trainer.frozen_trainer = team_trainer

        base_iter = max(team_trainer.iteration, victim_trainer.iteration)
        checkpoints = []

        pbar = tqdm(range(1, args.iterations + 1), desc="Co-evolving")
        for i in pbar:
            t = base_iter + i
            team_trainer.iteration = t
            for seat in sorted(team_seats_coev):
                team_trainer.cfr_traverse(game.deal_new_hand(), traverser=seat)

            victim_trainer.iteration = t
            victim_trainer.cfr_traverse(game.deal_new_hand(), traverser=victim_seat)

            if i % LOG_INTERVAL == 0:
                pbar.set_postfix({
                    "t_regret": f"{team_trainer._calculate_avg_regret():.2e}",
                    "v_regret": f"{victim_trainer._calculate_avg_regret():.2e}",
                })

            if args.plot_every > 0 and i % args.plot_every == 0:
                policies = [team_trainer if s != victim_seat else victim_trainer
                            for s in range(NUM_PLAYERS)]
                btn, sb, bb = _eval_selfplay_policies(game, policies, args.eval_hands)
                checkpoints.append((t, btn, sb, bb))
                print(f"  iter {t:>10,}  BTN {btn:+.1f}  SB {sb:+.1f}  BB {bb:+.1f}")

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        team_trainer.save(args.out)
        victim_out = args.victim_out or os.path.splitext(args.out)[0] + "_victim.pkl"
        victim_trainer.save(victim_out)
        print(f"Co-evolution complete. Team saved to {args.out}, victim saved to {victim_out}")

        if checkpoints:
            plot_out = args.plot_out or os.path.splitext(args.out)[0] + "_curve.png"
            _plot_curve(checkpoints, plot_out)
            _save_curve_data(checkpoints, os.path.splitext(plot_out)[0] + ".txt")
        return

    # ── Standard training loop ───────────────────────────────────────────────
    checkpoints = []

    if args.plot_every > 0:
        current = trainer.iteration
        while current < args.iterations:
            next_target = min(current + args.plot_every, args.iterations)
            trainer.train(target_iterations=next_target)
            current = trainer.iteration
            # Use frozen_trainer for non-team seats so the curve reflects
            # team performance against the actual frozen opponent, not self-play.
            if frozen_trainer is not None and trainer.team_seats:
                policies = [
                    trainer if s in trainer.team_seats else frozen_trainer
                    for s in range(NUM_PLAYERS)
                ]
                btn, sb, bb = _eval_selfplay_policies(game, policies, args.eval_hands)
            else:
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
        _save_curve_data(checkpoints, os.path.splitext(plot_out)[0] + ".txt")


if __name__ == "__main__":
    main()