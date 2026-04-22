"""
MCCFR training script.

Usage:
  python scripts/train.py --iterations 10000 --out output/blueprint.pkl

  # Resume training to a target total
  python scripts/train.py --resume-from output/leduc_1m.pkl --iterations 10000000 --out output/leduc_10m.pkl

  # Team: observable signaling (team seats 0,1 vs frozen BB)
  python scripts/train.py --team-seats 0,1 \
      --frozen-strategy output/leduc_ne.pkl --iterations 500000 --out output/leduc_obs_signal.pkl

  # Team: free communication (full info keys for team members)
  python scripts/train.py --team-seats 0,1 --shared-info \
      --frozen-strategy output/leduc_ne.pkl --iterations 500000 --out output/leduc_free_comm.pkl
"""
import argparse
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
    args = parser.parse_args()

    team_seats = [int(s) for s in args.team_seats.split(",")] if args.team_seats else None

    frozen_trainer = None
    if args.frozen_strategy:
        if os.path.exists(args.frozen_strategy):
            print(f"Loading frozen strategy from {args.frozen_strategy}...")
            frozen_trainer = CFRTrainer(GameModule(), num_players=NUM_PLAYERS)
            frozen_trainer.load(args.frozen_strategy)
        else:
            print(f"Warning: {args.frozen_strategy} not found. Proceeding without freezing.")

    trainer = CFRTrainer(
        GameModule(),
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

    trainer.train(target_iterations=args.iterations)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    trainer.save(args.out)
    print(f"Training complete. Final iteration: {trainer.iteration}. Strategy saved to {args.out}")


if __name__ == "__main__":
    main()
