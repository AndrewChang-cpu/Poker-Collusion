"""
Updated train script with support for resuming training from checkpoints.
Step 2 & 3: Added --resume-from argument and integrated load logic.
"""
import argparse
import os
import sys
import pickle
from poker_collusion.cfr import CFRTrainer
from poker_collusion.env import (
    deal_new_hand, get_current_player, get_legal_actions,
    get_info_key, is_terminal, get_payoffs, apply_action,
    is_chance_node, sample_chance, evaluate_hand
)
from poker_collusion.env.game_logic import _resolve_side_pots

class GameModule:
    """
    Comprehensive interface for the CFRTrainer to interact with the game environment.
    Exposes both core CFR methods and auxiliary environment logic.
    """
    deal_new_hand = staticmethod(deal_new_hand)
    get_current_player = staticmethod(get_current_player)
    get_legal_actions = staticmethod(get_legal_actions)
    get_info_key = staticmethod(get_info_key)
    is_terminal = staticmethod(is_terminal)
    get_payoffs = staticmethod(get_payoffs)
    apply_action = staticmethod(apply_action)
    is_chance_node = staticmethod(is_chance_node)
    sample_chance = staticmethod(sample_chance)
    
    # Auxiliary environment methods
    evaluate_hand = staticmethod(evaluate_hand)
    _resolve_side_pots = staticmethod(_resolve_side_pots)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", "-n", type=int, default=1000, 
                        help="Target total number of iterations (e.g. 200000)")
    parser.add_argument("--out", "-o", default="output/blueprint.pkl")
    parser.add_argument("--team-seats", type=str, default=None)
    parser.add_argument("--shared-info", action="store_true", help="Enable Shared Information (Psychic) variant")
    parser.add_argument("--team-objective", default="utilitarian")
    parser.add_argument("--frozen-strategy", type=str, default=None, 
                        help="Path to a baseline strategy file (.pkl) to freeze opponent seats")
    
    # Step 2: Resumption Argument
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to an existing strategy file (.pkl) to resume training")
    
    args = parser.parse_args()

    team_seats = [int(s) for s in args.team_seats.split(",")] if args.team_seats else None

    # Handle frozen strategy loading
    frozen_trainer = None
    if args.frozen_strategy:
        if os.path.exists(args.frozen_strategy):
            print(f"Loading frozen strategy from {args.frozen_strategy}...")
            frozen_trainer = CFRTrainer(GameModule(), num_players=3)
            frozen_trainer.load(args.frozen_strategy)
        else:
            print(f"Warning: Frozen strategy file {args.frozen_strategy} not found.")

    # Initialize trainer
    trainer = CFRTrainer(
        GameModule(),
        use_shared_info=args.shared_info,
        team_seats=team_seats,
        frozen_trainer=frozen_trainer,
        team_objective=args.team_objective
    )

    # Step 3: Integrate Load Logic
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"Resuming training from {args.resume_from}...")
            trainer.load(args.resume_from)
        else:
            print(f"Error: Checkpoint file {args.resume_from} not found.")
            sys.exit(1)

    # Begin training up to target iterations
    trainer.train(target_iterations=args.iterations)
    
    # Ensure output path exists
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    trainer.save(args.out)
    print(f"Training complete. Final iteration: {trainer.iteration}. Strategy saved to {args.out}")

if __name__ == "__main__":
    main()