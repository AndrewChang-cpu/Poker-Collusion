"""
Updated train script with expanded GameModule and --frozen-strategy support.
Fixes Bug 3: Standardized Policy Interfaces (Step 3).
"""
import argparse
import os
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
    parser.add_argument("--iterations", "-n", type=int, default=1000)
    parser.add_argument("--out", "-o", default="output/blueprint.pkl")
    parser.add_argument("--team-seats", type=str, default=None)
    parser.add_argument("--shared-info", action="store_true", help="Enable Shared Information (Psychic) variant")
    parser.add_argument("--team-objective", default="utilitarian")
    parser.add_argument("--frozen-strategy", type=str, default=None, 
                        help="Path to a baseline strategy file (.pkl) to freeze opponent seats")
    args = parser.parse_args()

    team_seats = [int(s) for s in args.team_seats.split(",")] if args.team_seats else None

    # Handle frozen strategy loading
    frozen_trainer = None
    if args.frozen_strategy:
        if os.path.exists(args.frozen_strategy):
            print(f"Loading frozen strategy from {args.frozen_strategy}...")
            # Initialize a trainer and load the pre-trained data
            frozen_trainer = CFRTrainer(GameModule(), num_players=3)
            frozen_trainer.load(args.frozen_strategy)
        else:
            print(f"Warning: Frozen strategy file {args.frozen_strategy} not found. Proceeding without freezing.")

    trainer = CFRTrainer(
        GameModule(),
        use_shared_info=args.shared_info,
        team_seats=team_seats,
        frozen_trainer=frozen_trainer,
        team_objective=args.team_objective
    )

    trainer.train(num_iterations=args.iterations)
    
    # Ensure output path exists
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    trainer.save(args.out)
    print(f"Training complete. Strategy saved to {args.out}")

if __name__ == "__main__":
    main()