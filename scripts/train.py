"""
Updated train script with --shared-info support.
"""
import argparse
from poker_collusion.cfr import CFRTrainer
from poker_collusion.env import (
    deal_new_hand, get_current_player, get_legal_actions,
    get_info_key, is_terminal, get_payoffs, apply_action,
    is_chance_node, sample_chance
)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", "-n", type=int, default=1000)
    parser.add_argument("--out", "-o", default="output/blueprint.pkl")
    parser.add_argument("--team-seats", type=str, default=None)
    parser.add_argument("--shared-info", action="store_true", help="Enable Shared Information (Psychic) variant")
    parser.add_argument("--team-objective", default="utilitarian")
    args = parser.parse_args()

    team_seats = [int(s) for s in args.team_seats.split(",")] if args.team_seats else None

    trainer = CFRTrainer(
        GameModule(),
        use_shared_info=args.shared_info, # NEW
        team_seats=team_seats,
        team_objective=args.team_objective
    )

    trainer.train(num_iterations=args.iterations)
    trainer.save(args.out)

if __name__ == "__main__":
    main()