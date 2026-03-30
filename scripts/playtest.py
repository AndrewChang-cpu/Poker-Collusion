#!/usr/bin/env python3
"""
Interactive playtest script: Play against two bots using a trained blueprint.
"""

import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from poker_collusion.env import (
    deal_new_hand,
    apply_action,
    sample_chance,
    is_terminal,
    is_chance_node,
    get_current_player,
    get_legal_actions,
    get_info_key,
    get_payoffs,
)
from poker_collusion.env.hand_eval import evaluate_hand, get_hand_description
from poker_collusion.abstraction.actions import get_action_description
from poker_collusion.cfr.strategy import Strategy
from poker_collusion.config import NUM_PLAYERS

def card_to_str(card_idx):
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suits = ['s', 'h', 'd', 'c']
    return f"{ranks[card_idx % 13]}{suits[card_idx // 13]}"

def print_game_state(state, human_seat):
    streets = ["Preflop", "Flop", "Turn", "River"]
    print(f"\n" + "="*50)
    print(f" {streets[state.round_idx].upper()} | BOARD: {[card_to_str(c) for c in state.board]}")
    print(f" POT: {state.pot:.1f} BB")
    for i in range(NUM_PLAYERS):
        role = "HUMAN" if i == human_seat else "BOT"
        cards = [card_to_str(c) for c in state.hole_cards[i]] if i == human_seat else ["??", "??"]
        print(f" P{i} ({role}): {state.stacks[i]:>4.1f} BB | Bets: {state.bets[i]:>4.1f} | {cards}")
    print("="*50)

def get_human_action(state, dry_run=False):
    legal_actions = get_legal_actions(state)
    if dry_run:
        action = 1 if 1 in legal_actions else 0
        print(f"\n[DRY-RUN] Human chooses: {get_action_description(state, action)}")
        return action

    print("\nActions:")
    for idx in legal_actions:
        print(f"  [{idx}] {get_action_description(state, idx)}")
    
    while True:
        try:
            choice = int(input("\nChoice: "))
            if choice in legal_actions: return choice
            print("Invalid index.")
        except ValueError:
            print("Enter a number.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blueprint", "-b", default="output/blueprint.pkl")
    parser.add_argument("--seat", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    strategy = Strategy.load(args.blueprint)
    state = deal_new_hand()
    MAX_ACTIONS = 100
    actions_taken = 0

    while not is_terminal(state) and actions_taken < MAX_ACTIONS:
        if is_chance_node(state):
            sample_chance(state)
            print(f"\n[DEAL] Board: {[card_to_str(c) for c in state.board]}")
            continue

        p = get_current_player(state)
        print_game_state(state, args.seat)

        if p == args.seat:
            action = get_human_action(state, args.dry_run)
        else:
            actions = get_legal_actions(state)
            info_key = get_info_key(state, p)
            action = strategy.sample_action(info_key, actions)
            print(f"\nBot P{p} chooses: {get_action_description(state, action)}")
        
        apply_action(state, action)
        actions_taken += 1

    print("\n" + "#"*55)
    print(" SHOWDOWN RESULTS")
    payoffs = get_payoffs(state)
    for i in range(NUM_PLAYERS):
        cards = [card_to_str(c) for c in state.hole_cards[i]]
        role = "HUMAN" if i == args.seat else "BOT"
        if state.active[i]:
            score = evaluate_hand(state.hole_cards[i] + state.board)
            desc = get_hand_description(score)
            print(f" P{i} ({role}): {payoffs[i]:>+5.1f} BB | {cards} | {desc}")
        else:
            print(f" P{i} ({role}): {payoffs[i]:>+5.1f} BB | {cards} | (Folded)")
    print("#"*55)

if __name__ == "__main__":
    main()