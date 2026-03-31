#!/usr/bin/env python3
"""
Interactive playtest: play against two bots using a trained blueprint.
Uses the same terminal layout as CFR debug (poker_collusion.terminal_display).
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
from poker_collusion.terminal_display import (
    RESET,
    MAGENTA,
    format_hand,
    print_nlhe_state,
    print_state_history,
    print_heading_centered,
    print_rule_line,
    print_showdown_footer,
    print_playtest_banner,
    print_playtest_showdown_recap,
    STREET_NAMES,
)


def get_human_action(state, dry_run=False):
    legal_actions = get_legal_actions(state)
    if dry_run:
        action = 1 if 1 in legal_actions else 0
        print(
            f"\n[DRY-RUN] Human chooses: {get_action_description(state, action)}"
        )
        return action

    print("\nActions:")
    for idx in legal_actions:
        print(f"  [{idx}] {get_action_description(state, idx)}")

    while True:
        try:
            choice = int(input("\nChoice: "))
            if choice in legal_actions:
                return choice
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
            ri = min(state.round_idx, len(STREET_NAMES) - 1)
            print_playtest_banner(f"{STREET_NAMES[ri]} dealt", state.board)
            continue

        p = get_current_player(state)
        print()
        print_rule_line()
        print_heading_centered("YOUR TURN" if p == args.seat else f"BOT P{p}")
        print_rule_line()
        print_nlhe_state(
            state,
            p,
            hole_visible=[i == args.seat for i in range(NUM_PLAYERS)],
        )
        print_state_history(state)

        if p == args.seat:
            action = get_human_action(state, args.dry_run)
        else:
            actions = get_legal_actions(state)
            info_key = get_info_key(state, p)
            action = strategy.sample_action(info_key, actions)
            print(
                f"\n  Bot P{p} chooses: {get_action_description(state, action)}"
            )

        apply_action(state, action)
        actions_taken += 1

    print_showdown_footer()
    print_playtest_showdown_recap(state)
    payoffs = get_payoffs(state)
    for i in range(NUM_PLAYERS):
        cards = format_hand(state.hole_cards[i])
        role = "HUMAN" if i == args.seat else "BOT"
        if state.active[i]:
            score = evaluate_hand(state.hole_cards[i] + state.board)
            desc = get_hand_description(score)
            print(
                f"  P{i} ({role}): {payoffs[i]:>+5.1f} BB | {cards} | {desc}"
            )
        else:
            print(
                f"  P{i} ({role}): {payoffs[i]:>+5.1f} BB | {cards} | (Folded)"
            )
    print(f"{MAGENTA}{'#' * 55}{RESET}\n")


if __name__ == "__main__":
    main()
