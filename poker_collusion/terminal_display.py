"""
Shared terminal formatting for CFR debug and interactive playtest.
Matches the visual language of poker_collusion.cfr.debug.CFRDebugger.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

from poker_collusion.config import NUM_PLAYERS
from poker_collusion.abstraction.actions import PREFLOP_RAISE_BB, POSTFLOP_POT_MULT
from poker_collusion.env.game_state import DEAL, reconstruct_actor_history

# ── ANSI ──────────────────────────────────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"

_SEATS = ["BTN", "SB ", "BB "]
STREET_NAMES = ("Preflop", "Flop", "Turn", "River")
_W = 20

_RANKS = "23456789TJQKA"
_SUITS = "♠♥♦♣"
_SUIT_COLOR = [CYAN, RED, RED, CYAN]


def strip_ansi(s: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", s)


def format_card(card_idx: int) -> str:
    r = _RANKS[card_idx % 13]
    s = _SUITS[card_idx // 13]
    col = _SUIT_COLOR[card_idx // 13]
    return f"{col}{BOLD}{r}{s}{RESET}"


def format_hand(cards: Sequence[int]) -> str:
    return "  ".join(format_card(c) for c in cards) if cards else f"{DIM}--{RESET}"


def format_action_label(idx: int, round_idx: int) -> str:
    if idx == 0:
        return "Fold"
    if idx == 1:
        return "Check/Call"
    if idx == 9:
        return "All-in"
    if round_idx == 0:
        bb = PREFLOP_RAISE_BB[idx - 2]
        return f"Raise {bb:g}BB"
    pct = int(POSTFLOP_POT_MULT[idx - 2] * 100)
    return f"Bet {pct}%pot"


def format_history_from_state(state) -> str:
    """Full action history string for a live NLHEState (uses undo_stack to recover actors)."""
    return format_history_line(state.action_history, reconstruct_actor_history(state))


def format_history_line(action_history: Sequence, actor_history: Sequence[int]) -> str:
    """Render action_history using actor_history for player labels (non-DEAL actions only)."""
    assert len(actor_history) == sum(1 for a in action_history if a != DEAL), (
        f"actor_history length {len(actor_history)} does not match "
        f"non-DEAL entries in action_history ({sum(1 for a in action_history if a != DEAL)})"
    )
    ri = 0
    parts = []
    actor_idx = 0
    street_names = {1: "Flop", 2: "Turn", 3: "River"}
    for a in action_history:
        if a == DEAL:
            ri += 1
            name = street_names.get(ri, f"Street{ri}")
            parts.append(f"{DIM}[{name}]{RESET}")
        else:
            label = format_action_label(a, ri)
            who = f"P{actor_history[actor_idx]}"
            actor_idx += 1
            parts.append(f"{DIM}{a}{RESET}({who} {label})")
    return " → ".join(parts) if parts else f"{DIM}(preflop start){RESET}"


def format_bar(prob: float) -> str:
    bar_w = 14
    filled = round(prob * bar_w)
    return f"{GREEN}{'█' * filled}{DIM}{'░' * (bar_w - filled)}{RESET}"


def print_heading_centered(label: str, color: str = MAGENTA, width: int = _W) -> None:
    pad = (width - len(strip_ansi(label))) // 2
    print(f"{' ' * pad}{color}{BOLD}{label}{RESET}")


def print_rule_line(char: str = "━", width: int = _W) -> None:
    print(char * width)


def print_field(key: str, value: str, indent: int = 1) -> None:
    prefix = "  " * indent
    print(f"{prefix}{DIM}{key}:{RESET}  {value}")


def print_blank() -> None:
    print()


def print_nlhe_state(
    state,
    acting_player: int,
    *,
    hole_visible: Optional[Sequence[bool]] = None,
    title: Optional[str] = None,
) -> None:
    """
    Pretty-print table state (same layout as CFRDebugger._print_state).
    hole_visible[i]: if False, show hidden hole cards for seat i (playtest vs bots).
    """
    if hole_visible is None:
        hole_visible = [True] * NUM_PLAYERS
    ri = state.round_idx

    if title:
        print_blank()
        print_rule_line()
        print_heading_centered(title)
        print_rule_line()

    board_str = format_hand(state.board) if state.board else f"{DIM}(no board yet){RESET}"
    print_field("Board", board_str)
    print_blank()

    for p in range(NUM_PLAYERS):
        if hole_visible[p]:
            cards = format_hand(state.hole_cards[p])
        else:
            cards = f"{DIM}??  ??{RESET}"
        stack = f"{state.stacks[p]:.1f} BB"
        if not state.active[p]:
            status = f"  {RED}FOLDED{RESET}"
        elif state.all_in[p]:
            status = f"  {YELLOW}ALL-IN{RESET}"
        else:
            status = ""
        marker = f"  {BOLD}← acting{RESET}" if p == acting_player else ""
        print(f"  P{p} {_SEATS[p]}  {cards}   {stack}{status}{marker}")

    print_blank()
    bets = "   ".join(f"P{i}={state.bets[i]:.1f}" for i in range(NUM_PLAYERS))
    print_field("Pot", f"{BOLD}{state.pot:.2f} BB{RESET}   street bets: {bets}")


def print_state_history(state) -> None:
    """History line matching CFR debug (call after print_nlhe_state)."""
    print_field("History", format_history_from_state(state))
    print_blank()


def print_playtest_banner(
    street_name: str, board_cards: Sequence[int], width: int = 50
) -> None:
    """Section header after board cards are dealt (matches playtest / debug vibe)."""
    b = format_hand(board_cards) if board_cards else f"{DIM}(no board yet){RESET}"
    print(f"\n{CYAN}{'═' * width}{RESET}")
    print(f"  {BOLD}{street_name}{RESET}  {b}")
    print(f"{CYAN}{'═' * width}{RESET}")


def print_showdown_footer() -> None:
    print("\n" + f"{MAGENTA}{'#' * 55}{RESET}")
    print(f"  {BOLD}SHOWDOWN RESULTS{RESET}")
    print(f"{MAGENTA}{'#' * 55}{RESET}")


def print_playtest_showdown_recap(state) -> None:
    """
    Print final board and full action history. Call right after print_showdown_footer.

    When the engine uses _run_out_board_and_resolve (e.g. only one player left who can
    still bet while others are all-in), streets are not stepped via sample_chance, so the
    interactive loop never printed flop/turn/river. This recap makes that visible.
    """
    print_blank()
    print(f"  {BOLD}Hand recap{RESET}")
    if not state.board:
        print_field(
            "Final board",
            f"{DIM}(none — hand ended before community cards){RESET}",
        )
    else:
        print_field("Final board", format_hand(state.board))
        if len(state.board) >= 3:
            print(f"    {DIM}Flop:{RESET}  {format_hand(state.board[:3])}")
        if len(state.board) >= 4:
            print(f"    {DIM}Turn:{RESET}  {format_hand(state.board[3:4])}")
        if len(state.board) >= 5:
            print(f"    {DIM}River:{RESET} {format_hand(state.board[4:5])}")
    print_blank()
    print_field("Full action history", format_history_from_state(state))
    if state.board and DEAL not in state.action_history:
        print(
            f"    {DIM}Note: the engine ran out the board in one step (typical when only one"
            f" player can still act). Street banners only appear when play uses chance nodes.{RESET}"
        )
    print_blank()
