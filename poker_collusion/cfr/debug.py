"""
Pretty debug observer for CFR traversal.

Attach to CFRTrainer via debug=True / debug_step=True:

    trainer = CFRTrainer(game, debug=True, debug_step=True)

Controls at each traverser-node pause:
    Enter / n    step to next traverser node
    c            continue (disable stepping, run freely)
    q            quit debug output entirely

By default (consolidate=True), opponent/chance/terminal lines and RESULT tables are
buffered and printed once at the end of each full traversal as one recap chart.
Use consolidate=False (CLI: --debug-stream) for the legacy interleaved stream.
"""

import re
import sys

from poker_collusion.config import NUM_PLAYERS
from poker_collusion.abstraction.actions import PREFLOP_RAISE_BB, POSTFLOP_POT_MULT
from poker_collusion.env.game_state import DEAL

# ── ANSI ──────────────────────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"

def _strip(s):
    """Remove ANSI codes to get printable length."""
    return re.sub(r'\033\[[0-9;]*m', '', s)

# ── Card rendering ─────────────────────────────────────────────────────────────
_RANKS = "23456789TJQKA"
_SUITS = "♠♥♦♣"
_SUIT_COLOR = [CYAN, RED, RED, CYAN]   # ♠ ♥ ♦ ♣

def _card(c):
    r = _RANKS[c % 13]
    s = _SUITS[c // 13]
    col = _SUIT_COLOR[c // 13]
    return f"{col}{BOLD}{r}{s}{RESET}"

def _hand(cards):
    return "  ".join(_card(c) for c in cards) if cards else f"{DIM}--{RESET}"

# ── Action labels ──────────────────────────────────────────────────────────────
_SEATS = ["BTN", "SB ", "BB "]
_STREET = ("Preflop", "Flop", "Turn", "River")
_RECAP_W = 78

def _action_label(idx, round_idx):
    if idx == 0: return "Fold"
    if idx == 1: return "Check/Call"
    if idx == 9: return "All-in"
    if round_idx == 0:
        bb = PREFLOP_RAISE_BB[idx - 2]
        return f"Raise {bb:g}BB"
    pct = int(POSTFLOP_POT_MULT[idx - 2] * 100)
    return f"Bet {pct}%pot"


def _players_for_action_history(action_history):
    """Replay on a fresh hand to recover who acted for each action (turn order is deterministic)."""
    from poker_collusion.env.game_state import deal_new_hand
    from poker_collusion.env.game_logic import (
        apply_action,
        get_current_player,
        is_chance_node,
        is_terminal,
        sample_chance,
    )

    state = deal_new_hand()
    players = []
    for a in action_history:
        if a == DEAL:
            if is_chance_node(state):
                sample_chance(state)
            continue
        if is_terminal(state):
            break
        p = get_current_player(state)
        if p < 0:
            break
        players.append(p)
        apply_action(state, a)
        if is_terminal(state):
            break
    return players


def _history_str(action_history):
    """Render action_history as a readable string, tracking street changes."""
    try:
        players = _players_for_action_history(action_history)
    except Exception:
        players = None

    ri = 0
    parts = []
    pi = 0
    street_names = {1: "Flop", 2: "Turn", 3: "River"}
    for a in action_history:
        if a == DEAL:
            ri += 1
            name = street_names.get(ri, f"Street{ri}")
            parts.append(f"{DIM}[{name}]{RESET}")
        else:
            label = _action_label(a, ri)
            if players is not None and pi < len(players):
                who = f"P{players[pi]}"
                pi += 1
            else:
                who = "P?"
            parts.append(f"{DIM}{a}{RESET}({who} {label})")
    return " → ".join(parts) if parts else f"{DIM}(preflop start){RESET}"

# ── Progress bar ───────────────────────────────────────────────────────────────
_BAR_W = 14

def _bar(prob):
    filled = round(prob * _BAR_W)
    return f"{GREEN}{'█' * filled}{DIM}{'░' * (_BAR_W - filled)}{RESET}"

# ── Layout helpers ─────────────────────────────────────────────────────────────
_W = 20
_HEAVY = "━" * _W
_LIGHT = "─" * _W

def _print(line=""):
    print(line)

def _heading(label, color=MAGENTA):
    pad = (_W - len(_strip(label))) // 2
    print(f"{' ' * pad}{color}{BOLD}{label}{RESET}")

def _rule(char="━"):
    print(char * _W)

def _field(key, value, indent=1):
    prefix = "  " * indent
    print(f"{prefix}{DIM}{key}:{RESET}  {value}")

def _blank():
    print()


# ══════════════════════════════════════════════════════════════════════════════
class CFRDebugger:
    """
    Observes CFR traversal and pretty-prints each node.
    Optionally pauses at traverser decision nodes for interactive step-through.
    """

    def __init__(self, enabled=True, step=True, consolidate=True):
        self.enabled = enabled
        self._step = step
        self.consolidate = consolidate
        self._buf_seq = 0
        self._buf_path = []
        self._buf_result = []
        self._recap_iter = 0
        self._recap_traverser = ""

    def begin_traversal(self, traverser, iteration):
        """Start of one cfr_traverse (depth-0). Buffers recap rows if consolidate."""
        if not self.enabled or not self.consolidate:
            return
        self._buf_seq = 0
        self._buf_path = []
        self._buf_result = []
        self._recap_iter = iteration
        self._recap_traverser = f"P{traverser} {_SEATS[traverser].strip()}"

    def end_traversal(self):
        """End of cfr_traverse: print one consolidated chart when buffering."""
        if not self.enabled or not self.consolidate:
            return
        if not self._buf_path and not self._buf_result:
            self._buf_path = []
            self._buf_result = []
            return
        self._flush_traversal_recap()
        self._buf_path = []
        self._buf_result = []

    def _append_path(self, kind, **kw):
        self._buf_seq += 1
        self._buf_path.append({"kind": kind, "seq": self._buf_seq, **kw})

    def _append_result(self, **kw):
        self._buf_seq += 1
        self._buf_result.append({"seq": self._buf_seq, **kw})

    def _flush_traversal_recap(self):
        print()
        print(f"{CYAN}{'═' * _RECAP_W}{RESET}")
        print(
            f"  {BOLD}Traversal recap{RESET}  "
            f"{DIM}iter{RESET} {self._recap_iter}  "
            f"{DIM}traverser{RESET} {self._recap_traverser}"
        )
        print(f"{CYAN}{'═' * _RECAP_W}{RESET}")

        if self._buf_path:
            print(f"\n  {BOLD}Sampled path{RESET}  {DIM}(chance / opponent / terminal){RESET}")
            print(f"  {DIM}{'─' * (_RECAP_W - 2)}{RESET}")
            last_term_key = None
            for row in sorted(self._buf_path, key=lambda r: r["seq"]):
                if row["kind"] == "chance":
                    d = row["depth"]
                    street = row["street"]
                    print(f"  {DIM}[|hist|={d}]{RESET} CHANCE → dealing {street}")
                elif row["kind"] == "opp":
                    d = row["depth"]
                    pl = row["player"]
                    label = row["label"]
                    prob = row["prob"]
                    print(
                        f"  {DIM}[|hist|={d}]{RESET} OPPONENT P{pl} {_SEATS[pl]} "
                        f"→ {YELLOW}{label}{RESET}{DIM} ({prob * 100:.1f}%){RESET}"
                    )
                elif row["kind"] == "term":
                    d = row["depth"]
                    key = (d, tuple(row["payoffs"]))
                    if key == last_term_key:
                        continue
                    last_term_key = key
                    parts = []
                    for i in range(NUM_PLAYERS):
                        po = row["payoffs"][i]
                        col = GREEN if po > 0 else RED if po < 0 else DIM
                        parts.append(f"P{i} {col}{po:+.2f}{RESET}")
                    payoff_str = "   ".join(parts)
                    print(f"  {DIM}[|hist|={d}]{RESET} TERMINAL   {payoff_str}")

        if self._buf_result:
            print(f"\n  {BOLD}Traverser regret updates{RESET}  {DIM}(shallow → deep){RESET}")
            print(f"  {DIM}{'─' * (_RECAP_W - 2)}{RESET}")
            for row in sorted(self._buf_result, key=lambda r: r["depth"]):
                st = row["state"]
                depth = row["depth"]
                pl = row["player"]
                ri = st.round_idx
                street = _STREET[ri] if 0 <= ri < len(_STREET) else f"r{ri}"
                ev = row["ev"]
                w = row["weight"]
                print(
                    f"\n  {BOLD}|hist|={depth}{RESET}  {DIM}{street}{RESET}  "
                    f"P{pl}  {BOLD}EV = {ev:+.3f} BB{RESET}   {DIM}weight ×{w}{RESET}"
                )
                self._print_result_table(
                    st,
                    row["actions"],
                    row["values"],
                    ev,
                    row["regret_update"],
                    w,
                    compact=True,
                )
        print(f"\n{CYAN}{'═' * _RECAP_W}{RESET}\n")

    # ── Public hooks (called by CFRTrainer) ────────────────────────────────────

    def on_traverser_node(self, state, player, traverser, depth, iteration,
                          actions, strategy, info_key):
        if not self.enabled:
            return
        _blank()
        _rule()
        _heading(f"TRAVERSER NODE   depth {depth}   iter {iteration}   "
                 f"P{player} {_SEATS[player]}")
        _rule()
        self._print_state(state, player)
        self._print_strategy(state, actions, strategy)
        _rule(_LIGHT)
        _field("Info key", f"{DIM}{info_key}{RESET}")
        _rule()
        if self._step:
            self._wait()

    def on_traverser_result(self, state, player, depth, iteration,
                            actions, values, ev, regret_update, weight):
        if not self.enabled:
            return
        if self.consolidate:
            self._append_result(
                state=state,
                player=player,
                depth=depth,
                actions=actions,
                values=values,
                ev=ev,
                regret_update=regret_update,
                weight=weight,
            )
            return
        _blank()
        _rule(_LIGHT)
        _heading(f"RESULT   EV = {BOLD}{ev:+.3f} BB{RESET}   weight ×{weight}", color=CYAN)
        _rule(_LIGHT)
        self._print_result_table(state, actions, values, ev, regret_update, weight, compact=False)
        _rule(_LIGHT)

    def on_opponent_node(self, state, player, traverser, depth, iteration,
                         actions, strategy, sampled_idx):
        if not self.enabled:
            return
        ri = state.round_idx
        sampled = actions[sampled_idx]
        label = _action_label(sampled, ri)
        prob = strategy[sampled_idx]
        if self.consolidate:
            self._append_path(
                "opp",
                depth=depth,
                player=player,
                label=label,
                prob=prob,
            )
            return
        indent = "  " * min(depth, 10)
        print(f"{indent}{DIM}[|hist|={depth}] OPPONENT P{player} {_SEATS[player]} "
              f"→ {YELLOW}{label}{RESET}{DIM} ({prob * 100:.1f}%){RESET}")

    def on_chance_node(self, state, traverser, depth, iteration):
        if not self.enabled:
            return
        streets = {0: "Flop", 1: "Turn", 2: "River"}
        street = streets.get(state.round_idx, "Street")
        if self.consolidate:
            self._append_path("chance", depth=depth, street=street)
            return
        indent = "  " * min(depth, 10)
        print(f"{indent}{DIM}[|hist|={depth}] CHANCE → dealing {street}{RESET}")

    def on_terminal(self, state, traverser, depth, iteration, payoffs):
        if not self.enabled:
            return
        pay = list(payoffs)
        if self.consolidate:
            self._append_path("term", depth=depth, payoffs=pay)
            return
        parts = []
        for i in range(NUM_PLAYERS):
            col = GREEN if payoffs[i] > 0 else RED if payoffs[i] < 0 else DIM
            parts.append(f"P{i} {col}{payoffs[i]:+.2f}{RESET}")
        indent = "  " * min(depth, 10)
        payoff_str = "   ".join(parts)
        print(f"{indent}{DIM}[|hist|={depth}] TERMINAL   {payoff_str}{RESET}")

    # ── Display helpers ────────────────────────────────────────────────────────

    def _print_state(self, state, acting_player):
        ri = state.round_idx

        # Board
        board_str = _hand(state.board) if state.board else f"{DIM}(no board yet){RESET}"
        _field("Board", board_str)

        _blank()

        # Players
        for p in range(NUM_PLAYERS):
            cards = _hand(state.hole_cards[p])
            stack = f"{state.stacks[p]:.1f} BB"
            if not state.active[p]:
                status = f"  {RED}FOLDED{RESET}"
            elif state.all_in[p]:
                status = f"  {YELLOW}ALL-IN{RESET}"
            else:
                status = ""
            marker = f"  {BOLD}← acting{RESET}" if p == acting_player else ""
            print(f"  P{p} {_SEATS[p]}  {cards}   {stack}{status}{marker}")

        _blank()

        # Pot / bets
        bets = "   ".join(f"P{i}={state.bets[i]:.1f}" for i in range(NUM_PLAYERS))
        _field("Pot", f"{BOLD}{state.pot:.2f} BB{RESET}   street bets: {bets}")

        # History
        _field("History", _history_str(state.action_history))
        _blank()

    def _print_strategy(self, state, actions, strategy):
        ri = state.round_idx
        print(f"  {BOLD}Actions & current strategy:{RESET}")
        for a, prob in zip(actions, strategy):
            label = _action_label(a, ri)
            bar = _bar(prob)
            pct = f"{prob * 100:5.1f}%"
            # right-align label to 16 chars (visible)
            vis_label = label.ljust(16)
            print(f"    [{a}] {vis_label}  {bar}  {pct}")

    def _print_result_table(self, state, actions, values, ev, regret_update, weight, compact=False):
        ri = state.round_idx
        header = f"  {'Action':<16}  {'Value':>8}  {'Regret':>8}  {'Δ regret_sum':>12}"
        print(header)
        print("  " + "─" * (len(_strip(header)) - 2))
        for i, a in enumerate(actions):
            label = _action_label(a, ri).ljust(16)
            val   = f"{values[i]:>+8.3f}"
            reg   = f"{regret_update[i]:>+8.3f}"
            delta = regret_update[i] * weight
            col   = GREEN if delta > 0 else RED if delta < 0 else DIM
            upd   = f"{col}{delta:>+12.1f}{RESET}"
            print(f"  {label}  {val}  {reg}  {upd}")
        if not compact:
            print(f"\n  {BOLD}EV = {ev:+.3f} BB{RESET}")

    # ── Interactive pause ──────────────────────────────────────────────────────

    def _wait(self):
        try:
            raw = input(
                f"  {BOLD}▶{RESET} {DIM}[Enter/n] step   [c] continue   [q] quit debug{RESET}   "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            raw = "q"

        if raw in ("q", "quit"):
            self.enabled = False
            print(f"  {DIM}Debug output disabled.{RESET}")
        elif raw in ("c", "cont", "continue"):
            self._step = False
            print(f"  {DIM}Stepping disabled — running freely.{RESET}")
        # else: Enter / "n" / anything else → step
