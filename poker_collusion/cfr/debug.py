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

Step-through (--step): pauses only when the traverser is about to choose an action
(on_traverser_node). After you press Enter, CFR may recurse through many opponent
and chance outcomes (shown only in the recap) before the next traverser pause.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from poker_collusion.config import NUM_PLAYERS
from poker_collusion.terminal_display import (
    RESET,
    BOLD,
    DIM,
    RED,
    GREEN,
    YELLOW,
    CYAN,
    MAGENTA,
    strip_ansi,
    format_action_label,
    format_bar,
    print_nlhe_state,
    print_state_history,
    print_blank,
    print_field,
    print_heading_centered,
    print_rule_line,
    STREET_NAMES,
)

# ── Layout (CFR recap uses wider rules than the 20-col table) ─────────────────
_SEATS = ["BTN", "SB ", "BB "]
_RECAP_W = 78
_W = 20
_LIGHT = "─" * _W


def _strip(s: str) -> str:
    return strip_ansi(s)


def _heading(label, color=MAGENTA):
    print_heading_centered(label, color=color, width=_W)


def _rule(char="━"):
    if len(char) > 1:
        print(char)
    else:
        print(char * _W)


def _field(key, value, indent=1):
    print_field(key, value, indent=indent)


def _blank():
    print_blank()


# ══════════════════════════════════════════════════════════════════════════════
class CFRDebugger:
    """
    Observes CFR traversal and pretty-prints each node.
    Optionally pauses at traverser decision nodes for interactive step-through.
    """

    def __init__(
        self, enabled: bool = True, step: bool = True, consolidate: bool = True
    ) -> None:
        self.enabled = enabled
        self._step = step
        self.consolidate = consolidate
        self._buf_seq = 0
        self._buf_path: List[Dict[str, Any]] = []
        self._buf_result: List[Dict[str, Any]] = []
        self._recap_iter = 0
        self._recap_traverser = ""

    def begin_traversal(self, traverser: int, iteration: int) -> None:
        """Start of one cfr_traverse (depth-0). Buffers recap rows if consolidate."""
        if not self.enabled or not self.consolidate:
            return
        self._buf_seq = 0
        self._buf_path = []
        self._buf_result = []
        self._recap_iter = iteration
        self._recap_traverser = f"P{traverser} {_SEATS[traverser].strip()}"

    def end_traversal(self) -> None:
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

    def _append_path(self, kind: str, **kw: Any) -> None:
        self._buf_seq += 1
        self._buf_path.append({"kind": kind, "seq": self._buf_seq, **kw})

    def _append_result(self, **kw: Any) -> None:
        self._buf_seq += 1
        self._buf_result.append({"seq": self._buf_seq, **kw})

    def _flush_traversal_recap(self) -> None:
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
                ri = row.get("round_idx_at_node", st.round_idx)
                street = STREET_NAMES[ri] if 0 <= ri < len(STREET_NAMES) else f"r{ri}"
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
                    round_idx_override=ri,
                )
        print(f"\n{CYAN}{'═' * _RECAP_W}{RESET}\n")

    # ── Public hooks (called by CFRTrainer) ────────────────────────────────────

    def on_traverser_node(
        self,
        state: Any,
        player: int,
        traverser: int,
        depth: int,
        iteration: int,
        actions: Sequence[int],
        strategy: Union[np.ndarray, Sequence[float]],
        info_key: Any,
    ) -> None:
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

    def on_traverser_result(
        self,
        state: Any,
        player: int,
        depth: int,
        iteration: int,
        actions: Sequence[int],
        values: Union[np.ndarray, Sequence[float]],
        ev: float,
        regret_update: Union[np.ndarray, Sequence[float]],
        weight: float,
    ) -> None:
        if not self.enabled:
            return
        if self.consolidate:
            self._append_result(
                state=state,
                round_idx_at_node=int(state.round_idx),
                player=player,
                depth=depth,
                actions=list(actions),
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
        self._print_result_table(
            state, actions, values, ev, regret_update, weight, compact=False
        )
        _rule(_LIGHT)

    def on_opponent_node(
        self,
        state: Any,
        player: int,
        traverser: int,
        depth: int,
        iteration: int,
        actions: Sequence[int],
        strategy: Union[np.ndarray, Sequence[float]],
        sampled_idx: int,
    ) -> None:
        if not self.enabled:
            return
        ri = state.round_idx
        sampled = actions[sampled_idx]
        label = format_action_label(sampled, ri)
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

    def on_chance_node(
        self, state: Any, traverser: int, depth: int, iteration: int
    ) -> None:
        if not self.enabled:
            return
        streets = {0: "Flop", 1: "Turn", 2: "River"}
        street = streets.get(state.round_idx, "Street")
        if self.consolidate:
            self._append_path("chance", depth=depth, street=street)
            return
        indent = "  " * min(depth, 10)
        print(f"{indent}{DIM}[|hist|={depth}] CHANCE → dealing {street}{RESET}")

    def on_terminal(
        self,
        state: Any,
        traverser: int,
        depth: int,
        iteration: int,
        payoffs: Sequence[float],
    ) -> None:
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

    def _print_state(self, state: Any, acting_player: int) -> None:
        print_nlhe_state(
            state, acting_player, hole_visible=[True] * NUM_PLAYERS
        )
        print_state_history(state)

    def _print_strategy(
        self,
        state: Any,
        actions: Sequence[int],
        strategy: Union[np.ndarray, Sequence[float]],
    ) -> None:
        ri = state.round_idx
        print(f"  {BOLD}Actions & current strategy:{RESET}")
        for a, prob in zip(actions, strategy):
            label = format_action_label(a, ri)
            bar = format_bar(prob)
            pct = f"{prob * 100:5.1f}%"
            # right-align label to 16 chars (visible)
            vis_label = label.ljust(16)
            print(f"    [{a}] {vis_label}  {bar}  {pct}")

    def _print_result_table(
        self,
        state: Any,
        actions: Sequence[int],
        values: Union[np.ndarray, Sequence[float]],
        ev: float,
        regret_update: Union[np.ndarray, Sequence[float]],
        weight: float,
        compact: bool = False,
        round_idx_override: Optional[int] = None,
    ) -> None:
        ri = (
            round_idx_override
            if round_idx_override is not None
            else state.round_idx
        )
        header = f"  {'Action':<16}  {'Value':>8}  {'Regret':>8}  {'Δ regret_sum':>12}"
        print(header)
        print("  " + "─" * (len(_strip(header)) - 2))
        for i, a in enumerate(actions):
            label = format_action_label(a, ri).ljust(16)
            val   = f"{values[i]:>+8.3f}"
            reg   = f"{regret_update[i]:>+8.3f}"
            delta = regret_update[i] * weight
            col   = GREEN if delta > 0 else RED if delta < 0 else DIM
            upd   = f"{col}{delta:>+12.1f}{RESET}"
            print(f"  {label}  {val}  {reg}  {upd}")
        if not compact:
            print(f"\n  {BOLD}EV = {ev:+.3f} BB{RESET}")

    # ── Interactive pause ──────────────────────────────────────────────────────

    def _wait(self) -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
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
