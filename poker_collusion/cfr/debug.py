"""
Pretty debug observer for CFR traversal.

Attach to CFRTrainer via debug=True / debug_step=True:

    trainer = CFRTrainer(game, debug=True, debug_step=True)

Controls at each pause (--step):
    Enter / n    proceed (label tells you what comes next)
    c            continue freely (disable stepping)
    q            quit debug output entirely

Step-through (--step): pauses twice per traverser node in DFS order:
  - Entry: full game state + strategy, before any branches are explored
  - Exit:  per-branch sampled paths + regret result, after all branches resolve

Use --debug-stream (consolidate=False) for the legacy interleaved stream.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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

_SEATS = ["BTN", "SB ", "BB "]
_W = 78


def _strip(s: str) -> str:
    return strip_ansi(s)


def _rule_heavy() -> None:
    print("█" * _W)


def _rule_light(label: str = "") -> None:
    if label:
        rest = _W - len(_strip(label)) - 4
        print(f"  {label}  " + "-" * max(rest, 4))
    else:
        print("  " + "-" * (_W - 2))


# ══════════════════════════════════════════════════════════════════════════════
class CFRDebugger:
    """
    Observes CFR traversal and pretty-prints each node.
    In step mode (consolidate=True): entry+exit pause per traverser node.
    In stream mode (consolidate=False): legacy immediate print.
    """

    def __init__(
        self, enabled: bool = True, step: bool = True, consolidate: bool = True
    ) -> None:
        self.enabled = enabled
        self._step = step
        self.consolidate = consolidate

        # Node counter (resets each traversal)
        self._node_counter: int = 0

        # Live DFS stack — one entry per open traverser node
        # Each: {node_id, depth, player, actions, strategy, round_idx,
        #        state_snapshot, pruned_set, branch_results, branch_events,
        #        resolved_ev}
        self._live_stack: List[Dict[str, Any]] = []

        # Branch tracking — pushed/popped by trainer around each action branch
        # Each: (action_index_in_list, action_abstract_idx, is_last)
        self._branch_stack: List[Tuple[int, int, bool]] = []

        # Current innermost open node id and branch action (for event routing)
        self._current_node_id: Optional[int] = None
        self._current_branch_action: Optional[int] = None

        self._recap_iter: int = 0
        self._recap_traverser: str = ""

    # ── Traversal lifecycle ───────────────────────────────────────────────────

    def begin_traversal(self, traverser: int, iteration: int) -> None:
        if not self.enabled or not self.consolidate:
            return
        self._node_counter = 0
        self._live_stack = []
        self._branch_stack = []
        self._current_node_id = None
        self._current_branch_action = None
        self._recap_iter = iteration
        self._recap_traverser = f"P{traverser} {_SEATS[traverser].strip()}"
        print()
        _rule_heavy()
        print(
            f"  {BOLD}TRAVERSAL{RESET}  "
            f"{DIM}iter {iteration}  traverser {self._recap_traverser}{RESET}"
        )
        _rule_heavy()

    def end_traversal(self) -> None:
        if not self.enabled or not self.consolidate:
            return
        n = self._node_counter
        print()
        _rule_heavy()
        print(
            f"  {DIM}END  {self._recap_traverser}  "
            f"{n} node{'s' if n != 1 else ''} visited{RESET}"
        )
        _rule_heavy()
        print()

    def push_branch(self, action_abstract_idx: int) -> None:
        """Called by trainer before recursing into each traverser action branch."""
        if not self.enabled or not self.consolidate:
            return
        if not self._live_stack:
            return
        top = self._live_stack[-1]
        actions = top["actions"]
        try:
            i = actions.index(action_abstract_idx)
        except ValueError:
            i = -1
        is_last = (i == len(actions) - 1)
        self._branch_stack.append((i, action_abstract_idx, is_last))
        self._current_branch_action = action_abstract_idx
        node_id = top["node_id"]
        if action_abstract_idx not in top["branch_events"]:
            top["branch_events"][action_abstract_idx] = []

    def pop_branch(self) -> None:
        """Called by trainer after returning from each traverser action branch."""
        if not self.enabled or not self.consolidate:
            return
        if self._branch_stack:
            self._branch_stack.pop()
        self._current_branch_action = (
            self._branch_stack[-1][1] if self._branch_stack else None
        )

    # ── Public hooks ─────────────────────────────────────────────────────────

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

        if not self.consolidate:
            # Legacy stream mode
            print()
            _rule_light(
                f"{MAGENTA}{BOLD}TRAVERSER NODE{RESET}  depth {depth}  "
                f"iter {iteration}  P{player} {_SEATS[player]}"
            )
            print_nlhe_state(state, player, hole_visible=[True] * NUM_PLAYERS)
            print_state_history(state)
            self._print_strategy_compact(state, actions, strategy)
            _rule_light(f"{DIM}Info key: {info_key}{RESET}")
            if self._step:
                self._wait()
            return

        self._node_counter += 1
        node_id = self._node_counter

        # Snapshot the state for display at exit
        from poker_collusion.terminal_display import format_history_line  # type: ignore
        from poker_collusion.env.game_state import reconstruct_actor_history  # type: ignore
        _actor_hist = reconstruct_actor_history(state)
        _hist_str = format_history_line(state.action_history, _actor_hist)
        state_snapshot = {
            "hole_cards": [list(h) for h in state.hole_cards],
            "board": list(state.board),
            "stacks": list(state.stacks),
            "bets": list(state.bets),
            "pot": float(state.pot),
            "active": list(state.active),
            "all_in": list(state.all_in),
            "history_str": _hist_str,
            "round_idx": int(state.round_idx),
        }

        self._live_stack.append({
            "node_id": node_id,
            "depth": depth,
            "player": player,
            "actions": list(actions),
            "strategy": list(strategy),
            "round_idx": int(state.round_idx),
            "info_key": info_key,
            "state_snapshot": state_snapshot,
            "pruned_set": set(),       # filled by mark_pruned (via branch skip)
            "branch_results": {},      # action -> {ev, payoffs_list} once resolved
            "branch_events": {},       # action -> list of path events
            "resolved_ev": None,
        })
        self._current_node_id = node_id

        ri_name = STREET_NAMES[depth] if False else (
            STREET_NAMES[state.round_idx]
            if 0 <= state.round_idx < len(STREET_NAMES) else f"r{state.round_idx}"
        )

        print()
        _rule_light(
            f"{MAGENTA}{BOLD}Node #{node_id}  ENTRY{RESET}  "
            f"depth {depth}  {DIM}{ri_name}{RESET}  "
            f"P{player} {_SEATS[player]}"
        )
        self._print_tree()
        self._print_state_compact(state_snapshot, player)
        self._print_strategy_compact_from_lists(
            list(actions), list(strategy), state.round_idx
        )

        if self._step:
            self._wait(self._entry_prompt(node_id))

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

        if not self.consolidate:
            print()
            _rule_light(
                f"{CYAN}{BOLD}RESULT{RESET}  EV={ev:+.3f} BB  weight×{weight}"
            )
            self._print_result_table(
                list(actions), list(values), ev, list(regret_update),
                weight, state.round_idx,
            )
            if self._live_stack:
                self._live_stack.pop()
            return

        top = self._live_stack.pop() if self._live_stack else {}
        node_id = top.get("node_id", 0)
        ri = top.get("round_idx", int(state.round_idx))
        snap = top.get("state_snapshot", {})
        top["resolved_ev"] = ev

        # Infer pruned actions: any action never pushed to branch_events was pruned
        branch_events = top.get("branch_events", {})
        pruned_set = set(a for a in top.get("actions", []) if a not in branch_events)
        top["pruned_set"] = pruned_set

        # Restore current_node_id to parent
        self._current_node_id = (
            self._live_stack[-1]["node_id"] if self._live_stack else None
        )
        # If we resolved inside a parent's branch, store result there
        if self._live_stack and self._current_branch_action is not None:
            parent = self._live_stack[-1]
            parent["branch_results"][self._current_branch_action] = {
                "node_id": node_id,
                "ev": ev,
            }

        ri_name = STREET_NAMES[ri] if 0 <= ri < len(STREET_NAMES) else f"r{ri}"

        print()
        _rule_light(
            f"{CYAN}{BOLD}Node #{node_id}  EXIT{RESET}  "
            f"depth {depth}  {DIM}{ri_name}{RESET}  "
            f"P{player} {_SEATS[player]}  "
            f"{BOLD}EV={ev:+.3f} BB{RESET}"
        )
        # Print tree with this node now marked resolved
        top["resolved_ev"] = ev
        self._live_stack.append(top)
        self._print_tree(resolving_node_id=node_id)
        self._live_stack.pop()

        self._print_state_compact(snap, player)
        self._print_branch_paths(
            list(actions), top.get("branch_events", {}),
            top.get("pruned_set", set()), ri,
        )
        print()
        self._print_result_table(
            list(actions), list(values), ev, list(regret_update), weight, ri,
        )

        if self._step:
            self._wait(self._exit_prompt(node_id))

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
        label = format_action_label(actions[sampled_idx], ri)
        prob = float(strategy[sampled_idx])

        if self.consolidate and self._current_node_id is not None:
            self._append_event({
                "kind": "opp", "player": player,
                "label": label, "prob": prob, "depth": depth,
            })
            return
        indent = "  " * min(depth, 8)
        print(
            f"{indent}{DIM}[{depth}] OPP P{player} {_SEATS[player]} "
            f"→ {YELLOW}{label}{RESET}{DIM} ({prob*100:.0f}%){RESET}"
        )

    def on_chance_node(
        self, state: Any, traverser: int, depth: int, iteration: int
    ) -> None:
        if not self.enabled:
            return
        streets = {0: "Flop", 1: "Turn", 2: "River"}
        street = streets.get(state.round_idx, "Street")
        if self.consolidate and self._current_node_id is not None:
            self._append_event({"kind": "chance", "street": street, "depth": depth})
            return
        indent = "  " * min(depth, 8)
        print(f"{indent}{DIM}[{depth}] CHANCE → {street}{RESET}")

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
        if self.consolidate and self._current_node_id is not None:
            self._append_event({"kind": "term", "payoffs": pay, "depth": depth})
            return
        parts = []
        for i in range(NUM_PLAYERS):
            col = GREEN if payoffs[i] > 0 else RED if payoffs[i] < 0 else DIM
            parts.append(f"P{i}{col}{payoffs[i]:+.1f}{RESET}")
        indent = "  " * min(depth, 8)
        print(f"{indent}{DIM}[{depth}] TERMINAL  {'  '.join(parts)}{RESET}")

    # ── Tree rendering ────────────────────────────────────────────────────────

    def _print_tree(self, resolving_node_id: Optional[int] = None) -> None:
        """Print the full DFS tree explored so far."""
        print(f"  {BOLD}Tree:{RESET}")
        if not self._live_stack:
            return
        # The live_stack contains open nodes outermost-first.
        # We render starting from the root (index 0).
        self._render_node(0, indent=4, resolving_node_id=resolving_node_id)

    def _render_node(
        self,
        stack_idx: int,
        indent: int,
        resolving_node_id: Optional[int],
    ) -> None:
        pad = " " * indent
        node = self._live_stack[stack_idx]
        node_id = node["node_id"]
        player = node["player"]
        ri = node["round_idx"]
        actions = node["actions"]
        pruned = node.get("pruned_set", set())
        branch_results = node.get("branch_results", {})

        is_resolving = (node_id == resolving_node_id)
        is_current = (stack_idx == len(self._live_stack) - 1 and not is_resolving)

        if is_current:
            marker = f"{BOLD}►{RESET} "
            label = f"{BOLD}#{node_id} [here]{RESET}"
        elif is_resolving:
            marker = f"{GREEN}✓{RESET} "
            label = f"{GREEN}#{node_id}{RESET}"
        elif stack_idx == 0 and len(self._live_stack) == 1:
            marker = "  "
            label = f"#{node_id} [root]"
        else:
            marker = "  "
            label = f"#{node_id}"

        print(
            f"{pad}{marker}{label}  "
            f"{DIM}P{player} {_SEATS[player]}{RESET}"
        )

        # Determine which action is currently being explored (child on stack)
        active_action: Optional[int] = None
        if stack_idx + 1 < len(self._live_stack):
            # The next node on the stack is a child — find which branch led there
            if self._branch_stack:
                active_action = self._branch_stack[
                    min(stack_idx, len(self._branch_stack) - 1)
                ][1]

        for a in actions:
            action_label = format_action_label(a, ri)
            action_pad = " " * (indent + 2)

            branch_events_map = node.get("branch_events", {})
            if a in pruned:
                print(f"{action_pad}[{a}]{DIM}{action_label} [PRUNED]{RESET}")
            elif a in branch_results:
                # This branch led to a child traverser node
                res = branch_results[a]
                child_ev = res["ev"]
                child_id = res["node_id"]
                ev_col = GREEN if child_ev >= 0 else RED
                print(f"{action_pad}[{a}]{action_label} →")
                child_stack_idx = next(
                    (i for i, n in enumerate(self._live_stack)
                     if n["node_id"] == child_id),
                    None,
                )
                if child_stack_idx is not None:
                    self._render_node(
                        child_stack_idx, indent + 4, resolving_node_id
                    )
                else:
                    print(
                        f"{' ' * (indent + 4)}"
                        f"{GREEN}✓{RESET} #{child_id}  "
                        f"{ev_col}EV={child_ev:+.3f} BB{RESET}"
                    )
            elif a in branch_events_map:
                # Explored branch that went straight to terminal (no child traverser node)
                print(f"{action_pad}[{a}]{GREEN}✓{RESET}{action_label}")
            elif a == active_action and stack_idx + 1 < len(self._live_stack):
                # Currently exploring this branch
                print(f"{action_pad}[{a}]{action_label} →")
                self._render_node(
                    stack_idx + 1, indent + 4, resolving_node_id
                )
            else:
                print(
                    f"{action_pad}[{a}]{DIM}{action_label} "
                    f"(pending exploration){RESET}"
                )

    # ── State display ─────────────────────────────────────────────────────────

    def _print_state_compact(
        self, snap: Dict[str, Any], acting_player: int
    ) -> None:
        ri = snap.get("round_idx", 0)
        board = snap.get("board", [])
        hole_cards = snap.get("hole_cards", [])
        stacks = snap.get("stacks", [])
        active = snap.get("active", [True] * NUM_PLAYERS)
        all_in = snap.get("all_in", [False] * NUM_PLAYERS)
        bets = snap.get("bets", [0.0] * NUM_PLAYERS)
        pot = snap.get("pot", 0.0)
        history_str = snap.get("history_str", "")

        # Board line
        if board:
            from poker_collusion.terminal_display import format_card as _card
            board_str = "  ".join(_card(c) for c in board)
        else:
            board_str = f"{DIM}(no board yet){RESET}"
        print(f"  {DIM}Board:{RESET} {board_str}")

        # Players line
        player_parts = []
        for p in range(NUM_PLAYERS):
            seat = _SEATS[p].strip()
            if hole_cards and p < len(hole_cards):
                from poker_collusion.terminal_display import format_card as _card
                cards = " ".join(_card(c) for c in hole_cards[p])
            else:
                cards = "?? ??"
            stack = f"{stacks[p]:.1f}BB" if p < len(stacks) else "?"
            status = ""
            if p < len(active) and not active[p]:
                status = f" {DIM}FOLD{RESET}"
            elif p < len(all_in) and all_in[p]:
                status = f" {YELLOW}AI{RESET}"
            acting = f" {BOLD}←{RESET}" if p == acting_player else ""
            player_parts.append(
                f"P{p}{seat}: {cards}  {DIM}{stack}{RESET}{status}{acting}"
            )
        print("  " + "   ".join(player_parts))

        # Pot + bets line
        bets_str = "  ".join(
            f"P{p}={bets[p]:.1f}" for p in range(NUM_PLAYERS)
            if p < len(bets)
        )
        print(f"  {DIM}Pot:{RESET} {BOLD}{pot:.2f} BB{RESET}   {DIM}street bets: {bets_str}{RESET}")

        print(f"  {DIM}History:{RESET} {history_str if history_str else DIM + '(preflop start)' + RESET}")

    # ── Strategy display ──────────────────────────────────────────────────────

    def _print_strategy_compact(
        self,
        state: Any,
        actions: Sequence[int],
        strategy: Union[np.ndarray, Sequence[float]],
    ) -> None:
        self._print_strategy_compact_from_lists(
            list(actions), list(strategy), state.round_idx
        )

    def _print_strategy_compact_from_lists(
        self, actions: List[int], strategy: List[float], ri: int
    ) -> None:
        parts = []
        for a, prob in zip(actions, strategy):
            label = format_action_label(a, ri)
            pct = f"{prob * 100:.0f}%"
            col = BOLD if prob > 0.5 else (DIM if prob < 0.05 else "")
            parts.append(f"[{a}]{col}{label}{RESET} {pct}")
        print("  " + "   ".join(parts))

    # ── Branch paths display ──────────────────────────────────────────────────

    def _print_branch_paths(
        self,
        actions: List[int],
        branch_events: Dict[int, List[Dict[str, Any]]],
        pruned_set: set,
        ri: int,
    ) -> None:
        print(f"  {BOLD}Branches:{RESET}")
        for a in actions:
            action_label = format_action_label(a, ri)
            if a in pruned_set:
                print(f"    [{a}]{DIM}{action_label}  [PRUNED]{RESET}")
                continue
            events = branch_events.get(a, [])
            if not events:
                print(f"    [{a}]{DIM}{action_label}  (no events){RESET}")
                continue
            parts: List[str] = []
            seen_terminals: set = set()
            for ev in events:
                kind = ev["kind"]
                if kind == "opp":
                    parts.append(
                        f"P{ev['player']}{DIM}{_SEATS[ev['player']]}{RESET}"
                        f"→{YELLOW}{ev['label']}{RESET}"
                        f"{DIM}({ev['prob']*100:.0f}%){RESET}"
                    )
                elif kind == "chance":
                    parts.append(f"{DIM}[{ev['street']}]{RESET}")
                elif kind == "term":
                    key = tuple(ev["payoffs"])
                    if key in seen_terminals:
                        continue
                    seen_terminals.add(key)
                    payoff_parts = []
                    for i in range(NUM_PLAYERS):
                        po = ev["payoffs"][i]
                        col = GREEN if po > 0 else RED if po < 0 else DIM
                        payoff_parts.append(f"P{i}{col}{po:+.1f}{RESET}")
                    parts.append(
                        f"TERMINAL[{'  '.join(payoff_parts)}]"
                    )
            path_str = "  →  ".join(parts) if parts else f"{DIM}(no path){RESET}"
            print(f"    [{a}]{YELLOW}{action_label}{RESET}:  {path_str}")

    # ── Result table ──────────────────────────────────────────────────────────

    def _print_result_table(
        self,
        actions: List[int],
        values: List[float],
        ev: float,
        regret_update: List[float],
        weight: float,
        ri: int,
    ) -> None:
        header = f"  {'Action':<16}  {'Value':>8}  {'Regret':>8}  {'Δ regret_sum':>12}"
        print(header)
        print("  " + "-" * (len(_strip(header)) - 2))
        for i, a in enumerate(actions):
            label = format_action_label(a, ri).ljust(16)
            val = f"{values[i]:>+8.3f}"
            reg = f"{regret_update[i]:>+8.3f}"
            delta = regret_update[i] * weight
            col = GREEN if delta > 0 else RED if delta < 0 else DIM
            upd = f"{col}{delta:>+12.1f}{RESET}"
            print(f"  {label}  {val}  {reg}  {upd}")

    # ── Event routing ─────────────────────────────────────────────────────────

    def _append_event(self, ev: Dict[str, Any]) -> None:
        """Route an event to the current innermost open node's branch bucket."""
        if not self._live_stack:
            return
        top = self._live_stack[-1]
        branch_a = self._current_branch_action
        if branch_a not in top["branch_events"]:
            top["branch_events"][branch_a] = []
        top["branch_events"][branch_a].append(ev)

    # ── Prompt helpers ────────────────────────────────────────────────────────

    def _entry_prompt(self, node_id: int) -> str:
        return (
            f"Node #{node_id}  "
            f"[Enter] step into first branch   [c] continue   [q] quit"
        )

    def _exit_prompt(self, node_id: int) -> str:
        if not self._live_stack:
            return (
                f"Node #{node_id}  "
                f"[Enter] end traversal   [c] continue   [q] quit"
            )
        if self._branch_stack:
            i, action_abstract_idx, is_last = self._branch_stack[-1]
            parent = self._live_stack[-1]
            parent_actions = parent["actions"]
            parent_id = parent["node_id"]
            if is_last:
                return (
                    f"Node #{node_id}  "
                    f"[Enter] return to parent (Node #{parent_id})   "
                    f"[c] continue   [q] quit"
                )
            else:
                next_i = i + 1
                if next_i < len(parent_actions):
                    next_label = format_action_label(
                        parent_actions[next_i], parent["round_idx"]
                    )
                    return (
                        f"Node #{node_id}  "
                        f"[Enter] next branch ({next_label})   "
                        f"[c] continue   [q] quit"
                    )
        return (
            f"Node #{node_id}  "
            f"[Enter] continue   [c] continue   [q] quit"
        )

    # ── Interactive pause ─────────────────────────────────────────────────────

    def _wait(self, prompt: Optional[str] = None) -> None:
        if prompt is None:
            prompt = "[Enter] continue   [c] continue freely   [q] quit"
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            raw = input(f"  {BOLD}▶{RESET} {DIM}{prompt}{RESET}   ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raw = "q"
        if raw in ("q", "quit"):
            self.enabled = False
            print(f"  {DIM}Debug output disabled.{RESET}")
        elif raw in ("c", "cont", "continue"):
            self._step = False
            print(f"  {DIM}Stepping disabled — running freely.{RESET}")
