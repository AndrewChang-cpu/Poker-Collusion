"""
MCCFR trainer: external sampling, Linear CFR, regret pruning.
Game module must provide: deal_new_hand, get_current_player, get_legal_actions,
get_info_key, is_terminal, get_payoffs, apply_action, undo_action, is_chance_node, sample_chance.
"""

from __future__ import annotations

import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from poker_collusion.cfr.strategy import regret_matching, get_average_strategy
from poker_collusion.cfr.debug import CFRDebugger, _history_str
from poker_collusion.config import (
    NUM_ACTIONS,
    PRUNE_THRESHOLD,
    PRUNE_WARM_UP_ITERATIONS,
    PRUNE_SKIP_PROBABILITY,
)
from poker_collusion.typing_defs import CFRGame

_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "logs",
)
_CFR_ERROR_LOG = os.path.join(_LOG_DIR, "cfr_error_traceback.log")


InfoKey = Any
StrategyTable = Dict[InfoKey, np.ndarray]
ActionMap = Dict[InfoKey, List[int]]


class CFRTrainer:
    def __init__(
        self,
        game_module: CFRGame,
        num_players: int = 3,
        use_linear_cfr: bool = True,
        prune_threshold: Optional[float] = PRUNE_THRESHOLD,
        prune_warm_up: int = PRUNE_WARM_UP_ITERATIONS,
        prune_skip_prob: float = PRUNE_SKIP_PROBABILITY,
        debug: bool = False,
        debug_step: bool = False,
        debug_consolidate: bool = True,
    ) -> None:
        self.game = game_module
        self.num_players = num_players
        self.use_linear_cfr = use_linear_cfr
        self.prune_threshold = prune_threshold
        self.prune_warm_up = prune_warm_up
        self.prune_skip_prob = prune_skip_prob
        self.debugger = CFRDebugger(
            enabled=debug,
            step=debug_step,
            consolidate=debug_consolidate,
        )

        self.regret_sum: StrategyTable = {}
        self.strategy_sum: StrategyTable = {}
        self.action_map: ActionMap = {}
        self.iteration: int = 0

    def get_strategy(self, info_key: InfoKey, legal_actions: List[int]) -> np.ndarray:
        """Return strategy distribution over legal_actions (length len(legal_actions))."""
        regrets_full = self.regret_sum.get(info_key, np.zeros(NUM_ACTIONS))
        if len(regrets_full) < NUM_ACTIONS:
            regrets_full = np.resize(regrets_full, NUM_ACTIONS)
        regrets_sub = np.array([regrets_full[a] for a in legal_actions])
        return regret_matching(regrets_sub, len(legal_actions))

    def get_average_strategy(
        self, info_key: InfoKey, legal_actions: Optional[List[int]] = None
    ) -> Optional[np.ndarray]:
        """If legal_actions given, return normalized dist over those; else full length-NUM_ACTIONS."""
        if info_key not in self.strategy_sum:
            return None
        s = self.strategy_sum[info_key]
        if len(s) < NUM_ACTIONS:
            s = np.resize(s, NUM_ACTIONS)
        if legal_actions is not None:
            s_sub = np.array([s[a] for a in legal_actions])
            return get_average_strategy(s_sub, len(legal_actions))
        return get_average_strategy(s, NUM_ACTIONS)

    def cfr_traverse(
        self, state: Any, traverser: int, depth: int = 0, depth_limit: int = 500
    ) -> float:
        # TODO (BUG-1, BUG-2): depth limit is a workaround for undo-stack corruption
        # and broken player-rotation in _is_round_complete. See TODO.md.
        print(f"!!! P{traverser}\tdepth: {depth}\tactions: {_history_str(state.action_history)}")
        if depth == 0:
            self.debugger.begin_traversal(traverser, self.iteration)
        if depth > depth_limit:
            return 0.0

        if self.game.is_terminal(state):
            payoffs = self.game.get_payoffs(state)
            self.debugger.on_terminal(state, traverser, depth, self.iteration, payoffs)
            return payoffs[traverser]

        if self.game.is_chance_node(state):
            self.debugger.on_chance_node(state, traverser, depth, self.iteration)
            self.game.sample_chance(state)
            val = self.cfr_traverse(state, traverser, len(state.action_history))
            # Undo DEAL so undo_stack matches caller's next undo_action (BUG-1).
            self.game.undo_action(state)
            return val

        player = self.game.get_current_player(state)
        actions = self.game.get_legal_actions(state)
        info_key = self.game.get_info_key(state, player)
        num_actions = len(actions)

        if num_actions == 0:
            return 0.0

        if info_key not in self.action_map:
            self.action_map[info_key] = list(actions)

        strategy = self.get_strategy(info_key, actions)

        if player == traverser:
            self.debugger.on_traverser_node(
                state, player, traverser, depth, self.iteration,
                actions, strategy, info_key,
            )

            values = np.zeros(num_actions)
            for i, action in enumerate(actions):
                if self._should_prune(info_key, action):
                    values[i] = 0.0
                    continue
                self.game.apply_action(state, action)
                values[i] = self.cfr_traverse(state, traverser, len(state.action_history))
                self.game.undo_action()

            ev = float(strategy @ values)
            regret_update = values - ev
            weight = self.iteration if self.use_linear_cfr else 1

            if info_key not in self.regret_sum:
                self.regret_sum[info_key] = np.zeros(NUM_ACTIONS)
            for i, a in enumerate(actions):
                self.regret_sum[info_key][a] += regret_update[i] * weight

            if info_key not in self.strategy_sum:
                self.strategy_sum[info_key] = np.zeros(NUM_ACTIONS)
            for i, a in enumerate(actions):
                self.strategy_sum[info_key][a] += strategy[i] * weight

            self.debugger.on_traverser_result(
                state, player, depth, self.iteration,
                actions, values, ev, regret_update, weight,
            )
            return ev

        else:
            action_idx = np.random.choice(num_actions, p=strategy)
            self.debugger.on_opponent_node(
                state, player, traverser, depth, self.iteration,
                actions, strategy, action_idx,
            )
            self.game.apply_action(state, actions[action_idx])
            val = self.cfr_traverse(state, traverser, len(state.action_history))
            self.game.undo_action()
            return val

    def _should_prune(self, info_key: InfoKey, action: int) -> bool:
        """action is the abstract action index (0..9)."""
        if self.prune_threshold is None or self.iteration <= self.prune_warm_up:
            return False
        regrets = self.regret_sum.get(info_key, np.zeros(NUM_ACTIONS))
        if action < len(regrets) and regrets[action] < self.prune_threshold:
            return np.random.random() < self.prune_skip_prob
        return False

    def train(
        self,
        num_iterations: int,
        log_interval: int = 1,
        checkpoint_interval: int = 0,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """
        Run num_iterations of MCCFR. If checkpoint_interval > 0 and checkpoint_path is set,
        save every checkpoint_interval iterations ({iter} in path is replaced with the number).
        """
        mode = "step-back"
        start = self.iteration
        end = start + num_iterations
        print(f"Starting MCCFR for {num_iterations} iterations (total {start} -> {end}) ({mode})...")

        try:
            for t in tqdm(range(start + 1, end + 1), "Training..."):
                self.iteration = t
                for traverser in range(self.num_players):
                    state = self.game.deal_new_hand()
                    self.cfr_traverse(state, traverser)

                if log_interval and t % log_interval == 0:
                    avg_regret = self._compute_avg_regret()
                    print(
                        f"  Iter {t}/{end} | "
                        f"Info sets: {len(self.regret_sum)} | "
                        f"Avg regret: {avg_regret:.7f}"
                    )

                if checkpoint_interval and checkpoint_path and t % checkpoint_interval == 0:
                    path = checkpoint_path.format(iter=t) if "{iter}" in checkpoint_path else checkpoint_path
                    self.save(path)

        except RecursionError:
            os.makedirs(_LOG_DIR, exist_ok=True)
            with open(_CFR_ERROR_LOG, "w") as f:
                f.write("RecursionError in CFR train()\n\n")
                f.write(traceback.format_exc())
            raise
        except Exception:
            os.makedirs(_LOG_DIR, exist_ok=True)
            with open(_CFR_ERROR_LOG, "w") as f:
                f.write("Exception in CFR train()\n\n")
                f.write(traceback.format_exc())
            raise

        print(f"Training complete. {len(self.regret_sum)} info sets.")

    def _compute_avg_regret(self) -> float:
        if not self.regret_sum or self.iteration == 0:
            return 0.0
        if self.use_linear_cfr:
            sum_weights = (self.iteration * (self.iteration + 1)) / 2
        else:
            sum_weights = self.iteration
        total_pos = sum(np.maximum(regrets, 0).mean() for regrets in self.regret_sum.values())
        return (total_pos / len(self.regret_sum)) / sum_weights

    def get_all_strategies(self) -> Dict[InfoKey, Tuple[List[int], np.ndarray]]:
        out: Dict[InfoKey, Tuple[List[int], np.ndarray]] = {}
        for info_key in self.strategy_sum:
            actions = self.action_map.get(info_key, list(range(NUM_ACTIONS)))
            avg = self.get_average_strategy(info_key, actions)
            if avg is not None:
                out[info_key] = (actions, avg)
        return out

    def save(self, path: str) -> None:
        import pickle
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "regret_sum": self.regret_sum,
                "strategy_sum": self.strategy_sum,
                "action_map": self.action_map,
                "iteration": self.iteration,
            }, f)
        print(f"\nSaved to {path}")

    def load(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.regret_sum = data["regret_sum"]
        self.strategy_sum = data["strategy_sum"]
        self.action_map = data["action_map"]
        self.iteration = data["iteration"]
        print(f"Loaded from {path} (iter {self.iteration})")
