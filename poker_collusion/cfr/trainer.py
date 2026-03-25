"""
MCCFR trainer: external sampling, Linear CFR, regret pruning.
Game module must provide: deal_new_hand, get_current_player, get_legal_actions,
get_info_key, is_terminal, get_payoffs, apply_action, is_chance_node, sample_chance.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from poker_collusion.cfr.strategy import regret_matching, get_average_strategy
from poker_collusion.cfr.debug import CFRDebugger
from poker_collusion.config import (
    NUM_ACTIONS,
    PRUNE_THRESHOLD,
    PRUNE_WARM_UP_ITERATIONS,
    PRUNE_SKIP_PROBABILITY,
    PARALLEL_WORKERS,
    PARALLEL_BATCH_SIZE,
)
from poker_collusion.typing_defs import CFRGame
from poker_collusion.env.game_state import NLHEState


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

    def cfr_traverse(self, state: NLHEState, traverser: int) -> float:
        """
        Recursively traverse the game tree from state for the given traverser.
        apply_action and sample_chance return new state copies — no undo needed.
        """
        if self.game.is_terminal(state):
            payoffs = self.game.get_payoffs(state)
            self.debugger.on_terminal(state, traverser, len(state.action_history), self.iteration, payoffs)
            return payoffs[traverser]

        if self.game.is_chance_node(state):
            self.debugger.on_chance_node(state, traverser, len(state.action_history), self.iteration)
            next_state = self.game.sample_chance(state)
            return self.cfr_traverse(next_state, traverser)

        player = self.game.get_current_player(state)
        actions = self.game.get_legal_actions(state)
        assert len(actions) > 0, (
            f"Non-terminal, non-chance node has no legal actions: "
            f"player={player}, done={state.done}, chance_pending={state.chance_pending}, "
            f"history={state.action_history}"
        )

        info_key = self.game.get_info_key(state, player)

        if info_key not in self.action_map:
            self.action_map[info_key] = list(actions)

        strategy = self.get_strategy(info_key, actions)

        if player == traverser:
            self.debugger.on_traverser_node(
                state, player, traverser, len(state.action_history), self.iteration,
                actions, strategy, info_key,
            )

            values = np.zeros(len(actions))
            for i, action in enumerate(actions):
                if self._should_prune(info_key, action):
                    values[i] = 0.0
                    continue
                next_state = self.game.apply_action(state, action)
                values[i] = self.cfr_traverse(next_state, traverser)

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
                state, player, len(state.action_history), self.iteration,
                actions, values, ev, regret_update, weight,
            )
            return ev

        else:
            action_idx = np.random.choice(len(actions), p=strategy)
            self.debugger.on_opponent_node(
                state, player, traverser, len(state.action_history), self.iteration,
                actions, strategy, action_idx,
            )
            next_state = self.game.apply_action(state, actions[action_idx])
            return self.cfr_traverse(next_state, traverser)

    def _should_prune(self, info_key: InfoKey, action: int) -> bool:
        """action is the abstract action index (0..9)."""
        if self.prune_threshold is None or self.iteration <= self.prune_warm_up:
            return False
        regrets = self.regret_sum.get(info_key, np.zeros(NUM_ACTIONS))
        if action < len(regrets) and regrets[action] < self.prune_threshold:
            return np.random.random() < self.prune_skip_prob
        return False

    # ── Parallel training helpers ──────────────────────────────────────────────

    def _should_prune_local(
        self, info_key: InfoKey, action: int, rng: np.random.Generator
    ) -> bool:
        """Thread-safe variant of _should_prune using a caller-supplied rng."""
        if self.prune_threshold is None or self.iteration <= self.prune_warm_up:
            return False
        regrets = self.regret_sum.get(info_key, np.zeros(NUM_ACTIONS))
        if action < len(regrets) and regrets[action] < self.prune_threshold:
            return bool(rng.random() < self.prune_skip_prob)
        return False

    def _cfr_traverse_local(
        self,
        state: NLHEState,
        traverser: int,
        weight: float,
        rng: np.random.Generator,
        delta_r: StrategyTable,
        delta_s: StrategyTable,
        delta_am: ActionMap,
    ) -> float:
        """
        Thread-safe CFR traversal. Reads self.regret_sum (read-only during batch phase).
        All writes go into caller-supplied delta_r / delta_s / delta_am dicts.
        Uses rng (thread-local Generator) instead of global np.random.
        Debug hooks are absent — incompatible with parallel mode.
        """
        if self.game.is_terminal(state):
            return self.game.get_payoffs(state)[traverser]

        if self.game.is_chance_node(state):
            next_state = self.game.sample_chance(state)
            return self._cfr_traverse_local(next_state, traverser, weight, rng, delta_r, delta_s, delta_am)

        player = self.game.get_current_player(state)
        actions = self.game.get_legal_actions(state)
        assert len(actions) > 0, (
            f"Non-terminal, non-chance node has no legal actions: "
            f"player={player}, done={state.done}, chance_pending={state.chance_pending}, "
            f"history={state.action_history}"
        )

        info_key = self.game.get_info_key(state, player)

        # Register new info keys locally; merged into self.action_map after the batch
        if info_key not in self.action_map and info_key not in delta_am:
            delta_am[info_key] = list(actions)

        strategy = self.get_strategy(info_key, actions)  # reads self.regret_sum — safe (read-only)

        if player == traverser:
            values = np.zeros(len(actions))
            for i, action in enumerate(actions):
                if self._should_prune_local(info_key, action, rng):
                    values[i] = 0.0
                    continue
                next_state = self.game.apply_action(state, action)
                values[i] = self._cfr_traverse_local(next_state, traverser, weight, rng, delta_r, delta_s, delta_am)

            ev = float(strategy @ values)
            regret_update = values - ev

            if info_key not in delta_r:
                delta_r[info_key] = np.zeros(NUM_ACTIONS)
            for i, a in enumerate(actions):
                delta_r[info_key][a] += regret_update[i] * weight

            if info_key not in delta_s:
                delta_s[info_key] = np.zeros(NUM_ACTIONS)
            for i, a in enumerate(actions):
                delta_s[info_key][a] += strategy[i] * weight

            return ev

        else:
            action_idx = int(rng.choice(len(actions), p=strategy))
            next_state = self.game.apply_action(state, actions[action_idx])
            return self._cfr_traverse_local(next_state, traverser, weight, rng, delta_r, delta_s, delta_am)

    def _traverse_worker(
        self,
        state: NLHEState,
        traverser: int,
        weight: float,
        rng: np.random.Generator,
    ) -> Tuple[StrategyTable, StrategyTable, ActionMap]:
        """
        Thread worker: one CFR traversal with thread-local delta dicts.
        Returns (delta_regret, delta_strategy, delta_action_map).
        state is a pre-dealt NLHEState owned exclusively by this worker.
        """
        delta_r: StrategyTable = {}
        delta_s: StrategyTable = {}
        delta_am: ActionMap = {}
        self._cfr_traverse_local(state, traverser, weight, rng, delta_r, delta_s, delta_am)
        return delta_r, delta_s, delta_am

    def _merge_deltas(
        self,
        results: List[Tuple[StrategyTable, StrategyTable, ActionMap]],
    ) -> None:
        """
        Merge a batch of (delta_r, delta_s, delta_am) into the shared tables.
        Called single-threaded after all futures in a batch have resolved.
        """
        for delta_r, delta_s, delta_am in results:
            for key, arr in delta_r.items():
                if key not in self.regret_sum:
                    self.regret_sum[key] = np.zeros(NUM_ACTIONS)
                self.regret_sum[key] += arr

            for key, arr in delta_s.items():
                if key not in self.strategy_sum:
                    self.strategy_sum[key] = np.zeros(NUM_ACTIONS)
                self.strategy_sum[key] += arr

            for key, actions in delta_am.items():
                if key not in self.action_map:
                    self.action_map[key] = actions

    def train_parallel(
        self,
        num_iterations: int,
        num_workers: int = PARALLEL_WORKERS,
        batch_size: int = PARALLEL_BATCH_SIZE,
        log_interval: int = 1,
        checkpoint_interval: int = 0,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """
        Run num_iterations of MCCFR using threaded minibatch parallelism.

        Each logical iteration t dispatches batch_size traversals across num_workers
        threads. Traversers are assigned round-robin. All writes are deferred to a
        single-threaded merge phase after each batch, so no locks are needed.

        Requires a free-threaded Python build (GIL disabled) for true parallelism.
        Incompatible with debug mode.
        batch_size should be a multiple of num_players (3).
        """
        assert not self.debugger.enabled, (
            "train_parallel is incompatible with debug mode. Use train() for debug traversals."
        )
        if batch_size % self.num_players != 0:
            warnings.warn(
                f"batch_size={batch_size} is not a multiple of num_players={self.num_players}. "
                "Traverser balance across the batch will be uneven.",
                stacklevel=2,
            )

        start = self.iteration
        end = start + num_iterations
        print(
            f"Starting parallel MCCFR: {num_iterations} iterations "
            f"(total {start} -> {end}), workers={num_workers}, batch_size={batch_size}..."
        )

        # One RNG per job slot — reused across iterations, never shared between concurrent jobs
        rngs = [np.random.default_rng(seed=i) for i in range(batch_size)]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for t in tqdm(range(start + 1, end + 1), desc="Training (parallel)..."):
                self.iteration = t
                weight = float(t) if self.use_linear_cfr else 1.0

                # Deal all hands on the main thread (uses global np.random, must stay serial)
                jobs = [
                    (self.game.deal_new_hand(), i % self.num_players, weight, rngs[i])
                    for i in range(batch_size)
                ]

                futures = [
                    executor.submit(self._traverse_worker, state, traverser, weight, rng)
                    for state, traverser, weight, rng in jobs
                ]

                results = [f.result() for f in futures]
                self._merge_deltas(results)

                if log_interval and t % log_interval == 0:
                    avg_regret = self._compute_avg_regret(batch_size=batch_size)
                    print(
                        f"  Iter {t}/{end} | "
                        f"Info sets: {len(self.regret_sum)} | "
                        f"Avg regret: {avg_regret:.7f}"
                    )

                if checkpoint_interval and checkpoint_path and t % checkpoint_interval == 0:
                    path = checkpoint_path.format(iter=t) if "{iter}" in checkpoint_path else checkpoint_path
                    self.save(path)

        print(f"Training complete. {len(self.regret_sum)} info sets.")

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
        start = self.iteration
        end = start + num_iterations
        print(f"Starting MCCFR for {num_iterations} iterations (total {start} -> {end})...")

        for t in tqdm(range(start + 1, end + 1), "Training..."):
            self.iteration = t
            for traverser in range(self.num_players):
                self.debugger.begin_traversal(traverser, t)
                state = self.game.deal_new_hand()
                self.cfr_traverse(state, traverser)
                self.debugger.end_traversal()

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

        print(f"Training complete. {len(self.regret_sum)} info sets.")

    def _compute_avg_regret(self, batch_size: int = 1) -> float:
        if not self.regret_sum or self.iteration == 0:
            return 0.0
        if self.use_linear_cfr:
            sum_weights = (self.iteration * (self.iteration + 1)) / 2
        else:
            sum_weights = self.iteration
        # In parallel mode regret_sum is inflated by batch_size per iteration
        sum_weights *= batch_size
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
