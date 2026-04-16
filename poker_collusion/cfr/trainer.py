"""
MCCFR trainer: external sampling, Linear CFR, regret pruning.
Game module must provide: deal_new_hand, get_current_player, get_legal_actions,
get_info_key, is_terminal, get_payoffs, apply_action, is_chance_node, sample_chance.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from poker_collusion.cfr.strategy import regret_matching, get_average_strategy
from poker_collusion.cfr.debug import CFRDebugger
from poker_collusion.config import (
    NUM_ACTIONS,
    LINEAR_CFR_CUTOFF,
    PRUNE_THRESHOLD,
    PRUNE_WARM_UP_ITERATIONS,
    PRUNE_SKIP_PROBABILITY,
    PARALLEL_WORKERS,
    PARALLEL_BATCH_SIZE,
    SMOOTH_LAMBDA,
    RISK_OFFSET,
)
from poker_collusion.typing_defs import CFRGame
from poker_collusion.env.game_state import NLHEState


InfoKey = Any
StrategyTable = Dict[InfoKey, np.ndarray]
ActionMap = Dict[InfoKey, List[int]]

_DEAL_SENTINEL = "DEAL"


def _to_street_only_key(key: Any) -> Any:
    """Reduce a full-history infoset key to its current-street-only equivalent.

    Used for backwards-compatible lookup in old-format (pre-full-history) trainers.
    On preflop there are no DEAL sentinels so the key is returned unchanged.
    """
    if not (isinstance(key, tuple) and len(key) == 3):
        return key
    round_idx, bucket, history = key
    if _DEAL_SENTINEL not in history:
        return key  # Preflop or no prior streets — formats are identical
    last_deal = max(i for i, e in enumerate(history) if e == _DEAL_SENTINEL)
    return (round_idx, bucket, history[last_deal + 1:])


class CFRTrainer:
    def __init__(
        self,
        game_module: CFRGame,
        num_players: int = 3,
        use_linear_cfr: bool = True,
        linear_cfr_cutoff: int = LINEAR_CFR_CUTOFF,
        prune_threshold: Optional[float] = PRUNE_THRESHOLD,
        prune_warm_up: int = PRUNE_WARM_UP_ITERATIONS,
        prune_skip_prob: float = PRUNE_SKIP_PROBABILITY,
        debug: bool = False,
        debug_step: bool = False,
        debug_consolidate: bool = True,
        team_seats: Optional[List[int]] = None,
        frozen_trainer: Optional['CFRTrainer'] = None,
        team_objective: str = "utilitarian",
    ) -> None:
        self.game = game_module
        self.num_players = num_players
        self.use_linear_cfr = use_linear_cfr
        self.linear_cfr_cutoff = linear_cfr_cutoff
        self.prune_threshold = prune_threshold
        self.prune_warm_up = prune_warm_up
        self.prune_skip_prob = prune_skip_prob
        self.debugger = CFRDebugger(
            enabled=debug,
            step=debug_step,
            consolidate=debug_consolidate,
        )

        self.full_history: bool = True  # always enabled; old pkls translate on lookup
        self.team_seats: Set[int] = set(team_seats) if team_seats else set()
        self.frozen_trainer: Optional['CFRTrainer'] = frozen_trainer
        if frozen_trainer and not frozen_trainer.strategy_sum:
            raise ValueError("Frozen trainer has no strategy data — empty or failed load.")

        _VALID_OBJECTIVES = {"utilitarian", "maxmin", "smooth", "risk"}
        if team_objective not in _VALID_OBJECTIVES:
            raise ValueError(f"Unknown team_objective '{team_objective}'; valid: {_VALID_OBJECTIVES}")
        self.team_objective: str = team_objective

        self.regret_sum: StrategyTable = {}
        self.strategy_sum: StrategyTable = {}
        self.action_map: ActionMap = {}
        self.iteration: int = 0

    def _iteration_weight(self, t: Optional[int] = None) -> float:
        """Return the weight for iteration *t* (defaults to self.iteration).

        With Linear CFR active and t <= cutoff, the weight is t (Linear CFR).
        Beyond the cutoff (or with Linear CFR disabled), the weight is 1.
        """
        if t is None:
            t = self.iteration
        if self.use_linear_cfr and t <= self.linear_cfr_cutoff:
            return float(t)
        return 1.0

    def _team_value(self, payoffs, traverser: int) -> float:
        """Compute the scalar return for *traverser* at a terminal node."""
        if not self.team_seats or traverser not in self.team_seats:
            return payoffs[traverser]
        team_payoffs = [payoffs[s] for s in self.team_seats]
        obj = self.team_objective
        if obj == "utilitarian":
            return sum(team_payoffs)
        if obj == "maxmin":
            return min(team_payoffs)
        if obj == "smooth":
            return sum(team_payoffs) + SMOOTH_LAMBDA * min(team_payoffs)
        if obj == "risk":
            import math
            return sum(math.log(RISK_OFFSET + u) for u in team_payoffs)
        raise ValueError(f"Unknown team objective: {obj}")

    def get_strategy(self, info_key: InfoKey, legal_actions: List[int]) -> np.ndarray:
        """Return strategy distribution over legal_actions (length len(legal_actions))."""
        regrets_full = self.regret_sum.get(info_key, np.zeros(NUM_ACTIONS))
        if len(regrets_full) != NUM_ACTIONS:
            raise ValueError(
                f"Regret array size mismatch for {info_key}: "
                f"got {len(regrets_full)}, expected {NUM_ACTIONS}. "
                f"Blueprint may be from an incompatible version."
            )
        regrets_sub = np.array([regrets_full[a] for a in legal_actions])
        return regret_matching(regrets_sub, len(legal_actions))

    def get_average_strategy(
        self, info_key: InfoKey, legal_actions: Optional[List[int]] = None
    ) -> Optional[np.ndarray]:
        """If legal_actions given, return normalized dist over those; else full length-NUM_ACTIONS.

        If this trainer was loaded from an old-format (street-only) pkl, the incoming
        full-history key is automatically translated to the street-only equivalent before
        lookup so that old strategies remain usable as frozen opponents.
        """
        if not self.full_history:
            info_key = _to_street_only_key(info_key)
        if info_key not in self.strategy_sum:
            return None
        s = self.strategy_sum[info_key]
        if len(s) != NUM_ACTIONS:
            raise ValueError(
                f"Strategy array size mismatch for {info_key}: "
                f"got {len(s)}, expected {NUM_ACTIONS}. "
                f"Blueprint may be from an incompatible version."
            )
        if legal_actions is not None:
            s_sub = np.array([s[a] for a in legal_actions])
            return get_average_strategy(s_sub, len(legal_actions))
        return get_average_strategy(s, NUM_ACTIONS)

    def cfr_traverse(self, state: NLHEState, traverser: int) -> float:
        """
        Recursively traverse the game tree from state for the given traverser.
        apply_action and sample_chance return new state copies — no undo needed.
        """
        # print(f"player: {traverser} depth: {len(state.action_history)} cfr_traverse: {state.action_history}")

        if self.game.is_terminal(state):
            payoffs = self.game.get_payoffs(state)
            self.debugger.on_terminal(state, traverser, len(state.action_history), self.iteration, payoffs)
            return self._team_value(payoffs, traverser)

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

        if self.frozen_trainer and player not in self.team_seats:
            frozen_strat = self.frozen_trainer.get_average_strategy(info_key, actions)
            if frozen_strat is not None:
                strategy = frozen_strat

        if player == traverser:
            self.debugger.on_traverser_node(
                state, player, traverser, len(state.action_history), self.iteration,
                actions, strategy, info_key,
            )

            values = np.zeros(len(actions))
            pruned = np.zeros(len(actions), dtype=bool)
            for i, action in enumerate(actions):
                if self._should_prune(info_key, action):
                    pruned[i] = True
                    continue
                self.debugger.push_branch(action)
                next_state = self.game.apply_action(state, action)
                values[i] = self.cfr_traverse(next_state, traverser)
                self.debugger.pop_branch()

            if pruned.any():
                s_masked = strategy.copy()
                s_masked[pruned] = 0.0
                s_total = s_masked.sum()
                ev = float((s_masked / s_total) @ values) if s_total > 0 else 0.0
            else:
                ev = float(strategy @ values)
            regret_update = values - ev
            regret_update[pruned] = 0.0
            weight = self._iteration_weight()

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
        _depth: int = 0,
    ) -> float:
        """
        Thread-safe CFR traversal. Reads self.regret_sum (read-only during batch phase).
        All writes go into caller-supplied delta_r / delta_s / delta_am dicts.
        Uses rng (thread-local Generator) instead of global np.random.
        Debug hooks are absent — incompatible with parallel mode.
        """
        if _depth > 500:
            raise RuntimeError(f"CFR traversal depth exceeded: history={state.action_history}")

        if self.game.is_terminal(state):
            payoffs = self.game.get_payoffs(state)
            return self._team_value(payoffs, traverser)

        if self.game.is_chance_node(state):
            next_state = self.game.sample_chance(state)
            return self._cfr_traverse_local(next_state, traverser, weight, rng, delta_r, delta_s, delta_am, _depth + 1)

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

        if self.frozen_trainer and player not in self.team_seats:
            frozen_strat = self.frozen_trainer.get_average_strategy(info_key, actions)
            if frozen_strat is not None:
                strategy = frozen_strat

        if player == traverser:
            values = np.zeros(len(actions))
            pruned = np.zeros(len(actions), dtype=bool)
            for i, action in enumerate(actions):
                if self._should_prune_local(info_key, action, rng):
                    pruned[i] = True
                    continue
                next_state = self.game.apply_action(state, action)
                values[i] = self._cfr_traverse_local(next_state, traverser, weight, rng, delta_r, delta_s, delta_am, _depth + 1)

            if pruned.any():
                s_masked = strategy.copy()
                s_masked[pruned] = 0.0
                s_total = s_masked.sum()
                ev = float((s_masked / s_total) @ values) if s_total > 0 else 0.0
            else:
                ev = float(strategy @ values)
            regret_update = values - ev
            regret_update[pruned] = 0.0

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
            return self._cfr_traverse_local(next_state, traverser, weight, rng, delta_r, delta_s, delta_am, _depth + 1)

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

    def _assert_full_history_for_training(self) -> None:
        """Raise immediately if this trainer was loaded from an old-format checkpoint."""
        if not self.full_history:
            raise RuntimeError(
                "Cannot train from a street-only checkpoint (full_history=False).\n"
                "The loaded strategy uses street-only infoset keys which are incompatible\n"
                "with the current full-history training format. Starting training on top\n"
                "of old keys would silently discard all prior regret data.\n"
                "Fix: omit --load to start fresh, or retrain the checkpoint with full history."
            )

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
        self._assert_full_history_for_training()
        assert not self.debugger.enabled, (
            "train_parallel is incompatible with debug mode. Use train() for debug traversals."
        )
        traversers = list(range(self.num_players))
        if self.team_seats:
            traversers = [p for p in traversers if p in self.team_seats]
        num_traversers = len(traversers)

        if batch_size % num_traversers != 0:
            warnings.warn(
                f"batch_size={batch_size} is not a multiple of active traversers={num_traversers}. "
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
            pbar = tqdm(range(start + 1, end + 1), desc="Training (parallel)...")
            for t in pbar:
                self.iteration = t
                weight = self._iteration_weight(t)

                # Deal all hands on the main thread (uses global np.random, must stay serial)
                jobs = [
                    (self.game.deal_new_hand(), traversers[i % num_traversers], weight, rngs[i])
                    for i in range(batch_size)
                ]

                futures = [
                    executor.submit(self._traverse_worker, state, traverser, weight, rng)
                    for state, traverser, weight, rng in jobs
                ]

                results = [f.result() for f in futures]
                self._merge_deltas(results)

                if log_interval and t % log_interval == 0:
                    avg_regret = self._compute_avg_regret()
                    tqdm.write(
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
        self._assert_full_history_for_training()
        start = self.iteration
        end = start + num_iterations
        print(f"Starting MCCFR for {num_iterations} iterations (total {start} -> {end})...")

        traversers = list(range(self.num_players))
        if self.team_seats:
            traversers = [p for p in traversers if p in self.team_seats]

        pbar = tqdm(range(start + 1, end + 1), "Training...")
        for t in pbar:
            self.iteration = t
            for traverser in traversers:
                self.debugger.begin_traversal(traverser, t)
                state = self.game.deal_new_hand()
                self.cfr_traverse(state, traverser)
                self.debugger.end_traversal()

            if log_interval and t % log_interval == 0:
                avg_regret = self._compute_avg_regret()
                tqdm.write(
                    f"  Iter {t}/{end} | "
                    f"Info sets: {len(self.regret_sum)} | "
                    f"Avg regret: {avg_regret:.7f}"
                )

            if checkpoint_interval and checkpoint_path and t % checkpoint_interval == 0:
                path = checkpoint_path.format(iter=t) if "{iter}" in checkpoint_path else checkpoint_path
                self.save(path)

        print(f"Training complete. {len(self.regret_sum)} info sets.")

    def _compute_avg_regret(self) -> float:
        if not self.regret_sum or self.iteration == 0:
            return 0.0
        t = self.iteration
        c = self.linear_cfr_cutoff
        if self.use_linear_cfr:
            if t <= c:
                sum_weights = (t * (t + 1)) / 2
            else:
                sum_weights = (c * (c + 1)) / 2 + (t - c)
        else:
            sum_weights = t
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
        data = {
            "regret_sum": self.regret_sum,
            "strategy_sum": self.strategy_sum,
            "action_map": self.action_map,
            "iteration": self.iteration,
            "linear_cfr_cutoff": self.linear_cfr_cutoff,
        }
        data["full_history"] = self.full_history
        if self.team_seats:
            data["team_seats"] = sorted(self.team_seats)
            data["team_objective"] = self.team_objective
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"\nSaved to {path}")

    def load(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.regret_sum = data["regret_sum"]
        self.strategy_sum = data["strategy_sum"]
        self.action_map = data["action_map"]
        self.iteration = data["iteration"]
        if "linear_cfr_cutoff" in data:
            self.linear_cfr_cutoff = data["linear_cfr_cutoff"]
        saved_fh = data.get("full_history", False)
        if not saved_fh:
            print(
                "Warning: loaded strategy was trained with street-only history "
                "(full_history=False).\n"
                "  Postflop strategy lookups will be automatically translated to the "
                "full-history key format.\n"
                "  This strategy cannot be used as a training checkpoint — "
                "call train() or train_parallel() will raise an error.\n"
                "  For best results, retrain the strategy with full history enabled."
            )
        self.full_history = saved_fh
        if "team_seats" in data:
            self.team_seats = set(data["team_seats"])
        if "team_objective" in data:
            self.team_objective = data["team_objective"]
        print(f"Loaded from {path} (iter {self.iteration})")
