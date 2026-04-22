"""
MCCFR trainer: modified to support Shared Information (psychic) collusion and Frozen Strategies.
Step 1: Added MCCFR Pruning logic to skip suboptimal paths.
Step 2: Restricted traversal to team seats to prevent frozen opponent drift.
Step 3: Expanded metadata persistence to include team_objective.
Step 4: Refactored helpers for unit test compatibility.
"""

from __future__ import annotations
import os
import numpy as np
from tqdm import tqdm
from poker_collusion.cfr.strategy import regret_matching
from poker_collusion.config import (
    NUM_ACTIONS, 
    LINEAR_CFR_CUTOFF, 
    LOG_INTERVAL,
    PRUNE_THRESHOLD,
    PRUNE_WARM_UP_ITERATIONS,
    PRUNE_SKIP_PROBABILITY
)

class CFRTrainer:
    def __init__(
        self,
        game_module,
        num_players: int = 3,
        use_shared_info: bool = False,
        team_seats = None,
        frozen_trainer = None,
        team_objective: str = "utilitarian",
        **kwargs
    ) -> None:
        self.game = game_module
        self.num_players = num_players
        self.use_shared_info = use_shared_info
        self.team_seats = set(team_seats) if team_seats else set()
        self.frozen_trainer = frozen_trainer
        self.team_objective = team_objective
        self.regret_sum = {}
        self.strategy_sum = {}
        self.action_map = {}
        self.iteration = 0
        self.linear_cfr_cutoff = LINEAR_CFR_CUTOFF
        self.use_linear_cfr = kwargs.get("use_linear_cfr", True)
        # Internalize pruning for tests
        self.prune_threshold = kwargs.get("prune_threshold", PRUNE_THRESHOLD)
        self.prune_warm_up = kwargs.get("prune_warm_up", PRUNE_WARM_UP_ITERATIONS)

    def _iteration_weight(self) -> float:
        """Calculate weight for current iteration (Capped Linear CFR)."""
        if not self.use_linear_cfr:
            return 1.0
        return float(min(self.iteration, self.linear_cfr_cutoff))

    def _should_prune(self, info_key: Any, action: int) -> bool:
        """Check if an action path should be skipped (Step 1)."""
        if self.prune_threshold is None or self.iteration <= self.prune_warm_up:
            return False
        
        regrets = self.regret_sum.get(info_key)
        if regrets is not None and regrets[action] < self.prune_threshold:
            return np.random.random() < PRUNE_SKIP_PROBABILITY
        return False

    def train(self, num_iterations: int) -> None:
        """Main MCCFR training loop (Step 2)."""
        pbar = tqdm(range(1, num_iterations + 1), desc="Training MCCFR")
        for t in pbar:
            self.iteration = t
            
            # Step 2: Traverse only team seats if frozen opponent exists.
            if self.frozen_trainer and self.team_seats:
                traverser_seats = list(self.team_seats)
            else:
                traverser_seats = list(range(self.num_players))

            for p in traverser_seats:
                state = self.game.deal_new_hand()
                self.cfr_traverse(state, traverser=p)
            
            if t % LOG_INTERVAL == 0:
                avg_regret = self._calculate_avg_regret()
                pbar.set_postfix({"avg_regret": f"{avg_regret:.4f}"})

    def _calculate_avg_regret(self) -> float:
        """Diagnostic to measure convergence."""
        if not self.regret_sum: return 0.0
        total_pos_regret = 0.0
        count = 0
        for info_key in self.regret_sum:
            regrets = self.regret_sum[info_key]
            actions = self.action_map.get(info_key, [])
            if not actions: continue
            pos_regret = sum(max(regrets[a], 0) for a in actions)
            total_pos_regret += pos_regret / len(actions)
            count += 1
        return total_pos_regret / count if count > 0 else 0.0

    def get_average_strategy(self, info_key, legal_actions):
        strat_sum = self.strategy_sum.get(info_key)
        if strat_sum is None:
            return np.ones(len(legal_actions)) / len(legal_actions)
        strat = np.array([strat_sum[a] for a in legal_actions])
        s = np.sum(strat)
        return strat / s if s > 0 else np.ones(len(legal_actions)) / len(legal_actions)

    def get_strategy(self, info_key, legal_actions):
        regrets_full = self.regret_sum.get(info_key, np.zeros(NUM_ACTIONS))
        regrets_sub = np.array([regrets_full[a] for a in legal_actions])
        return regret_matching(regrets_sub, len(legal_actions))

    def cfr_traverse(self, state, traverser):
        if self.game.is_terminal(state):
            payoffs = self.game.get_payoffs(state)
            return self._team_value(payoffs, traverser)

        if self.game.is_chance_node(state):
            return self.cfr_traverse(self.game.sample_chance(state), traverser)

        player = self.game.get_current_player(state)
        actions = self.game.get_legal_actions(state)
        
        t_seats = list(self.team_seats) if self.use_shared_info else None
        info_key = self.game.get_info_key(state, player, team_seats=t_seats)

        if info_key not in self.action_map:
            self.action_map[info_key] = list(actions)

        # Strategy lookup: use frozen opponent strategy if applicable
        strategy = self.get_strategy(info_key, actions)
        if self.frozen_trainer and player not in self.team_seats:
            strategy = self.frozen_trainer.get_average_strategy(info_key, actions)

        if player == traverser:
            values = np.zeros(len(actions))
            pruned = [False] * len(actions)
            
            for i, action in enumerate(actions):
                # Step 1: MCCFR Pruning logic helper
                if self._should_prune(info_key, action):
                    pruned[i] = True
                    continue

                values[i] = self.cfr_traverse(self.game.apply_action(state, action), traverser)

            ev = float(strategy @ values)
            regret_update = values - ev
            
            weight = self._iteration_weight()

            if info_key not in self.regret_sum: self.regret_sum[info_key] = np.zeros(NUM_ACTIONS)
            if info_key not in self.strategy_sum: self.strategy_sum[info_key] = np.zeros(NUM_ACTIONS)
            
            for i, a in enumerate(actions):
                if not pruned[i]:
                    self.regret_sum[info_key][a] += regret_update[i] * weight
                self.strategy_sum[info_key][a] += strategy[i] * weight
            return ev
        else:
            action_idx = np.random.choice(len(actions), p=strategy)
            return self.cfr_traverse(self.game.apply_action(state, actions[action_idx]), traverser)

    def _team_value(self, payoffs, traverser):
        if not self.team_seats or traverser not in self.team_seats:
            return payoffs[traverser]
        team_payoffs = [payoffs[s] for s in self.team_seats]
        if self.team_objective == "utilitarian": return sum(team_payoffs)
        return payoffs[traverser]

    def save(self, path):
        import pickle
        data = {
            "regret_sum": self.regret_sum, "strategy_sum": self.strategy_sum,
            "action_map": self.action_map, "iteration": self.iteration,
            "use_shared_info": self.use_shared_info,
            "team_seats": sorted(list(self.team_seats)),
            "team_objective": self.team_objective
        }
        with open(path, "wb") as f: pickle.dump(data, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f: data = pickle.load(f)
        self.regret_sum = data["regret_sum"]
        self.strategy_sum = data["strategy_sum"]
        self.action_map = data["action_map"]
        self.iteration = data["iteration"]
        self.use_shared_info = data.get("use_shared_info", False)
        self.team_seats = set(data.get("team_seats", []))
        self.team_objective = data.get("team_objective", "utilitarian")