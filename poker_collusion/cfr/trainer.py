"""
MCCFR trainer: modified to support Shared Information (psychic) collusion.
"""

from __future__ import annotations
import os
import numpy as np
from tqdm import tqdm
from poker_collusion.cfr.strategy import regret_matching
from poker_collusion.config import NUM_ACTIONS, LINEAR_CFR_CUTOFF, PRUNE_THRESHOLD, PRUNE_WARM_UP_ITERATIONS, PRUNE_SKIP_PROBABILITY

class CFRTrainer:
    def __init__(
        self,
        game_module,
        num_players: int = 3,
        use_shared_info: bool = False, # NEW
        team_seats = None,
        frozen_trainer = None,
        team_objective: str = "utilitarian",
        **kwargs
    ) -> None:
        self.game = game_module
        self.num_players = num_players
        self.use_shared_info = use_shared_info # NEW
        self.team_seats = set(team_seats) if team_seats else set()
        self.frozen_trainer = frozen_trainer
        self.team_objective = team_objective
        self.regret_sum = {}
        self.strategy_sum = {}
        self.action_map = {}
        self.iteration = 0
        self.linear_cfr_cutoff = LINEAR_CFR_CUTOFF
        self.use_linear_cfr = True
        self.full_history = True

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
        
        # Determine if we should pass team_seats for psychic keys
        t_seats = list(self.team_seats) if self.use_shared_info else None
        info_key = self.game.get_info_key(state, player, team_seats=t_seats)

        if info_key not in self.action_map:
            self.action_map[info_key] = list(actions)

        strategy = self.get_strategy(info_key, actions)
        if self.frozen_trainer and player not in self.team_seats:
            frozen_strat = self.frozen_trainer.get_average_strategy(info_key, actions)
            if frozen_strat is not None:
                strategy = frozen_strat

        if player == traverser:
            values = np.zeros(len(actions))
            for i, action in enumerate(actions):
                values[i] = self.cfr_traverse(self.game.apply_action(state, action), traverser)

            ev = float(strategy @ values)
            regret_update = values - ev
            weight = self.iteration if self.iteration <= self.linear_cfr_cutoff else 1.0

            if info_key not in self.regret_sum: self.regret_sum[info_key] = np.zeros(NUM_ACTIONS)
            if info_key not in self.strategy_sum: self.strategy_sum[info_key] = np.zeros(NUM_ACTIONS)
            
            for i, a in enumerate(actions):
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
            "use_shared_info": self.use_shared_info, # NEW
            "team_seats": sorted(list(self.team_seats))
        }
        with open(path, "wb") as f: pickle.dump(data, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f: data = pickle.load(f)
        self.regret_sum = data["regret_sum"]
        self.strategy_sum = data["strategy_sum"]
        self.action_map = data["action_map"]
        self.iteration = data["iteration"]
        self.use_shared_info = data.get("use_shared_info", False) # NEW
        self.team_seats = set(data.get("team_seats", []))