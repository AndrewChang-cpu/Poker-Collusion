"""
Monte Carlo CFR trainer with external sampling.

Works with any game that provides:
  - deal_new_hand() -> state
  - get_current_player(state) -> int
  - get_legal_actions(state) -> list
  - get_info_key(state, player) -> str
  - is_terminal(state) -> bool
  - get_payoffs(state) -> list[float]
  - apply_action(state, action) -> state
  - is_chance_node(state) -> bool
  - sample_chance(state) -> state
"""

import numpy as np
from collections import defaultdict


class CFRTrainer:
    def __init__(self, game_module, num_players=3, use_linear_cfr=True, prune_threshold=-300):
        """
        Args:
            game_module: Module with game functions (kuhn3p or nlhe3p)
            num_players: Number of players
            use_linear_cfr: If True, weight iterations linearly (Linear CFR)
            prune_threshold: Prune actions with regret below this (set None to disable)
        """
        self.game = game_module
        self.num_players = num_players
        self.use_linear_cfr = use_linear_cfr
        self.prune_threshold = prune_threshold

        # Core data structures
        self.regret_sum = {}     # info_key -> np.array of cumulative regrets
        self.strategy_sum = {}   # info_key -> np.array of cumulative strategy
        self.action_map = {}     # info_key -> list of actions (to maintain ordering)

        # Tracking
        self.iteration = 0

    def get_strategy(self, info_key, num_actions):
        """Regret matching: convert regrets to action probabilities."""
        regrets = self.regret_sum.get(info_key, np.zeros(num_actions))
        positive = np.maximum(regrets, 0)
        total = positive.sum()
        if total > 0:
            return positive / total
        return np.ones(num_actions) / num_actions

    def get_average_strategy(self, info_key):
        """Final strategy after training (normalized cumulative strategy)."""
        if info_key not in self.strategy_sum:
            return None
        s = self.strategy_sum[info_key]
        total = s.sum()
        if total > 0:
            return s / total
        n = len(s)
        return np.ones(n) / n

    def cfr_traverse(self, state, traverser):
        """
        Recursive MCCFR with external sampling.

        - Traverser's actions: explore ALL
        - Opponent's actions: SAMPLE one from current strategy
        - Chance nodes: SAMPLE one outcome
        """
        # Terminal check
        if self.game.is_terminal(state):
            return self.game.get_payoffs(state)[traverser]

        # Chance node
        if self.game.is_chance_node(state):
            new_state = self.game.sample_chance(state)
            return self.cfr_traverse(new_state, traverser)

        player = self.game.get_current_player(state)
        actions = self.game.get_legal_actions(state)
        info_key = self.game.get_info_key(state, player)
        num_actions = len(actions)

        # Store action ordering for this info set
        if info_key not in self.action_map:
            self.action_map[info_key] = list(actions)

        strategy = self.get_strategy(info_key, num_actions)

        if player == traverser:
            # ---- Traverser: explore all actions ----
            values = np.zeros(num_actions)

            for i, action in enumerate(actions):
                # Pruning: skip very negative regret actions most of the time
                if self.prune_threshold is not None and self.iteration > 100:
                    regrets = self.regret_sum.get(info_key, np.zeros(num_actions))
                    if regrets[i] < self.prune_threshold and np.random.random() < 0.95:
                        continue

                next_state = self.game.apply_action(state, action)
                values[i] = self.cfr_traverse(next_state, traverser)

            # Expected value under current strategy
            ev = strategy @ values

            # Update regrets
            regret_update = values - ev
            if info_key not in self.regret_sum:
                self.regret_sum[info_key] = np.zeros(num_actions)

            if self.use_linear_cfr:
                # Linear CFR: weight regret by iteration number t
                # This makes early (bad) iterations decay as 2/(T*(T+1))
                self.regret_sum[info_key] += regret_update * self.iteration
            else:
                self.regret_sum[info_key] += regret_update

            # Update cumulative strategy for average computation
            if info_key not in self.strategy_sum:
                self.strategy_sum[info_key] = np.zeros(num_actions)

            if self.use_linear_cfr:
                self.strategy_sum[info_key] += strategy * self.iteration
            else:
                self.strategy_sum[info_key] += strategy

            return ev

        else:
            # ---- Opponent: sample one action ----
            action_idx = np.random.choice(num_actions, p=strategy)
            next_state = self.game.apply_action(state, actions[action_idx])
            return self.cfr_traverse(next_state, traverser)

    def train(self, num_iterations, log_interval=1000):
        """
        Main training loop.

        Each iteration: rotate traverser, deal hand, traverse.
        """
        print(f"Starting MCCFR training for {num_iterations} iterations...")

        for t in range(1, num_iterations + 1):
            self.iteration = t

            for traverser in range(self.num_players):
                state = self.game.deal_new_hand()
                self.cfr_traverse(state, traverser)

            if t % log_interval == 0:
                num_info_sets = len(self.regret_sum)
                avg_regret = self._compute_avg_regret()
                print(f"  Iteration {t}/{num_iterations} | "
                      f"Info sets: {num_info_sets} | "
                      f"Avg abs regret: {avg_regret:.4f}")

        print(f"Training complete. Total info sets: {len(self.regret_sum)}")

    def _compute_avg_regret(self):
        """Average absolute regret across all info sets (convergence metric).
        Normalized by iteration to account for Linear CFR weighting."""
        if not self.regret_sum:
            return 0.0
        total = 0.0
        count = 0
        for key, regrets in self.regret_sum.items():
            total += np.abs(regrets).mean()
            count += 1
        avg = total / count if count > 0 else 0.0
        # Normalize: with linear CFR, regrets scale as O(T^2), so divide by T
        if self.use_linear_cfr and self.iteration > 0:
            avg /= self.iteration
        return avg

    def get_all_strategies(self):
        """Return dict of info_key -> (actions, average_strategy)."""
        strategies = {}
        for info_key in self.strategy_sum:
            avg = self.get_average_strategy(info_key)
            actions = self.action_map.get(info_key, [])
            if avg is not None:
                strategies[info_key] = (actions, avg)
        return strategies

    def save(self, path):
        """Save trained strategy to file."""
        import pickle
        data = {
            "regret_sum": self.regret_sum,
            "strategy_sum": self.strategy_sum,
            "action_map": self.action_map,
            "iteration": self.iteration,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Strategy saved to {path}")

    def load(self, path):
        """Load trained strategy from file."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.regret_sum = data["regret_sum"]
        self.strategy_sum = data["strategy_sum"]
        self.action_map = data["action_map"]
        self.iteration = data["iteration"]
        print(f"Strategy loaded from {path} (iteration {self.iteration})")
