"""
Monte Carlo CFR trainer with external sampling.

Supports two modes (auto-detected):
  1. Copy-based: game.apply_action returns a new state (kuhn3p, nlhe3p)
  2. Step-back-based: game uses shared env with undo_action() (rlcard_nlhe3p)

  

Notes:
    - We might want to include a calculated parameter that measures how "exploitable" a play/hand is by finding how many chips would be lost against a "best response" play
        -- To do this we might fix the strategies of colluding bots and find the best way to exploit their strategies - this is our measure of convergence
"""

import numpy as np


class CFRTrainer:
    def __init__(self, game_module, num_players=3, use_linear_cfr=True, prune_threshold=-300):

        """Note: We might want to occasionally re-introduce "pruned" strategies so that the bots don't "cut out" a viable action due to bad luck and never rediscover it"""
        
        self.game = game_module
        self.num_players = num_players
        self.use_linear_cfr = use_linear_cfr
        self.prune_threshold = prune_threshold

        # Auto-detect if game uses step_back pattern
        self.use_step_back = hasattr(game_module, 'undo_action')

        # Core data structures
        self.regret_sum = {}     # info_key -> np.array of cumulative regrets
        self.strategy_sum = {}   # info_key -> np.array of cumulative strategy
        self.action_map = {}     # info_key -> list of actions

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
        return np.ones(len(s)) / len(s)

    def cfr_traverse(self, state, traverser):
        """Recursive MCCFR with external sampling."""
        if self.game.is_terminal(state):
            return self.game.get_payoffs(state)[traverser]

        if self.game.is_chance_node(state):
            new_state = self.game.sample_chance(state)
            return self.cfr_traverse(new_state, traverser)

        player = self.game.get_current_player(state)
        actions = self.game.get_legal_actions(state)
        info_key = self.game.get_info_key(state, player)
        num_actions = len(actions)

        if num_actions == 0:
            return 0.0

        if info_key not in self.action_map:
            self.action_map[info_key] = list(actions)

        strategy = self.get_strategy(info_key, num_actions)

        if player == traverser:
            # ---- Explore ALL traverser actions ----
            values = np.zeros(num_actions)
            for i, action in enumerate(actions):
                if self._should_prune(info_key, i, num_actions):
                    continue

                next_state = self.game.apply_action(state, action)
                values[i] = self.cfr_traverse(next_state, traverser)
                if self.use_step_back:
                    self.game.undo_action()

            ev = strategy @ values
            regret_update = values - ev
            weight = self.iteration if self.use_linear_cfr else 1

            if info_key not in self.regret_sum:
                self.regret_sum[info_key] = np.zeros(num_actions)
            self.regret_sum[info_key] += regret_update * weight

            if info_key not in self.strategy_sum:
                self.strategy_sum[info_key] = np.zeros(num_actions)
            self.strategy_sum[info_key] += strategy * weight

            return ev
        else:
            # ---- Sample ONE opponent action ----
            action_idx = np.random.choice(num_actions, p=strategy)
            next_state = self.game.apply_action(state, actions[action_idx])
            val = self.cfr_traverse(next_state, traverser)
            if self.use_step_back:
                self.game.undo_action()
            return val

    def _should_prune(self, info_key, action_idx, num_actions):
        if self.prune_threshold is None or self.iteration <= 100:
            return False
        regrets = self.regret_sum.get(info_key, np.zeros(num_actions))
        if action_idx < len(regrets) and regrets[action_idx] < self.prune_threshold:
            return np.random.random() < 0.95
        return False

    def train(self, num_iterations, log_interval=1000):
        """Main training loop."""
        mode = "step-back (RLCard)" if self.use_step_back else "copy-based"
        print(f"Starting MCCFR training for {num_iterations} iterations ({mode})...")

        for t in range(1, num_iterations + 1):
            self.iteration = t
            for traverser in range(self.num_players):
                state = self.game.deal_new_hand()
                self.cfr_traverse(state, traverser)

            if t % log_interval == 0:
                avg_regret = self._compute_avg_regret()
                print(f"  Iter {t}/{num_iterations} | "
                      f"Info sets: {len(self.regret_sum)} | "
                      f"Avg regret: {avg_regret:.4f}")

        print(f"Training complete. {len(self.regret_sum)} info sets.")

    def _compute_avg_regret(self):
        if not self.regret_sum:
            return 0.0
        total = sum(np.abs(r).mean() for r in self.regret_sum.values())
        avg = total / len(self.regret_sum)
        if self.use_linear_cfr and self.iteration > 0:
            avg /= self.iteration
        return avg

    def get_all_strategies(self):
        strategies = {}
        for info_key in self.strategy_sum:
            avg = self.get_average_strategy(info_key)
            actions = self.action_map.get(info_key, [])
            if avg is not None:
                strategies[info_key] = (actions, avg)
        return strategies

    def save(self, path):
        import pickle
        data = {
            "regret_sum": self.regret_sum,
            "strategy_sum": self.strategy_sum,
            "action_map": self.action_map,
            "iteration": self.iteration,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved to {path}")

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.regret_sum = data["regret_sum"]
        self.strategy_sum = data["strategy_sum"]
        self.action_map = data["action_map"]
        self.iteration = data["iteration"]
        print(f"Loaded from {path} (iter {self.iteration})")