"""
MCCFR trainer: external sampling, Linear CFR, regret pruning.
Game module must provide: deal_new_hand, get_current_player, get_legal_actions,
get_info_key, is_terminal, get_payoffs, apply_action, undo_action, is_chance_node, sample_chance.
"""

import numpy as np
from poker_collusion.cfr.strategy import regret_matching, get_average_strategy
from poker_collusion.config import (
    NUM_ACTIONS,
    PRUNE_THRESHOLD,
    PRUNE_WARM_UP_ITERATIONS,
    PRUNE_SKIP_PROBABILITY,
)


class CFRTrainer:
    def __init__(
        self,
        game_module,
        num_players=3,
        use_linear_cfr=True,
        prune_threshold=PRUNE_THRESHOLD,
        prune_warm_up=PRUNE_WARM_UP_ITERATIONS,
        prune_skip_prob=PRUNE_SKIP_PROBABILITY,
    ):
        self.game = game_module
        self.num_players = num_players
        self.use_linear_cfr = use_linear_cfr
        self.prune_threshold = prune_threshold
        self.prune_warm_up = prune_warm_up
        self.prune_skip_prob = prune_skip_prob
        self.use_step_back = hasattr(game_module, "undo_action") and callable(getattr(game_module, "undo_action", None))

        self.regret_sum = {}
        self.strategy_sum = {}
        self.action_map = {}
        self.iteration = 0

    def get_strategy(self, info_key, legal_actions):
        """Return strategy distribution over legal_actions (length len(legal_actions))."""
        regrets_full = self.regret_sum.get(info_key, np.zeros(NUM_ACTIONS))
        if len(regrets_full) < NUM_ACTIONS:
            regrets_full = np.resize(regrets_full, NUM_ACTIONS)
        regrets_sub = np.array([regrets_full[a] for a in legal_actions])
        return regret_matching(regrets_sub, len(legal_actions))

    def get_average_strategy(self, info_key, legal_actions=None):
        """If legal_actions given, return normalized dist over those (len(legal_actions)); else full length-NUM_ACTIONS."""
        if info_key not in self.strategy_sum:
            return None
        s = self.strategy_sum[info_key]
        if len(s) < NUM_ACTIONS:
            s = np.resize(s, NUM_ACTIONS)
        if legal_actions is not None:
            s_sub = np.array([s[a] for a in legal_actions])
            return get_average_strategy(s_sub, len(legal_actions))
        return get_average_strategy(s, NUM_ACTIONS)

    def cfr_traverse(self, state, traverser):
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

        strategy = self.get_strategy(info_key, actions)

        if player == traverser:
            values = np.zeros(num_actions)
            for i, action in enumerate(actions):
                if self._should_prune(info_key, action):
                    values[i] = 0.0
                    continue
                self.game.apply_action(state, action)
                values[i] = self.cfr_traverse(state, traverser)
                if self.use_step_back:
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

            return ev
        else:
            action_idx = np.random.choice(num_actions, p=strategy)
            self.game.apply_action(state, actions[action_idx])
            val = self.cfr_traverse(state, traverser)
            if self.use_step_back:
                self.game.undo_action()
            return val

    def _should_prune(self, info_key, action):
        """action is the abstract action index (0..9)."""
        if self.prune_threshold is None or self.iteration <= self.prune_warm_up:
            return False
        regrets = self.regret_sum.get(info_key, np.zeros(NUM_ACTIONS))
        if action < len(regrets) and regrets[action] < self.prune_threshold:
            return np.random.random() < self.prune_skip_prob
        return False

    def train(self, num_iterations, log_interval=1000):
        mode = "step-back" if self.use_step_back else "copy-based"
        print(f"Starting MCCFR for {num_iterations} iterations ({mode})...")

        for t in range(1, num_iterations + 1):
            self.iteration = t
            for traverser in range(self.num_players):
                state = self.game.deal_new_hand()
                self.cfr_traverse(state, traverser)

            if log_interval and t % log_interval == 0:
                avg_regret = self._compute_avg_regret()
                print(f"  Iter {t}/{num_iterations} | Info sets: {len(self.regret_sum)} | Avg regret: {avg_regret:.7f}")

        print(f"Training complete. {len(self.regret_sum)} info sets.")

    def _compute_avg_regret(self):
        if not self.regret_sum or self.iteration == 0:
            return 0.0
        if self.use_linear_cfr:
            sum_weights = (self.iteration * (self.iteration + 1)) / 2
        else:
            sum_weights = self.iteration
        total_pos = sum(np.maximum(regrets, 0).mean() for regrets in self.regret_sum.values())
        return (total_pos / len(self.regret_sum)) / sum_weights

    def get_all_strategies(self):
        out = {}
        for info_key in self.strategy_sum:
            actions = self.action_map.get(info_key, list(range(NUM_ACTIONS)))
            avg = self.get_average_strategy(info_key, actions)
            if avg is not None:
                out[info_key] = (actions, avg)
        return out

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "regret_sum": self.regret_sum,
                "strategy_sum": self.strategy_sum,
                "action_map": self.action_map,
                "iteration": self.iteration,
            }, f)
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
