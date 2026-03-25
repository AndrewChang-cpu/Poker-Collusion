"""
MCCFR trainer: external sampling, Linear CFR, regret pruning.
Game module must provide: deal_new_hand, get_current_player, get_legal_actions,
get_info_key, is_terminal, get_payoffs, apply_action, undo_action, is_chance_node, sample_chance.
"""
"""
MCCFR trainer: external sampling, Linear CFR, regret pruning.
"""

import sys
import traceback
import json
import numpy as np
import os
from tqdm import tqdm
from poker_collusion.cfr.strategy import regret_matching, get_average_strategy
from poker_collusion.config import (
    NUM_ACTIONS,
    PRUNE_THRESHOLD,
    PRUNE_WARM_UP_ITERATIONS,
    PRUNE_SKIP_PROBABILITY,
)


# Debug: full traceback and NDJSON logs (used by train() and cfr_traverse)
_CFR_ERROR_LOG = "logs/cfr_error_traceback.log"
_DEBUG_LOG = "logs/debug.log"


def write_to_debug(entry):
    try:
        with open(_DEBUG_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

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

    def cfr_traverse(self, state, traverser, depth=0, depth_limit=500):
        if depth > depth_limit:
            return 0.0
    
        if self.game.is_terminal(state):
            return self.game.get_payoffs(state)[traverser]

        if self.game.is_chance_node(state):
            new_state = self.game.sample_chance(state)
            return self.cfr_traverse(new_state, traverser, len(new_state.action_history))

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

            return ev
        else:
            action_idx = np.random.choice(num_actions, p=strategy)
            self.game.apply_action(state, actions[action_idx])
            val = self.cfr_traverse(state, traverser, len(state.action_history))
            self.game.undo_action()
            return val

    def _should_prune(self, info_key, action):
        if self.prune_threshold is None or self.iteration <= self.prune_warm_up:
            return False
        regrets = self.regret_sum.get(info_key, np.zeros(NUM_ACTIONS))
        if action < len(regrets) and regrets[action] < self.prune_threshold:
            return np.random.random() < self.prune_skip_prob
        return False

    def train(self, num_iterations, log_interval=1, checkpoint_interval=0, checkpoint_path=None, show_progress=True):
        start = self.iteration
        end = start + num_iterations
        
        iterator = range(start + 1, end + 1)
        if show_progress:
            iterator = tqdm(iterator, desc="Training...")

        try:
            for t in iterator:
                self.iteration = t
                for traverser in range(self.num_players):
                    state = self.game.deal_new_hand()
                    self.cfr_traverse(state, traverser)

                #if True:
                if log_interval and t % log_interval == 0 and show_progress:
                    avg_regret = self._compute_avg_regret()
                    print(f"  Iter {t}/{end} | Info sets: {len(self.regret_sum)} | Avg regret: {avg_regret:.7f}")

                    if checkpoint_interval and checkpoint_path and t % checkpoint_interval == 0:
                        path = checkpoint_path.format(iter=t) if "{iter}" in checkpoint_path else checkpoint_path
                        self.save(path)
        except Exception:
            with open(_CFR_ERROR_LOG, "a") as f:
                f.write(f"\nException at iteration {self.iteration}\n")
                f.write(traceback.format_exc())
            raise

    def merge(self, other):
        """Sum regrets and strategies from another trainer instance."""
        for key, val in other.regret_sum.items():
            if key in self.regret_sum:
                # Resize if necessary to maintain consistency
                if len(self.regret_sum[key]) < len(val):
                    self.regret_sum[key] = np.resize(self.regret_sum[key], len(val))
                self.regret_sum[key] += val[:len(self.regret_sum[key])]
            else:
                self.regret_sum[key] = val.copy()
        
        for key, val in other.strategy_sum.items():
            if key in self.strategy_sum:
                if len(self.strategy_sum[key]) < len(val):
                    self.strategy_sum[key] = np.resize(self.strategy_sum[key], len(val))
                self.strategy_sum[key] += val[:len(self.strategy_sum[key])]
            else:
                self.strategy_sum[key] = val.copy()
        
        for key, val in other.action_map.items():
            if key not in self.action_map:
                self.action_map[key] = val

    def _compute_avg_regret(self):
        if not self.regret_sum or self.iteration == 0:
            return 0.0
        sum_weights = (self.iteration * (self.iteration + 1)) / 2 if self.use_linear_cfr else self.iteration
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
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "regret_sum": self.regret_sum,
                "strategy_sum": self.strategy_sum,
                "action_map": self.action_map,
                "iteration": self.iteration,
            }, f)

    def load(self, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.regret_sum = data["regret_sum"]
        self.strategy_sum = data["strategy_sum"]
        self.action_map = data["action_map"]
        self.iteration = data["iteration"]
        #print(f"Loaded from {path} (iter {self.iteration})")