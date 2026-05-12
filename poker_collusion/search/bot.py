"""
PluribusBot: online play agent using blueprint on preflop and depth-limited
subgame search on all postflop streets.

Lifecycle per hand:
    bot.reset_hand(state)              # at hand start
    action = bot.select_action(state)  # when it's the bot's turn
    bot.observe_action(state, action)  # after every action (including own)

The bot maintains a reach-probability vector (1326 entries, one per hole-card
pair) which is updated every time it takes an action.  On postflop streets the
bot runs the subgame solver using those reach probabilities to choose an action.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from poker_collusion.config import (
    BIAS_FACTOR,
    NUM_ACTIONS,
    SUBGAME_CFR_ITERATIONS,
    SUBGAME_DEPTH_LIMIT,
    SUBGAME_LEAF_ROLLOUTS,
)
from poker_collusion.search.reach import (
    NUM_PAIRS,
    ReachTracker,
    _build_pair_table,
    pair_index,
)
from poker_collusion.search.solver import SubgameSolver


class PluribusBot:
    """
    Self-contained Pluribus-style poker bot.

    Parameters
    ----------
    game : CFRGame-compatible module (deal_new_hand, apply_action, …)
    blueprint : object with get_average_strategy(info_key, legal_actions)
    seat : int — which player index this bot occupies (0, 1, or 2)
    num_players : int
    cfr_iterations : int — iterations for the subgame solver
    depth_limit : int — subgame depth in betting rounds
    leaf_rollouts : int — MC rollouts per non-terminal leaf
    bias_factor : float — multiplier for biased continuation strategies
    """

    def __init__(
        self,
        game: Any,
        blueprint: Any,
        seat: int,
        num_players: int = 3,
        cfr_iterations: int = SUBGAME_CFR_ITERATIONS,
        depth_limit: int = SUBGAME_DEPTH_LIMIT,
        leaf_rollouts: int = SUBGAME_LEAF_ROLLOUTS,
        bias_factor: float = BIAS_FACTOR,
    ) -> None:
        self.game = game
        self.blueprint = blueprint
        self.seat = seat
        self.num_players = num_players
        self.cfr_iterations = cfr_iterations
        self.depth_limit = depth_limit
        self.leaf_rollouts = leaf_rollouts
        self.bias_factor = bias_factor

        self._reach = ReachTracker()
        self._hole: Optional[Tuple[int, int]] = None
        self._bucket_cache: Dict = {}

    def reset_hand(self, state: Any) -> None:
        """
        Initialise for a new hand.  Call once at the start of each hand
        after cards are dealt.
        """
        c0, c1 = state.hole_cards[self.seat]
        self._hole = (min(c0, c1), max(c0, c1))
        visible = set(state.hole_cards[self.seat]) | set(state.board)
        self._reach.reset(visible)
        self._bucket_cache = {}

    def select_action(self, state: Any) -> int:
        """
        Choose an abstract action index for the current game state.

        On preflop: sample from the blueprint average strategy.
        On postflop: run the subgame solver and use the resulting strategy
        for the bot's actual hand.
        """
        actions = self.game.get_legal_actions(state)
        if not actions:
            raise RuntimeError("select_action called with no legal actions")

        if state.round_idx == 0:
            return self._preflop_action(state, actions)

        return self._postflop_action(state, actions)

    def observe_action(self, state_before: Any, action: int, actor: int) -> None:
        """
        Notify the bot of an action taken (by anyone, including itself).

        If the action was taken by this bot, update the reach probabilities.
        Also update reach probs when new board cards are dealt (call
        ``observe_new_street`` for that).
        """
        if actor == self.seat:
            self._update_reach_after_own_action(state_before, action)

    def observe_new_street(self, state: Any) -> None:
        """Zero out pairs containing newly dealt board cards and renormalize."""
        self._reach.zero_out(set(state.board))

    # ── Private helpers ──────────────────────────────────────────────────

    def _preflop_action(self, state: Any, actions: List[int]) -> int:
        """Sample from blueprint average strategy."""
        info_key = self.game.get_info_key(state, self.seat)
        probs = self.blueprint.get_average_strategy(info_key, actions)
        if probs is None or len(probs) != len(actions):
            probs = np.ones(len(actions)) / len(actions)
        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]

    def _postflop_action(self, state: Any, actions: List[int]) -> int:
        """Run subgame solver and pick the action for our actual hand."""
        solver = SubgameSolver(
            game=self.game,
            blueprint=self.blueprint,
            num_players=self.num_players,
            cfr_iterations=self.cfr_iterations,
            depth_limit=self.depth_limit,
            leaf_rollouts=self.leaf_rollouts,
            bias_factor=self.bias_factor,
        )

        hand_strategies = solver.solve(state, self.seat, self._reach.probs)

        probs = hand_strategies.get(self._hole)
        if probs is None or len(probs) != len(actions):
            probs = np.ones(len(actions)) / len(actions)

        idx = np.random.choice(len(actions), p=probs)
        return actions[idx]

    def _update_reach_after_own_action(
        self, state_before: Any, action: int
    ) -> None:
        """
        After the bot takes *action* at *state_before*, update reach probs.

        For each feasible hand, compute the probability that the blueprint
        strategy would choose *action* in that infoset, then use that as
        the per-pair update weight.

        Optimization: group feasible hands by their bucket (infoset key),
        look up the blueprint strategy once per unique bucket, and broadcast.
        """
        actions = self.game.get_legal_actions(state_before)
        if not actions or action not in actions:
            return

        action_pos = actions.index(action)
        table = _build_pair_table()
        action_probs = np.zeros(NUM_PAIRS, dtype=np.float64)

        feasible = self._reach.feasible_pairs()

        from poker_collusion.abstraction.bucketing import get_bucket, hole_to_canonical
        round_idx = state_before.round_idx
        board = tuple(state_before.board)
        history = tuple(state_before.action_history)

        bucket_to_indices: Dict[Any, List[int]] = {}
        for idx in feasible:
            c0, c1 = int(table[idx, 0]), int(table[idx, 1])
            hole = (min(c0, c1), max(c0, c1))
            cache_key = (hole, round_idx)
            if cache_key in self._bucket_cache:
                bucket = self._bucket_cache[cache_key]
            elif round_idx == 0:
                bucket = int(hole_to_canonical(hole))
                self._bucket_cache[cache_key] = bucket
            else:
                bucket = int(get_bucket(hole, board, round_idx))
                self._bucket_cache[cache_key] = bucket
            bucket_to_indices.setdefault(bucket, []).append(idx)

        for bucket, indices in bucket_to_indices.items():
            info_key = (bucket, history)
            bp_probs = self.blueprint.get_average_strategy(info_key, actions)
            if bp_probs is None or len(bp_probs) != len(actions):
                bp_probs = np.ones(len(actions)) / len(actions)
            prob = bp_probs[action_pos]
            for idx in indices:
                action_probs[idx] = prob

        self._reach.update(action, action_probs)

    # ── Policy interface for evaluation harness ──────────────────────────

    def get_action_probs(
        self, state: Any, player: int, legal_actions: List[int]
    ) -> np.ndarray:
        """
        Evaluation-compatible interface: returns action probabilities.

        For the bot's own seat, uses the full select logic.
        For other seats, falls back to blueprint.
        """
        if player != self.seat:
            info_key = self.game.get_info_key(state, player)
            probs = self.blueprint.get_average_strategy(info_key, legal_actions)
            if probs is None or len(probs) != len(legal_actions):
                return np.ones(len(legal_actions)) / len(legal_actions)
            return probs

        if state.round_idx == 0:
            info_key = self.game.get_info_key(state, self.seat)
            probs = self.blueprint.get_average_strategy(info_key, legal_actions)
            if probs is None or len(probs) != len(legal_actions):
                return np.ones(len(legal_actions)) / len(legal_actions)
            return probs

        solver = SubgameSolver(
            game=self.game,
            blueprint=self.blueprint,
            num_players=self.num_players,
            cfr_iterations=self.cfr_iterations,
            depth_limit=self.depth_limit,
            leaf_rollouts=self.leaf_rollouts,
            bias_factor=self.bias_factor,
        )
        hand_strategies = solver.solve(state, self.seat, self._reach.probs)
        probs = hand_strategies.get(self._hole)
        if probs is None or len(probs) != len(legal_actions):
            return np.ones(len(legal_actions)) / len(legal_actions)
        return probs
