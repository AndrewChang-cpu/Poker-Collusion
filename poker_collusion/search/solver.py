"""
Depth-limited subgame CFR solver for online search.

Given a game state and the searching player's reach-probability vector, this
module runs a fixed number of Linear CFR iterations over the subgame tree and
returns a strategy (distribution over legal actions) for every feasible hand
the searcher could hold.

Subgame structure
-----------------
* **Root**: chance node; one branch per feasible hole-card pair, weighted by
  the searcher's normalised reach probabilities.
* **Current street**: infosets keyed by exact card identity (canonical hand id
  for preflop, raw hole pair for postflop current street).
* **Future streets**: infosets keyed by blueprint buckets (equity-based
  abstraction), same as the offline blueprint.
* **Depth limit**: counted in betting rounds (street transitions).  When a node
  exceeds the limit **or** the game is terminal, it is a leaf.
* **Leaf values**: terminal nodes use exact payoff.  Non-terminal depth-limit
  leaves are evaluated via continuation-strategy rollouts (see
  ``continuation.py``).

The solver is stateless across calls — all regret/strategy tables are local to
each ``solve()`` invocation and discarded afterward.
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
from poker_collusion.abstraction.bucketing import get_bucket, hole_to_canonical
from poker_collusion.abstraction.info_set import get_info_key
from poker_collusion.cfr.strategy import regret_matching, get_average_strategy
from poker_collusion.search.continuation import (
    NUM_CONTINUATION_STRATEGIES,
    rollout_leaf_value,
)
from poker_collusion.search.reach import NUM_PAIRS, ReachTracker, pair_index, _build_pair_table

InfoKey = Any
StrategyTable = Dict[InfoKey, np.ndarray]


def _subgame_info_key(
    state: Any,
    player: int,
    root_street: int,
    bucket_cache: Optional[Dict] = None,
) -> InfoKey:
    """
    Build an infoset key for the subgame.

    On the *root_street* (the street when search was initiated), use exact
    card identities — the canonical hand id for preflop, or a (c0, c1) tuple
    for postflop.  On later streets, fall back to blueprint bucketing.

    If *bucket_cache* is provided, postflop bucket lookups are memoized by
    (sorted_hole, round_idx) since the board is deterministic per street
    within a single solve() call.
    """
    round_idx = state.round_idx
    hole = tuple(sorted(state.hole_cards[player]))

    if round_idx == root_street:
        if round_idx == 0:
            bucket = int(hole_to_canonical(hole))
        else:
            bucket = hole
    else:
        cache_key = (hole, round_idx)
        if bucket_cache is not None and cache_key in bucket_cache:
            bucket = bucket_cache[cache_key]
        else:
            board = tuple(state.board)
            bucket = int(get_bucket(hole, board, round_idx))
            if bucket_cache is not None:
                bucket_cache[cache_key] = bucket

    history = []
    for a in state.action_history:
        if isinstance(a, list):
            history.append(tuple(a))
        else:
            history.append(a)
    return (bucket, tuple(history))


class SubgameSolver:
    """
    Runs depth-limited CFR on a subgame rooted at a given game state.

    The solver is single-use: call ``solve()`` once and use the result.
    """

    def __init__(
        self,
        game: Any,
        blueprint: Any,
        num_players: int = 3,
        cfr_iterations: int = SUBGAME_CFR_ITERATIONS,
        depth_limit: int = SUBGAME_DEPTH_LIMIT,
        leaf_rollouts: int = SUBGAME_LEAF_ROLLOUTS,
        bias_factor: float = BIAS_FACTOR,
    ) -> None:
        self.game = game
        self.blueprint = blueprint
        self.num_players = num_players
        self.cfr_iterations = cfr_iterations
        self.depth_limit = depth_limit
        self.leaf_rollouts = leaf_rollouts
        self.bias_factor = bias_factor

    def solve(
        self,
        root_state: Any,
        searcher: int,
        reach_probs: np.ndarray,
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Run subgame CFR and return a strategy for the searcher.

        Parameters
        ----------
        root_state : NLHEState at the decision point (postflop).
        searcher : player index of the searching bot.
        reach_probs : shape (1326,) reach probability vector for the searcher.

        Returns
        -------
        dict mapping (c0, c1) -> np.ndarray of probabilities over legal_actions
        at the root.  Only feasible hands (reach > 0) are included.
        """
        root_street = root_state.round_idx
        table = _build_pair_table()

        feasible_idx = np.nonzero(reach_probs > 0)[0]
        if len(feasible_idx) == 0:
            return {}

        weights = reach_probs[feasible_idx].copy()
        weights /= weights.sum()

        regret_sum: StrategyTable = {}
        strategy_sum: StrategyTable = {}
        bucket_cache: Dict = {}

        for t in range(1, self.cfr_iterations + 1):
            lcfr_weight = float(t)

            hand_idx = np.random.choice(feasible_idx, p=weights)
            c0, c1 = int(table[hand_idx, 0]), int(table[hand_idx, 1])

            state = root_state.copy()
            state.hole_cards[searcher] = [c0, c1]

            self._cfr_traverse(
                state=state,
                searcher=searcher,
                root_street=root_street,
                depth=0,
                weight=lcfr_weight,
                regret_sum=regret_sum,
                strategy_sum=strategy_sum,
                bucket_cache=bucket_cache,
            )

        result: Dict[Tuple[int, int], np.ndarray] = {}
        root_actions = self.game.get_legal_actions(root_state)
        if not root_actions:
            return result

        for idx in feasible_idx:
            c0, c1 = int(table[idx, 0]), int(table[idx, 1])
            state_copy = root_state.copy()
            state_copy.hole_cards[searcher] = [c0, c1]
            info_key = _subgame_info_key(state_copy, searcher, root_street, bucket_cache)

            if info_key in strategy_sum:
                s = strategy_sum[info_key]
                s_sub = np.array([s[a] if a < len(s) else 0.0 for a in root_actions])
                total = s_sub.sum()
                if total > 0:
                    probs = s_sub / total
                else:
                    probs = np.ones(len(root_actions)) / len(root_actions)
            else:
                probs = np.ones(len(root_actions)) / len(root_actions)

            result[(c0, c1)] = probs

        return result

    def _cfr_traverse(
        self,
        state: Any,
        searcher: int,
        root_street: int,
        depth: int,
        weight: float,
        regret_sum: StrategyTable,
        strategy_sum: StrategyTable,
        bucket_cache: Optional[Dict] = None,
    ) -> float:
        """Recursive CFR traversal of the subgame tree."""
        if self.game.is_terminal(state):
            return self.game.get_payoffs(state)[searcher]

        streets_advanced = state.round_idx - root_street
        if streets_advanced >= self.depth_limit:
            return self._leaf_value(state, searcher, bucket_cache)

        if self.game.is_chance_node(state):
            next_state = self.game.sample_chance(state)
            return self._cfr_traverse(
                next_state, searcher, root_street, depth + 1,
                weight, regret_sum, strategy_sum, bucket_cache,
            )

        player = self.game.get_current_player(state)
        actions = self.game.get_legal_actions(state)
        if not actions:
            return self.game.get_payoffs(state)[searcher]

        info_key = _subgame_info_key(state, player, root_street, bucket_cache)

        regrets_full = regret_sum.get(info_key, np.zeros(NUM_ACTIONS))
        regrets_sub = np.array([regrets_full[a] for a in actions])
        strategy = regret_matching(regrets_sub, len(actions))

        if player == searcher:
            values = np.zeros(len(actions))
            for i, action in enumerate(actions):
                next_state = self.game.apply_action(state, action)
                values[i] = self._cfr_traverse(
                    next_state, searcher, root_street, depth + 1,
                    weight, regret_sum, strategy_sum, bucket_cache,
                )

            ev = float(strategy @ values)
            regret_update = values - ev

            if info_key not in regret_sum:
                regret_sum[info_key] = np.zeros(NUM_ACTIONS)
            for i, a in enumerate(actions):
                regret_sum[info_key][a] += regret_update[i] * weight

            if info_key not in strategy_sum:
                strategy_sum[info_key] = np.zeros(NUM_ACTIONS)
            for i, a in enumerate(actions):
                strategy_sum[info_key][a] += strategy[i] * weight

            return ev
        else:
            action_idx = np.random.choice(len(actions), p=strategy)
            next_state = self.game.apply_action(state, actions[action_idx])
            return self._cfr_traverse(
                next_state, searcher, root_street, depth + 1,
                weight, regret_sum, strategy_sum, bucket_cache,
            )

    def _leaf_value(
        self, state: Any, searcher: int, bucket_cache: Optional[Dict] = None,
    ) -> float:
        """
        Estimate value at a non-terminal depth-limit leaf.

        Samples a random combination of continuation strategies for all
        players and rolls out the remainder of the game.
        """
        continuation_ids = [
            np.random.randint(NUM_CONTINUATION_STRATEGIES)
            for _ in range(self.num_players)
        ]
        payoffs = rollout_leaf_value(
            game=self.game,
            state=state,
            blueprint=self.blueprint,
            num_players=self.num_players,
            continuation_ids=continuation_ids,
            num_rollouts=self.leaf_rollouts,
            bias_factor=self.bias_factor,
            bucket_cache=bucket_cache,
        )
        return float(payoffs[searcher])
