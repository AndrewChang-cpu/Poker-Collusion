"""
Bucketing Determinism Tests
============================
Tests for the correctness and reproducibility of the bucketing module.
These tests were written to FIND BUGS, not to pass.

KNOWN BUG (discovered via these tests):
  _estimate_equity() in poker_collusion/abstraction/bucketing.py uses
  Python's `random.shuffle()` instead of `np.random.shuffle()`.
  This makes postflop bucket assignments non-deterministic across calls
  even when np.random.seed() is fixed.
  Reference: .planning/phase-1/PLAN.md threat model + bug report in BUG_REPORT.md
"""

import numpy as np
import random
import pytest

from poker_collusion.abstraction.bucketing import get_bucket


class TestBucketingDeterminism:
    """Bucket assignment must be deterministic (same inputs → same output)."""

    # A specific hand: AA (card 0 = Ace of Spades, card 13 = Ace of Hearts)
    HOLE_AA = [0, 13]
    # A specific board: 3c (card 2), 3h (card 15), 7s (card 28)
    BOARD_3 = [2, 15, 28]
    # Turn board
    BOARD_4 = [2, 15, 28, 5]
    # River board
    BOARD_5 = [2, 15, 28, 5, 18]

    def test_preflop_bucket_deterministic(self):
        """
        Preflop bucketing uses a precomputed table (no randomness).
        Calling get_bucket() multiple times must return the same value.
        """
        buckets = [get_bucket(self.HOLE_AA, [], round_idx=0) for _ in range(10)]
        assert len(set(buckets)) == 1, \
            f"Preflop bucket for AA is non-deterministic: {buckets}"

    def test_flop_bucket_deterministic(self):
        """
        Flop bucketing uses equity estimation.  Without seeding Python's random,
        get_bucket() returns different buckets on different calls.

        THIS TEST IS EXPECTED TO FAIL due to the bug in _estimate_equity()
        which uses random.shuffle() (Python random) instead of np.random.shuffle().
        [BUG: _estimate_equity uses Python random module — non-deterministic]
        """
        # Do NOT seed random here — exposing the bug
        buckets = [get_bucket(self.HOLE_AA, self.BOARD_3, round_idx=1)
                   for _ in range(20)]
        unique = set(buckets)
        assert len(unique) == 1, (
            f"FLOP BUCKETING IS NON-DETERMINISTIC: same AA hand assigned to "
            f"{len(unique)} different buckets across 20 calls: {sorted(unique)}\n"
            f"BUG: _estimate_equity() uses random.shuffle() (Python's random module) "
            f"instead of np.random.shuffle() (numpy RNG). Seeding np.random.seed() "
            f"does NOT fix this — Python's random module has separate state."
        )

    def test_turn_bucket_deterministic(self):
        """
        Turn bucketing has the same bug.
        [BUG: _estimate_equity uses Python random module]
        """
        buckets = [get_bucket(self.HOLE_AA, self.BOARD_4, round_idx=2)
                   for _ in range(20)]
        unique = set(buckets)
        assert len(unique) == 1, (
            f"TURN BUCKETING IS NON-DETERMINISTIC: {len(unique)} different buckets: "
            f"{sorted(unique)}"
        )

    def test_river_bucket_deterministic(self):
        """
        River bucketing has the same bug.
        Note: river bucket should be deterministic if precomputed tables are loaded,
        because at river all 5 cards are known and equity is exact.
        But with fallback _postflop_fallback (no randomness), river IS deterministic.
        [Tests the fallback path]
        """
        # River has all 5 cards: evaluate_hand is deterministic
        buckets = [get_bucket(self.HOLE_AA, self.BOARD_5, round_idx=3)
                   for _ in range(10)]
        assert len(set(buckets)) == 1, \
            f"River bucket should be deterministic (no rollouts needed for full board)"

    def test_same_hand_same_bucket_with_fixed_seed(self):
        """
        With random.seed() AND np.random.seed() both fixed,
        bucket assignment is deterministic.
        This confirms the root cause: Python random module is the culprit.
        """
        buckets = []
        for _ in range(10):
            random.seed(42)  # Must seed Python random too
            np.random.seed(42)
            b = get_bucket(self.HOLE_AA, self.BOARD_3, round_idx=1)
            buckets.append(b)
        assert len(set(buckets)) == 1, \
            f"Even with random.seed(42), flop bucket varies: {buckets}"

    def test_equity_variance_across_rollouts(self):
        """
        With n_rollouts=100, equity estimate has high variance (up to ~14% range).
        With 50 buckets, this translates to ~7 bucket differences.
        This shows the equity estimate is too noisy to be reliable.
        [Documents the severity of the randomness bug]
        """
        hole = self.HOLE_AA
        board = self.BOARD_3

        # Collect equity-implied buckets by repeatedly calling with different random seeds
        buckets = set()
        for seed in range(50):
            random.seed(seed)
            b = get_bucket(hole, board, round_idx=1)
            buckets.add(b)

        # With a correctly deterministic implementation, there should be exactly 1 bucket.
        # With the bug, we expect many distinct buckets.
        num_distinct = len(buckets)
        assert num_distinct == 1, (
            f"AA flop bucket varies across {num_distinct} distinct values with different "
            f"random seeds: {sorted(buckets)}\n"
            f"This indicates the equity estimate has high variance AND uses Python random. "
            f"A correct implementation would use a deterministic equity lookup or seed numpy."
        )


class TestRealGameDeterminism:
    """
    The real training loop must be reproducible with a fixed seed.
    This test validates the Determinism invariant from PLAN.md Task 12,
    applied to the real game instead of the stub game.
    """

    def test_real_game_training_deterministic_with_np_seed_only(self):
        """
        Training must be deterministic with np.random.seed() fixed.
        EXPECTED TO FAIL because _estimate_equity uses Python random module.
        [BUG: non-deterministic training due to Python random in bucketing]
        """
        import poker_collusion.env.game_state as gs
        import poker_collusion.env.game_logic as gl
        import poker_collusion.abstraction.info_set as info_set
        from poker_collusion.cfr.trainer import CFRTrainer

        class RealGame:
            def deal_new_hand(self): return gs.deal_new_hand()
            def is_terminal(self, s): return gl.is_terminal(s)
            def is_chance_node(self, s): return gl.is_chance_node(s)
            def sample_chance(self, s): return gl.sample_chance(s)
            def get_current_player(self, s): return gl.get_current_player(s)
            def get_legal_actions(self, s): return gl.get_legal_actions(s)
            def get_info_key(self, s, p): return info_set.get_info_key(s, p)
            def get_payoffs(self, s): return gs.get_payoffs(s)
            def apply_action(self, s, a): return gl.apply_action(s, a)

        def run():
            np.random.seed(42)
            # NOTE: NOT seeding random.seed() — exposing the bug
            trainer = CFRTrainer(RealGame(), num_players=3, use_linear_cfr=False,
                                 prune_threshold=None, debug=False)
            trainer.train(num_iterations=3, log_interval=0)
            return frozenset(trainer.regret_sum.keys())

        run1 = run()
        run2 = run()
        run3 = run()

        assert run1 == run2 == run3, (
            f"Training is non-deterministic with np.random.seed(42) alone.\n"
            f"Run 1 unique keys: {len(run1)}, Run 2: {len(run2)}, Run 3: {len(run3)}\n"
            f"Symmetric diff 1↔2: {run1.symmetric_difference(run2)}\n"
            f"ROOT CAUSE: poker_collusion/abstraction/bucketing.py _estimate_equity() "
            f"uses random.shuffle() (Python's random module) instead of "
            f"np.random.shuffle(). Fix: replace random.shuffle with "
            f"np.random.shuffle, or use np.random.choice without replacement."
        )
