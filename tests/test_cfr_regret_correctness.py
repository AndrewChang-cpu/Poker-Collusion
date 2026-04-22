"""
CFR Regret / Strategy Update Correctness Tests
===============================================
Every assertion traces back to the mathematical derivations in
.planning/phase-1/PLAN.md  (Brown & Sandholm 2019 — Pluribus paper).

Design principle: tests are written from the math, NOT massaged to match
the code.  If a test fails, there is a bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from poker_collusion.cfr.trainer import CFRTrainer
from poker_collusion.cfr.strategy import regret_matching, get_average_strategy
from poker_collusion.config import (
    NUM_ACTIONS,
    LINEAR_CFR_CUTOFF,
    PRUNE_THRESHOLD,
    PRUNE_WARM_UP_ITERATIONS,
    PRUNE_SKIP_PROBABILITY,
)
from tests.conftest import (
    FakeState,
    StubGame,
    PLAN_V_TABLE,
    PLAN_LEGAL,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trainer(traverser=0, prune_threshold=None, prune_warm_up=100,
                 linear_cfr_cutoff=LINEAR_CFR_CUTOFF, use_linear_cfr=True,
                 team_seats=None, frozen_trainer=None, train_seats=None):
    """Return a CFRTrainer wired to the PLAN.md root state for testing."""
    root = FakeState(
        kind='traverser_root',
        traverser=traverser,
        actor=traverser,
        legal_actions=PLAN_LEGAL,
        value_table=PLAN_V_TABLE,
    )
    game = StubGame(root)
    return CFRTrainer(
        game_module=game,
        num_players=3, 
        use_linear_cfr=use_linear_cfr,
        linear_cfr_cutoff=linear_cfr_cutoff,
        prune_threshold=prune_threshold,
        prune_warm_up=prune_warm_up,
        team_seats=team_seats,
        frozen_trainer=frozen_trainer,
        train_seats=train_seats,
        debug=False,
    )


INFO_KEY_P0 = (0, tuple(PLAN_LEGAL))
INFO_KEY_P1 = (1, tuple(PLAN_LEGAL))
INFO_KEY_P2 = (2, tuple(PLAN_LEGAL))

def _R(trainer, key=INFO_KEY_P0):
    """Convenience: regret_sum array for a specific info-key."""
    return trainer.regret_sum.get(key, np.zeros(NUM_ACTIONS)).copy()

def _S(trainer, key=INFO_KEY_P0):
    """Convenience: strategy_sum array for a specific info-key."""
    return trainer.strategy_sum.get(key, np.zeros(NUM_ACTIONS)).copy()


# ---------------------------------------------------------------------------
# Task 11: regret_matching unit tests
# ---------------------------------------------------------------------------

class TestRegretMatching:
    """PLAN.md Task 11 — regret_matching() unit tests."""

    def test_all_zero_gives_uniform(self):
        """All-zero R → uniform σ."""
        r = np.zeros(10)
        sigma = regret_matching(r, 10)
        expected = np.full(10, 0.1)
        np.testing.assert_allclose(sigma, expected, atol=1e-12)

    def test_all_negative_gives_uniform(self):
        """All-negative R → uniform σ."""
        r = np.array([-5.0, -1.0, -0.001, -100.0, -2.0, -3.0, -4.0, -6.0, -7.0, -8.0])
        sigma = regret_matching(r, 10)
        expected = np.full(10, 0.1)
        np.testing.assert_allclose(sigma, expected, atol=1e-12)

    def test_mixed_gives_positive_only(self):
        """Mixed R → only positive entries get mass."""
        r = np.array([-1.25, -0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
        sigma = regret_matching(r, 10)
        assert sigma[0] == 0.0
        assert sigma[1] == 0.0
        expected_raise = 0.25 / (8 * 0.25)
        np.testing.assert_allclose(sigma[2:], expected_raise, atol=1e-12)


# ---------------------------------------------------------------------------
# Weight and Boundary Tests
# ---------------------------------------------------------------------------

class TestIterationWeight:
    """PLAN.md Task 7 — Linear CFR weight schedule and Phase 3 capping."""

    def test_weight_at_t1(self):
        t = make_trainer()
        t.iteration = 1
        assert t._iteration_weight() == 1.0

    def test_weight_at_cutoff(self):
        t = make_trainer()
        t.iteration = LINEAR_CFR_CUTOFF
        assert t._iteration_weight() == float(LINEAR_CFR_CUTOFF)

    def test_weight_past_cutoff(self):
        """Weights must stay capped at LINEAR_CFR_CUTOFF (Step 4 Fix)."""
        t = make_trainer()
        t.iteration = LINEAR_CFR_CUTOFF + 500
        assert t._iteration_weight() == float(LINEAR_CFR_CUTOFF)


# ---------------------------------------------------------------------------
# Victim Modeling and Immutability Verification
# ---------------------------------------------------------------------------

class TestVictimModelingImmutability:
    """
    Phase 3: Verifies that 'Frozen' seats do not drift during training.
    """

    def test_frozen_colluders_stay_frozen(self):
        """
        If train_seats={2}, regrets for seats 0 and 1 must not be created.
        Fixed: Explicitly set traverser=2 so Seat 2 actually acts in the test game.
        """
        trainer = make_trainer(traverser=2, train_seats=[2])
        assert not trainer.regret_sum
        
        trainer.train(10)
        
        # Seat 2 (Victim) should now have regrets
        assert INFO_KEY_P2 in trainer.regret_sum
        # Seats 0 and 1 (Colluders) should remain empty/untouched
        assert INFO_KEY_P0 not in trainer.regret_sum
        assert INFO_KEY_P1 not in trainer.regret_sum

    def test_merged_load_immutability(self):
        """Resuming training for one seat should not modify existing data for others."""
        trainer = make_trainer(traverser=2, train_seats=[2])
        dummy_regret = np.ones(NUM_ACTIONS) * 55.0
        trainer.regret_sum[INFO_KEY_P0] = dummy_regret.copy()
        
        trainer.train(10)
        
        # Verify Seat 0 regret is EXACTLY as it was before
        np.testing.assert_array_equal(trainer.regret_sum[INFO_KEY_P0], dummy_regret)


# ---------------------------------------------------------------------------
# Invariant and Reset Tests
# ---------------------------------------------------------------------------

class TestSumInvariant:
    """PLAN.md Task 13 — Σ S[I] = Σ wₜ for all info-keys visited."""

    def _expected_weight_sum(self, num_iters, linear_cutoff):
        if num_iters <= linear_cutoff:
            return num_iters * (num_iters + 1) / 2
        linear_part = linear_cutoff * (linear_cutoff + 1) / 2
        post_linear = num_iters - linear_cutoff
        return linear_part + post_linear * float(linear_cutoff)

    @pytest.mark.parametrize("num_iters", [1, 5, 10])
    def test_strategy_sum_accumulation(self, num_iters):
        t = make_trainer(train_seats=[0])
        for i in range(1, num_iters + 1):
            t.iteration = i
            t.cfr_traverse(t.game.deal_new_hand(), traverser=0)
        
        expected = self._expected_weight_sum(num_iters, LINEAR_CFR_CUTOFF)
        assert _S(t, INFO_KEY_P0).sum() == pytest.approx(expected)


class TestResetLogic:
    """Phase 3: Verifies the adversarial pivot reset logic."""

    def test_partial_strategy_reset(self):
        """Verifies that we can reset strategy sums for specific seats."""
        trainer = make_trainer()
        trainer.strategy_sum[INFO_KEY_P0] = np.ones(NUM_ACTIONS)
        trainer.strategy_sum[INFO_KEY_P2] = np.ones(NUM_ACTIONS)
        
        trainer.reset_strategy_sum(seats=[2])
        
        assert INFO_KEY_P0 in trainer.strategy_sum
        assert INFO_KEY_P2 not in trainer.strategy_sum


# ---------------------------------------------------------------------------
# Pruning Boundary Tests
# ---------------------------------------------------------------------------

class TestPruningBoundary:
    """PLAN.md Task 6 — pruning behavior verification."""

    def test_prune_does_not_fire_before_warmup(self):
        trainer = make_trainer(prune_threshold=-300.0, prune_warm_up=100)
        trainer.regret_sum[INFO_KEY_P0] = np.full(NUM_ACTIONS, -10000.0)
        trainer.iteration = 100 
        for a in range(NUM_ACTIONS):
            assert not trainer._should_prune(INFO_KEY_P0, a)

    def test_prune_eligible_after_warmup(self):
        trainer = make_trainer(prune_threshold=-300.0, prune_warm_up=100)
        trainer.regret_sum[INFO_KEY_P0] = np.full(NUM_ACTIONS, -10000.0)
        trainer.iteration = 101
        np.random.seed(0)
        fired = any(trainer._should_prune(INFO_KEY_P0, 0) for _ in range(50))
        assert fired, "Pruning should fire when iteration > warm_up and regret < threshold"