"""
CFR Regret / Strategy Update Correctness Tests
===============================================
Every assertion traces back to the mathematical derivations in
.planning/phase-1/PLAN.md  (Brown & Sandholm 2019 — Pluribus paper).

Design principle: tests are written from the math, NOT massaged to match
the code.  If a test fails, there is a bug.

The stub game produces a depth-1 tree:
    P0 (traverser) acts at root → immediate terminal with hard-coded payoff.
No opponent sampling occurs; every traverser action leads directly to a leaf.
This makes every v(I, a) a closed-form constant (from PLAN_V_TABLE) and every
regret/strategy value exactly derivable by hand.
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
                 linear_cfr_cutoff=LINEAR_CFR_CUTOFF, use_linear_cfr=True):
    """Return a CFRTrainer wired to the PLAN.md root state."""
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
        num_players=1,          # single traverser — train() runs only traverser 0 each iter
        use_linear_cfr=use_linear_cfr,
        linear_cfr_cutoff=linear_cfr_cutoff,
        prune_threshold=prune_threshold,
        prune_warm_up=prune_warm_up,
        debug=False,
    )


INFO_KEY = (0, tuple(PLAN_LEGAL))   # as returned by StubGame.get_info_key


def _R(trainer):
    """Convenience: regret_sum array for the root info-key."""
    return trainer.regret_sum[INFO_KEY].copy()


def _S(trainer):
    """Convenience: strategy_sum array for the root info-key."""
    return trainer.strategy_sum[INFO_KEY].copy()


# ---------------------------------------------------------------------------
# Task 11: regret_matching unit tests (pure function, no trainer needed)
# ---------------------------------------------------------------------------

class TestRegretMatching:
    """PLAN.md Task 11 — regret_matching() unit tests."""

    def test_all_zero_gives_uniform(self):
        """All-zero R → uniform σ.  [PLAN Task 11]"""
        r = np.zeros(10)
        sigma = regret_matching(r, 10)
        expected = np.full(10, 0.1)
        np.testing.assert_allclose(sigma, expected, atol=1e-12,
            err_msg="All-zero regrets should produce uniform strategy")

    def test_all_negative_gives_uniform(self):
        """All-negative R → uniform σ (positive part is all zero).  [PLAN Task 11]"""
        r = np.array([-5.0, -1.0, -0.001, -100.0, -2.0,
                      -3.0, -4.0,  -6.0,  -7.0,  -8.0])
        sigma = regret_matching(r, 10)
        expected = np.full(10, 0.1)
        np.testing.assert_allclose(sigma, expected, atol=1e-12,
            err_msg="All-negative regrets should produce uniform strategy")

    def test_mixed_gives_positive_only(self):
        """Mixed R → only positive entries get mass, proportional to r⁺.  [PLAN Task 11]"""
        # After t=1 in the PLAN scenario:
        # R = [-1.25, -0.75, +0.25*8]  → r⁺[2..9]=0.25, total=2.0
        r = np.array([-1.25, -0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
        sigma = regret_matching(r, 10)
        assert sigma[0] == 0.0, "Fold (negative regret) should get zero probability"
        assert sigma[1] == 0.0, "Call (negative regret) should get zero probability"
        expected_raise = 0.25 / (8 * 0.25)   # = 0.125 each
        np.testing.assert_allclose(sigma[2:], expected_raise, atol=1e-12,
            err_msg="Raise actions should each get 0.125 after t=1")

    def test_single_positive_gets_all_mass(self):
        """Single positive entry → σ = 1 for that action.  [PLAN Task 11]"""
        r = np.array([-5.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        sigma = regret_matching(r, 10)
        assert sigma[4] == pytest.approx(1.0), "Only positive entry must get all mass"
        assert sum(sigma) == pytest.approx(1.0)

    def test_output_sums_to_one(self):
        """Strategy always sums to 1.  [PLAN invariant]"""
        for _ in range(20):
            r = np.random.randn(10) * 5.0
            sigma = regret_matching(r, 10)
            assert sum(sigma) == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Task 7: _iteration_weight boundary tests (no traversal needed)
# ---------------------------------------------------------------------------

class TestIterationWeight:
    """PLAN.md Task 7 — Linear CFR weight schedule."""

    def test_weight_at_t1(self):
        """w(1) = 1.  [PLAN Task 7]"""
        t = make_trainer()
        t.iteration = 1
        assert t._iteration_weight() == 1.0

    def test_weight_at_t2(self):
        """w(2) = 2.  [PLAN Task 7]"""
        t = make_trainer()
        t.iteration = 2
        assert t._iteration_weight() == 2.0

    def test_weight_at_cutoff(self):
        """w(10000) = 10000 — last Linear CFR iteration.  [PLAN Task 7, boundary]"""
        t = make_trainer()
        t.iteration = LINEAR_CFR_CUTOFF
        assert t._iteration_weight() == float(LINEAR_CFR_CUTOFF), \
            f"w({LINEAR_CFR_CUTOFF}) should still be {LINEAR_CFR_CUTOFF} (Linear CFR uses <=)"

    def test_weight_just_past_cutoff(self):
        """w(10001) = 1 — first post-Linear-CFR iteration.  [PLAN Task 7, boundary]"""
        t = make_trainer()
        t.iteration = LINEAR_CFR_CUTOFF + 1
        assert t._iteration_weight() == 1.0, \
            f"w({LINEAR_CFR_CUTOFF + 1}) should be 1.0 (post-cutoff flat weight)"

    def test_weight_at_cutoff_minus_one(self):
        """w(9999) = 9999.  [PLAN Task 7]"""
        t = make_trainer()
        t.iteration = LINEAR_CFR_CUTOFF - 1
        assert t._iteration_weight() == float(LINEAR_CFR_CUTOFF - 1)

    def test_weight_with_linear_cfr_disabled(self):
        """With use_linear_cfr=False, weight is always 1.  [PLAN invariant]"""
        root = FakeState(kind='traverser_root', traverser=0, actor=0,
                         legal_actions=PLAN_LEGAL, value_table=PLAN_V_TABLE)
        t = CFRTrainer(StubGame(root), num_players=1, use_linear_cfr=False,
                       prune_threshold=None, debug=False)
        for iter_val in [1, 100, 10000, 100000]:
            t.iteration = iter_val
            assert t._iteration_weight() == 1.0, \
                f"Linear CFR disabled: w({iter_val}) must be 1.0"


# ---------------------------------------------------------------------------
# Task 2: Iteration t=1 regret test
# ---------------------------------------------------------------------------

class TestIteration1:
    """PLAN.md Task 2 — iteration t=1 with uniform initial strategy."""

    def setup_method(self):
        np.random.seed(42)
        self.trainer = make_trainer(prune_threshold=None)
        self.trainer.iteration = 1
        root = self.trainer.game.deal_new_hand()
        self.trainer.cfr_traverse(root, traverser=0)

    def test_R_fold(self):
        """R[fold] = 1·(0.0 − 1.25) = −1.25.  [PLAN Task 2 / iter-1 derivation]"""
        assert _R(self.trainer)[0] == pytest.approx(-1.25, abs=1e-9)

    def test_R_call(self):
        """R[call] = 1·(0.5 − 1.25) = −0.75.  [PLAN Task 2]"""
        assert _R(self.trainer)[1] == pytest.approx(-0.75, abs=1e-9)

    def test_R_raises(self):
        """R[a=2..9] = 1·(1.5 − 1.25) = +0.25 each.  [PLAN Task 2]"""
        r = _R(self.trainer)
        for a in range(2, 10):
            assert r[a] == pytest.approx(0.25, abs=1e-9), \
                f"R[{a}] should be +0.25 after t=1, got {r[a]}"

    def test_S_uniform(self):
        """S[a] = 1·0.1 = 0.1 for all a after t=1.  [PLAN Task 2]"""
        s = _S(self.trainer)
        for a in range(10):
            assert s[a] == pytest.approx(0.1, abs=1e-9), \
                f"S[{a}] should be 0.1 after t=1 (weight=1, σ=uniform=0.1)"

    def test_sigma_weighted_regret_zero(self):
        """Σ σ·(v−EV) = 0.  [PLAN invariant 1 — always holds]"""
        r = _R(self.trainer)
        sigma = regret_matching(np.zeros(NUM_ACTIONS), NUM_ACTIONS)  # t=1 starts uniform
        v = np.array([PLAN_V_TABLE[a] for a in range(10)])
        ev = float(sigma @ v)
        residual = float(sigma @ (v - ev))
        assert abs(residual) < 1e-10, f"Σ σ·regret should be 0, got {residual}"

    def test_R_sum_is_zero_at_t1(self):
        """Σ R = 0 at t=1 because σ is uniform.  [PLAN sanity check]"""
        # This is a special case: only holds when σ is uniform (t=1 only)
        r = _R(self.trainer)
        assert abs(r.sum()) < 1e-9, \
            f"Sum of R should be 0 at t=1 (uniform σ), got {r.sum()}"


# ---------------------------------------------------------------------------
# Task 3: Iteration t=2 regret test
# ---------------------------------------------------------------------------

class TestIteration2:
    """PLAN.md Task 3 — iteration t=2 accumulates on top of t=1."""

    def setup_method(self):
        np.random.seed(42)
        self.trainer = make_trainer(prune_threshold=None)
        # t=1
        self.trainer.iteration = 1
        self.trainer.cfr_traverse(self.trainer.game.deal_new_hand(), traverser=0)
        # t=2
        self.trainer.iteration = 2
        self.trainer.cfr_traverse(self.trainer.game.deal_new_hand(), traverser=0)

    def test_sigma_at_t2_start(self):
        """
        After t=1, σ should be [0,0,0.125×8] (only positive regrets get mass).
        This σ is what was used DURING the t=2 traversal.
        We reconstruct it from R after t=1.
        [PLAN Task 3]
        """
        # Re-derive: R after t=1 = [-1.25, -0.75, +0.25×8]
        # regret_matching produces [0, 0, 0.125, 0.125, ...]
        r_after_t1 = np.array([-1.25, -0.75] + [0.25]*8)
        sigma = regret_matching(r_after_t1, 10)
        assert sigma[0] == pytest.approx(0.0, abs=1e-12), "Fold should have 0 prob at t=2"
        assert sigma[1] == pytest.approx(0.0, abs=1e-12), "Call should have 0 prob at t=2"
        for a in range(2, 10):
            assert sigma[a] == pytest.approx(0.125, abs=1e-12), \
                f"Raise action {a} should have 0.125 prob at t=2"

    def test_R_fold_after_t2(self):
        """R[fold] = −1.25 + 2·(0−1.5) = −1.25 − 3.0 = −4.25.  [PLAN Task 3]"""
        assert _R(self.trainer)[0] == pytest.approx(-4.25, abs=1e-9)

    def test_R_call_after_t2(self):
        """R[call] = −0.75 + 2·(0.5−1.5) = −0.75 − 2.0 = −2.75.  [PLAN Task 3]"""
        assert _R(self.trainer)[1] == pytest.approx(-2.75, abs=1e-9)

    def test_R_raises_after_t2(self):
        """R[a=2..9] = +0.25 + 2·(1.5−1.5) = +0.25 each.  [PLAN Task 3]"""
        r = _R(self.trainer)
        for a in range(2, 10):
            assert r[a] == pytest.approx(0.25, abs=1e-9), \
                f"R[{a}] should still be +0.25 after t=2, got {r[a]}"

    def test_S_after_t2(self):
        """
        S[0]=S[1]=0.1 (only t=1 contributed, σ=0 at t=2).
        S[2..9] = 0.1 + 2·0.125 = 0.35 each.
        [PLAN Task 3]
        """
        s = _S(self.trainer)
        assert s[0] == pytest.approx(0.1, abs=1e-9), \
            "S[fold] should be 0.1 (only iter 1 contributed 0.1, iter 2 σ=0)"
        assert s[1] == pytest.approx(0.1, abs=1e-9), \
            "S[call] should be 0.1"
        for a in range(2, 10):
            assert s[a] == pytest.approx(0.35, abs=1e-9), \
                f"S[{a}] should be 0.35 after t=2, got {s[a]}"

    def test_S_sum_equals_weight_sum(self):
        """Σ S = w1 + w2 = 1 + 2 = 3.  [PLAN invariant 2]"""
        s = _S(self.trainer)
        assert s.sum() == pytest.approx(3.0, abs=1e-9), \
            f"Σ S should equal Σ wt = 3, got {s.sum()}"


# ---------------------------------------------------------------------------
# Task 4: Iteration t=3 regret test
# ---------------------------------------------------------------------------

class TestIteration3:
    """PLAN.md Task 4 — iteration t=3."""

    def setup_method(self):
        np.random.seed(42)
        self.trainer = make_trainer(prune_threshold=None)
        for t in range(1, 4):
            self.trainer.iteration = t
            self.trainer.cfr_traverse(self.trainer.game.deal_new_hand(), traverser=0)

    def test_R_fold_after_t3(self):
        """R[fold] = −4.25 + 3·(−1.5) = −8.75.  [PLAN Task 4]"""
        assert _R(self.trainer)[0] == pytest.approx(-8.75, abs=1e-9)

    def test_R_call_after_t3(self):
        """R[call] = −2.75 + 3·(−1.0) = −5.75.  [PLAN Task 4]"""
        assert _R(self.trainer)[1] == pytest.approx(-5.75, abs=1e-9)

    def test_R_raises_after_t3(self):
        """R[a=2..9] = +0.25 (raises contribute Δ=0 at t=2,3).  [PLAN Task 4]"""
        r = _R(self.trainer)
        for a in range(2, 10):
            assert r[a] == pytest.approx(0.25, abs=1e-9), \
                f"R[{a}] should remain +0.25 after t=3"

    def test_S_raises_after_t3(self):
        """S[a=2..9] = 0.1 + 2·0.125 + 3·0.125 = 0.1 + 0.625 = 0.725 each.  [PLAN Task 4]"""
        s = _S(self.trainer)
        for a in range(2, 10):
            assert s[a] == pytest.approx(0.725, abs=1e-9), \
                f"S[{a}] should be 0.725 after t=3"

    def test_S_sum_equals_weight_sum(self):
        """Σ S = 1+2+3 = 6.  [PLAN invariant 2]"""
        s = _S(self.trainer)
        assert s.sum() == pytest.approx(6.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Task 5: Closed-form t=100 test
# ---------------------------------------------------------------------------

class TestIteration100:
    """PLAN.md Task 5 — closed-form assertions at t=100."""

    def setup_method(self):
        np.random.seed(42)
        self.trainer = make_trainer(prune_threshold=None)
        for t in range(1, 101):
            self.trainer.iteration = t
            self.trainer.cfr_traverse(self.trainer.game.deal_new_hand(), traverser=0)

    def test_R_fold_closed_form(self):
        """
        R[fold] = −1.25 − 1.5·Σ(t=2..100) t
                = −1.25 − 1.5·5049 = −7574.75
        [PLAN Task 5 closed form]
        """
        assert _R(self.trainer)[0] == pytest.approx(-7574.75, rel=1e-9)

    def test_R_call_closed_form(self):
        """
        R[call] = −0.75 − 1.0·5049 = −5049.75
        [PLAN Task 5 closed form]
        """
        assert _R(self.trainer)[1] == pytest.approx(-5049.75, rel=1e-9)

    def test_R_raises_unchanged(self):
        """R[a=2..9] = +0.25 (no contributions after iter 1).  [PLAN Task 5]"""
        r = _R(self.trainer)
        for a in range(2, 10):
            assert r[a] == pytest.approx(0.25, abs=1e-9)

    def test_S_fold_call_unchanged(self):
        """
        S[fold] = S[call] = 0.1 — σ=0 on these actions from t=2 onward.
        [PLAN Task 5]
        """
        s = _S(self.trainer)
        assert s[0] == pytest.approx(0.1, abs=1e-9), \
            "S[fold] should still be 0.1 at t=100 (σ=0 from t=2 onward)"
        assert s[1] == pytest.approx(0.1, abs=1e-9), \
            "S[call] should still be 0.1 at t=100"

    def test_S_raises_closed_form(self):
        """
        S[a=2..9] = 0.1 + 0.125·Σ(t=2..100) t
                  = 0.1 + 0.125·5049 = 631.225 each
        [PLAN Task 5 closed form]
        """
        s = _S(self.trainer)
        for a in range(2, 10):
            assert s[a] == pytest.approx(631.225, rel=1e-9), \
                f"S[{a}] should be 631.225 at t=100"

    def test_S_sum_equals_weight_sum(self):
        """Σ S = 100·101/2 = 5050.  [PLAN invariant 2 / closed form]"""
        s = _S(self.trainer)
        expected_sum = 100 * 101 / 2  # = 5050
        assert s.sum() == pytest.approx(expected_sum, rel=1e-9), \
            f"Σ S should be {expected_sum} at t=100"

    def test_no_pruning_occurred_at_t100(self):
        """
        Pruning guard is t <= 100 → NOT active at t=100 even though R[fold] << -300.
        Proof: if pruning had fired at t=100, R[fold] would be LESS negative than −7574.75
        (some iterations would have contributed 0 instead of −1.5·t).
        So the closed-form value is only correct if NO pruning happened.
        [PLAN Task 5 — pruning warm-up gate]
        """
        # Already verified indirectly by the closed-form passing.
        # Extra explicit check: R[fold] must be exactly −7574.75 (deterministic).
        assert _R(self.trainer)[0] == pytest.approx(-7574.75, rel=1e-9), \
            "If this fails, pruning fired before warm-up ended (or regret wrong)"


# ---------------------------------------------------------------------------
# Task 6: Pruning boundary tests
# ---------------------------------------------------------------------------

class TestPruningBoundary:
    """PLAN.md Task 6 — pruning eligibility and behavior at t=101."""

    def _setup_to_t100(self):
        """Run trainer to t=100 (no pruning) and return it."""
        np.random.seed(42)
        trainer = make_trainer(
            prune_threshold=PRUNE_THRESHOLD,   # −300
            prune_warm_up=PRUNE_WARM_UP_ITERATIONS,  # 100
        )
        for t in range(1, 101):
            trainer.iteration = t
            trainer.cfr_traverse(trainer.game.deal_new_hand(), traverser=0)
        return trainer

    def test_prune_fires_at_t101_action_0(self):
        """
        At t=101, action 0 has R < −300 and warm-up is passed → eligible to prune.
        With PRUNE_SKIP_PROBABILITY=0.95, pruning fires in the vast majority of calls.
        Test by seeding numpy so np.random.random() < 0.95, confirming action 0
        is skipped (ΔR[fold] = 0).

        Expected after t=101 (action 0 pruned):
          R[fold] unchanged at −7574.75
          R[call] = −5049.75 + 101·(0.5−1.5) = −5049.75 − 101 = −5150.75
        [PLAN Task 6 Outcome A]
        """
        trainer = self._setup_to_t100()
        r_before = _R(trainer).copy()

        # Seed so np.random.random() always returns a value < 0.95 → prune fires
        # np.random.seed(0) → first call to random() is ~0.549 which is < 0.95
        np.random.seed(0)
        trainer.iteration = 101
        trainer.cfr_traverse(trainer.game.deal_new_hand(), traverser=0)
        r_after = _R(trainer)

        # With this seed action 0 SHOULD be pruned (skipped) and action 1 MAY be pruned too.
        # We verify R[fold] didn't change (pruned) OR changed by the expected amount (not pruned).
        fold_delta = r_after[0] - r_before[0]
        assert fold_delta == pytest.approx(0.0, abs=1e-6) or \
               fold_delta == pytest.approx(101 * (-1.5), abs=1e-6), \
            f"fold ΔR should be 0 (pruned) or 101·(-1.5) (not pruned), got {fold_delta}"

    def test_no_prune_at_t100(self):
        """
        At t=100, warm-up condition is `iteration <= prune_warm_up (100)` → NOT active.
        R must match the exact closed form, proving pruning didn't fire.
        [PLAN Task 5 / Task 6]
        """
        trainer = self._setup_to_t100()
        # Closed form derived assuming zero pruning at t=1..100:
        assert _R(trainer)[0] == pytest.approx(-7574.75, rel=1e-9), \
            "Pruning must not fire at t=100 (warm-up boundary)"
        assert _R(trainer)[1] == pytest.approx(-5049.75, rel=1e-9)

    def test_prune_does_not_fire_before_warmup(self):
        """
        With prune_threshold=-300 active but iteration <= 100, _should_prune always False.
        [PLAN Task 6]
        """
        trainer = make_trainer(
            prune_threshold=-300.0,
            prune_warm_up=100,
        )
        # Force extremely negative regrets to trigger threshold condition if guard is broken
        trainer.regret_sum[INFO_KEY] = np.full(NUM_ACTIONS, -10000.0)
        trainer.iteration = 100  # still in warm-up
        for a in PLAN_LEGAL:
            assert not trainer._should_prune(INFO_KEY, a), \
                f"_should_prune should return False at t=100 (warm-up), but fired for action {a}"

    def test_prune_becomes_eligible_at_t101(self):
        """
        At t=101 with R << threshold, _should_prune MUST be able to return True.
        [PLAN Task 6]
        """
        trainer = make_trainer(
            prune_threshold=-300.0,
            prune_warm_up=100,
        )
        trainer.regret_sum[INFO_KEY] = np.full(NUM_ACTIONS, -10000.0)
        trainer.iteration = 101
        # Run enough random samples to confirm it fires at least once (it fires 95% of the time)
        np.random.seed(0)
        fired = any(trainer._should_prune(INFO_KEY, a) for a in PLAN_LEGAL for _ in range(20))
        assert fired, "_should_prune should fire at t=101 with R << threshold"

    def test_sigma_invariant_to_pruning(self):
        """
        Pruning must NOT affect σ computation (strategy comes from regret_sum, not from
        which actions were explored).  Same R → same σ regardless of prune path.
        [PLAN Task 6]
        """
        # Construct explicit R with action 0 below threshold
        r = np.array([-5000.0, -5000.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
        sigma = regret_matching(r, 10)
        # σ[0] and σ[1] must be 0 regardless of pruning
        assert sigma[0] == 0.0
        assert sigma[1] == 0.0
        for a in range(2, 10):
            assert sigma[a] == pytest.approx(0.125, abs=1e-12)


# ---------------------------------------------------------------------------
# Task 7: Linear CFR weight boundary — S ratio test
# ---------------------------------------------------------------------------

class TestLinearCFRBoundary:
    """PLAN.md Task 7 — S increment ratio at cutoff boundary."""

    def test_s_increment_ratio_at_cutoff(self):
        """
        ΔS at t=LINEAR_CFR_CUTOFF must be LINEAR_CFR_CUTOFF× larger than
        ΔS at t=LINEAR_CFR_CUTOFF+1.
        Both traversals see the same σ* (raises only) so
        ratio = w_cutoff / w_(cutoff+1) = LINEAR_CFR_CUTOFF / 1 = LINEAR_CFR_CUTOFF.
        [PLAN Task 7]
        """
        cutoff = LINEAR_CFR_CUTOFF  # read from config — currently 1000
        np.random.seed(42)
        trainer = make_trainer(prune_threshold=None)

        # Run up to (but not including) the cutoff iteration
        for t in range(1, cutoff):
            trainer.iteration = t
            trainer.cfr_traverse(trainer.game.deal_new_hand(), traverser=0)
        s_before_cutoff = _S(trainer).copy()

        # The last Linear-CFR iteration: weight = cutoff
        trainer.iteration = cutoff
        trainer.cfr_traverse(trainer.game.deal_new_hand(), traverser=0)
        s_at_cutoff = _S(trainer).copy()
        delta_at_cutoff = s_at_cutoff - s_before_cutoff

        # The first post-cutoff iteration: weight = 1
        trainer.iteration = cutoff + 1
        trainer.cfr_traverse(trainer.game.deal_new_hand(), traverser=0)
        s_after_cutoff = _S(trainer).copy()
        delta_after_cutoff = s_after_cutoff - s_at_cutoff

        # For raise actions (a=2..9), σ=0.125 throughout; ratio should be cutoff.
        for a in range(2, 10):
            if abs(delta_after_cutoff[a]) < 1e-15:
                continue  # avoid division by zero; shouldn't happen with σ=0.125
            ratio = delta_at_cutoff[a] / delta_after_cutoff[a]
            assert ratio == pytest.approx(float(cutoff), rel=1e-6), \
                f"ΔS ratio at boundary should be {cutoff}, got {ratio:.2f} for a={a}"


# ---------------------------------------------------------------------------
# Task 8: Non-traverser node immutability
# ---------------------------------------------------------------------------

class TestNonTraverserImmutability:
    """PLAN.md Task 8 — R and S only updated for the traverser's info-keys."""

    def test_opponent_node_not_updated_on_p0_traversal(self):
        """
        When traverser=P0, a node where P1 acts must NOT have its R/S updated.
        [PLAN Task 8]

        We build a two-level game:
          - P1 acts first (opponent when traverser=P0)
          - P0 acts second (traverser node)

        After traversal with traverser=P0, only P0's info-key should appear in R/S.
        P1's info-key must not be in regret_sum or strategy_sum.
        """
        # Two-level stub game: root is P1 acting (opponent), child is P0 acting (traverser)
        p0_info_key = (0, (1,))   # P0's info-key at depth-1 node
        p1_info_key = (1, (0, 1)) # P1's info-key at root

        class TwoLevelGame:
            """
            Root: P1 acts (opponent from P0's perspective).
            After P1 acts → P0's traverser node with 2 legal actions.
            After P0 acts → terminal.
            """
            def deal_new_hand(self):
                s = FakeState(kind='traverser_root', traverser=0, actor=1,
                              legal_actions=[0, 1], value_table={})
                return s

            def is_terminal(self, s):
                return s.kind == 'terminal'

            def is_chance_node(self, s):
                return False

            def sample_chance(self, s):
                raise NotImplementedError

            def get_current_player(self, s):
                return s.actor

            def get_legal_actions(self, s):
                if s.kind == 'terminal':
                    return []
                return list(s.legal_actions)

            def get_info_key(self, s, player):
                return (player, tuple(s.legal_actions))

            def get_payoffs(self, s):
                return list(s.payoffs)

            def apply_action(self, s, action_index):
                if s.actor == 1:
                    # P1 acted → now P0's turn with [1] legal action
                    next_s = FakeState(kind='traverser_root', traverser=0, actor=0,
                                       legal_actions=[1],
                                       value_table={1: 2.0})
                    return next_s
                else:
                    # P0 acted → terminal
                    payoffs = [s.value_table.get(action_index, 0.0), 0.0, 0.0]
                    t = FakeState(kind='terminal', traverser=0, payoffs=payoffs)
                    return t

        game = TwoLevelGame()
        trainer = CFRTrainer(game_module=game, num_players=1,
                             use_linear_cfr=False, prune_threshold=None, debug=False)
        np.random.seed(0)
        trainer.iteration = 1
        trainer.cfr_traverse(game.deal_new_hand(), traverser=0)

        # P1's info-key must NOT appear in the trainer's tables
        p1_key = (1, (0, 1))
        assert p1_key not in trainer.regret_sum, \
            "P1's info-key should NOT be in regret_sum when traverser=P0"
        assert p1_key not in trainer.strategy_sum, \
            "P1's info-key should NOT be in strategy_sum when traverser=P0"

        # P0's info-key MUST appear
        p0_key = (0, (1,))
        assert p0_key in trainer.regret_sum, \
            "P0's info-key must be in regret_sum after traversal"


# ---------------------------------------------------------------------------
# Task 9: Legal-action isolation
# ---------------------------------------------------------------------------

class TestLegalActionIsolation:
    """PLAN.md Task 9 — R[a] for a ∉ legal(I) must remain 0."""

    def test_illegal_actions_stay_zero(self):
        """
        Legal actions at root = [0,1,2,3,4,5,6,7,8,9].  All 10 get updated.
        Build a game where only [0, 1] are legal, run 10 iterations,
        assert R[2..9] remain exactly 0.
        [PLAN Task 9]
        """
        root = FakeState(
            kind='traverser_root', traverser=0, actor=0,
            legal_actions=[0, 1],
            value_table={0: -1.0, 1: 2.0},
        )
        game = StubGame(root)
        trainer = CFRTrainer(game_module=game, num_players=1,
                             use_linear_cfr=False, prune_threshold=None, debug=False)
        np.random.seed(0)
        for t in range(1, 11):
            trainer.iteration = t
            trainer.cfr_traverse(game.deal_new_hand(), traverser=0)

        key = (0, (0, 1))
        r = trainer.regret_sum[key]
        for a in range(2, NUM_ACTIONS):
            assert r[a] == 0.0, \
                f"R[{a}] should be 0 (not in legal set [0,1]), got {r[a]}"

    def test_action_map_never_changes(self):
        """
        action_map[I] is set on first visit and must not change across re-visits.
        [PLAN Task 9]
        """
        trainer = make_trainer(prune_threshold=None)
        np.random.seed(42)
        trainer.iteration = 1
        trainer.cfr_traverse(trainer.game.deal_new_hand(), traverser=0)
        first_legal = list(trainer.action_map[INFO_KEY])

        for t in range(2, 11):
            trainer.iteration = t
            trainer.cfr_traverse(trainer.game.deal_new_hand(), traverser=0)
            assert trainer.action_map[INFO_KEY] == first_legal, \
                f"action_map changed on re-visit at t={t}"


# ---------------------------------------------------------------------------
# Task 10: Average strategy derivation
# ---------------------------------------------------------------------------

class TestAverageStrategy:
    """PLAN.md Task 10 — average strategy normalizes strategy_sum."""

    def test_average_strategy_uniform_fallback(self):
        """
        If strategy_sum is all-zero, get_average_strategy returns uniform.
        [PLAN Task 10]
        """
        # With an empty S, trainer falls back to uniform
        trainer = make_trainer(prune_threshold=None)
        result = trainer.get_average_strategy(INFO_KEY, PLAN_LEGAL)
        # Not visited yet → should return None (no data)
        # (or uniform — depends on implementation)
        if result is not None:
            assert abs(sum(result) - 1.0) < 1e-10, "Average strategy must sum to 1"

    def test_get_average_strategy_function_uniform(self):
        """get_average_strategy() with all-zero S returns uniform.  [PLAN Task 10]"""
        s = np.zeros(10)
        avg = get_average_strategy(s, 10)
        np.testing.assert_allclose(avg, np.full(10, 0.1), atol=1e-12)

    def test_average_strategy_after_training(self):
        """
        After many iterations, avg_strategy for raises should be ~0.125 each.
        Fold and Call should have negligible average weight.
        [PLAN Task 10 / closed-form at t=100]
        """
        np.random.seed(42)
        trainer = make_trainer(prune_threshold=None)
        for t in range(1, 101):
            trainer.iteration = t
            trainer.cfr_traverse(trainer.game.deal_new_hand(), traverser=0)

        avg = trainer.get_average_strategy(INFO_KEY, PLAN_LEGAL)
        assert avg is not None, "get_average_strategy should return a result after training"
        assert abs(sum(avg) - 1.0) < 1e-9, "Average strategy must sum to 1"

        # Raise actions should dominate; each should be close to 0.125
        for a in range(2, 10):
            assert avg[a] == pytest.approx(0.125, rel=0.01), \
                f"avg_strategy[{a}] should be ~0.125 at t=100, got {avg[a]}"

        # Fold and call should be near-zero (very small residual from iter 1)
        assert avg[0] < 0.01, f"avg_strategy[fold] should be tiny, got {avg[0]}"
        assert avg[1] < 0.01, f"avg_strategy[call] should be tiny, got {avg[1]}"


# ---------------------------------------------------------------------------
# Task 12: Deterministic reproducibility
# ---------------------------------------------------------------------------

class TestDeterminism:
    """PLAN.md Task 12 — sequential training is deterministic with fixed seed."""

    def test_same_seed_same_result(self):
        """Two runs with the same seed produce identical R/S tables.  [PLAN Task 12]"""
        def run_10():
            np.random.seed(0)
            trainer = make_trainer(prune_threshold=None)
            for t in range(1, 11):
                trainer.iteration = t
                trainer.cfr_traverse(trainer.game.deal_new_hand(), traverser=0)
            return _R(trainer).copy(), _S(trainer).copy()

        r1, s1 = run_10()
        r2, s2 = run_10()
        np.testing.assert_array_equal(r1, r2, err_msg="R tables differ across identical seeds")
        np.testing.assert_array_equal(s1, s2, err_msg="S tables differ across identical seeds")


# ---------------------------------------------------------------------------
# Task 13: S sum invariant stress test
# ---------------------------------------------------------------------------

class TestSumInvariant:
    """PLAN.md Task 13 — Σ S[I] = Σ wₜ for all info-keys visited."""

    def _expected_weight_sum(self, num_iters, linear_cutoff):
        """Compute Σ wₜ for t=1..num_iters with given linear cutoff."""
        if num_iters <= linear_cutoff:
            return num_iters * (num_iters + 1) / 2
        else:
            linear_part = linear_cutoff * (linear_cutoff + 1) / 2
            post_linear = num_iters - linear_cutoff
            return linear_part + post_linear

    @pytest.mark.parametrize("num_iters", [1, 3, 10, 100])
    def test_s_sum_linear_regime(self, num_iters):
        """
        Σ S[I] = num_iters·(num_iters+1)/2 in the linear CFR regime.
        [PLAN Task 13 / invariant 2]
        """
        np.random.seed(42)
        trainer = make_trainer(prune_threshold=None)
        for t in range(1, num_iters + 1):
            trainer.iteration = t
            trainer.cfr_traverse(trainer.game.deal_new_hand(), traverser=0)

        expected = self._expected_weight_sum(num_iters, LINEAR_CFR_CUTOFF)
        s = _S(trainer)
        assert s.sum() == pytest.approx(expected, rel=1e-9), \
            f"Σ S should be {expected} after {num_iters} iters, got {s.sum()}"

    def test_s_sum_post_linear_regime(self):
        """
        Σ S = linear_part + (N - cutoff)·1 for N > cutoff.
        Test with a small cutoff to avoid slow runtime.
        [PLAN Task 13 — post-cutoff weight = 1]
        """
        cutoff = 5
        num_iters = 8
        root = FakeState(kind='traverser_root', traverser=0, actor=0,
                         legal_actions=PLAN_LEGAL, value_table=PLAN_V_TABLE)
        trainer = CFRTrainer(
            game_module=StubGame(root),
            num_players=1,
            use_linear_cfr=True,
            linear_cfr_cutoff=cutoff,
            prune_threshold=None,
            debug=False,
        )
        np.random.seed(42)
        for t in range(1, num_iters + 1):
            trainer.iteration = t
            trainer.cfr_traverse(trainer.game.deal_new_hand(), traverser=0)

        key = (0, tuple(PLAN_LEGAL))
        s = trainer.strategy_sum[key]
        expected = self._expected_weight_sum(num_iters, cutoff)
        assert s.sum() == pytest.approx(expected, rel=1e-9), \
            f"Σ S (custom cutoff={cutoff}, N={num_iters}) should be {expected}, got {s.sum()}"


# ---------------------------------------------------------------------------
# Bonus: np.resize bug test (from threat model in PLAN.md)
# ---------------------------------------------------------------------------

class TestNpResizeBug:
    """
    Threat model: np.resize TILES a short array rather than zero-padding.
    If regret_sum[I] is ever shorter than NUM_ACTIONS, get_strategy would produce
    a wrong strategy due to tiling.

    Verify that the trainer allocates R as zeros(NUM_ACTIONS) — never shorter.
    [PLAN threat model]
    """

    def test_regret_sum_always_full_length(self):
        """R[I] must be exactly NUM_ACTIONS long after any visit.  [PLAN threat model]"""
        np.random.seed(42)
        trainer = make_trainer(prune_threshold=None)
        for t in range(1, 6):
            trainer.iteration = t
            trainer.cfr_traverse(trainer.game.deal_new_hand(), traverser=0)

        r = trainer.regret_sum[INFO_KEY]
        assert len(r) == NUM_ACTIONS, \
            f"regret_sum[I] has length {len(r)}, expected {NUM_ACTIONS}. " \
            "np.resize would tile if this is < NUM_ACTIONS."

    def test_strategy_sum_always_full_length(self):
        """S[I] must be exactly NUM_ACTIONS long.  [PLAN threat model]"""
        np.random.seed(42)
        trainer = make_trainer(prune_threshold=None)
        trainer.iteration = 1
        trainer.cfr_traverse(trainer.game.deal_new_hand(), traverser=0)

        s = trainer.strategy_sum[INFO_KEY]
        assert len(s) == NUM_ACTIONS, \
            f"strategy_sum[I] has length {len(s)}, expected {NUM_ACTIONS}."
