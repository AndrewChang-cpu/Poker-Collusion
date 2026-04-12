# Phase 1: CFR Regret/Strategy Update Correctness — TDD Verification Plan

## Phase Goal
Verify, through test-driven development, that the MCCFR regret and strategy updates in `poker_collusion/cfr/trainer.py` are mathematically correct and consistent with the Pluribus paper (Brown & Sandholm 2019). Cover all critical training milestones: Linear CFR regime, post-cutoff, pruning boundary, and average strategy derivation.

---

## Background & Mathematical Specification

This section contains the authoritative math for every test case. All test assertions must trace back to these derivations.

### Conventions (from codebase)
- **Variant:** External-sampling MCCFR. Traverser visits every legal action; opponents are sampled one action from σ.
- **Strategy:** σ(I, a) = regret_matching over legal actions only. r⁺ₐ = max(R(I,a), 0); σ = r⁺ / Σr⁺ if Σr⁺ > 0 else uniform.
- **Counterfactual value:** v(I) = Σₐ σ(I,a)·v(I,a) where v(I,a) is the scalar traverser payoff returned by recursion.
- **Regret update:** ΔR(I,a) = wₜ·(v(I,a) − v(I)) for unpruned a; 0 for pruned a.
- **Strategy-sum update:** ΔS(I,a) = wₜ·σ(I,a) for ALL legal a (pruning does not suppress S update; σ[pruned_a]=0 makes it vacuous).
- **Linear CFR weight:** wₜ = t if t ≤ 10,000 (LINEAR_CFR_CUTOFF), wₜ = 1 if t > 10,000.
- **Payoffs:** Net profit in BB. STARTING_STACK = 20 BB. SB=0.5, BB=1.0.
- **Pruning:** Skip action a if t > 100 (PRUNE_WARM_UP) AND R(I,a) < −300 AND Bernoulli(0.95) = True.

### Key Invariants
1. **σ-weighted regret sums to zero:** Σₐ σ(I,a)·(v(I,a)−v(I)) = 0 at every traverser node.
2. **S total equals sum of weights:** Σₐ S(I,a) = Σ wₛ over all iterations s where I was visited by its traverser.
3. **Non-traverser nodes untouched:** R[I] and S[I] must not change on a traversal where the actor at I ≠ traverser.
4. **Legal-action isolation:** R[I][a] for a ∉ legal(I) must remain 0 across all visits to I.

---

## Worked Examples (Hard-Coded Expected Values for Tests)

### Scenario Setup
- **3-player NLHE, 20 BB stacks.** P0=BTN, P1=SB, P2=BB.
- **Traverser = P0** (BTN) for all examples unless noted.
- **Info-key I₀** = P0's first preflop decision node (bucket = premium hand, e.g. AA).
- **Legal actions at I₀:** [0 (Fold), 1 (Call 1BB), 2 (Raise 2BB), 3 (2.5BB), 4 (3BB), 5 (4BB), 6 (5BB), 7 (8BB), 8 (12BB), 9 (All-in 20BB)] → 10 actions.
- **Stipulated downstream values (fixed by RNG seed in tests):**
  - Downstream for all non-fold actions by P1 and P2: both fold.
  - v(I₀, 0) = 0.0 (P0 folds, no chips in pot preflop)
  - v(I₀, 1) = 0.5 (call, stipulated downstream result)
  - v(I₀, 2..9) = 1.5 each (any raise → P1/P2 fold → P0 wins SB+BB=1.5)

### Iteration t=1 (Linear CFR, w₁=1)

**Input state:** R[I₀] empty (first visit). σ = uniform = [0.1]×10.

**Calculations:**
```
v(I₀) = 0.1·0 + 0.1·0.5 + 8·0.1·1.5 = 0 + 0.05 + 1.2 = 1.25
r̃[a] = v(I₀,a) − v(I₀):
  a=0: 0.0 − 1.25 = −1.25
  a=1: 0.5 − 1.25 = −0.75
  a=2..9: 1.5 − 1.25 = +0.25 each
ΔR = 1 · r̃
```

**Expected state after t=1:**
```
R[I₀] = [−1.25, −0.75, +0.25, +0.25, +0.25, +0.25, +0.25, +0.25, +0.25, +0.25]
S[I₀] = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
```

**Sanity check:** Σ R[I₀] = −1.25 − 0.75 + 8·0.25 = −2 + 2 = 0 ✓ (holds only at t=1 because σ is uniform).

### Iteration t=2 (Linear CFR, w₂=2)

**Input state:** R[I₀] from t=1. Strategy from regret matching:
```
r⁺ = [0, 0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
Σr⁺ = 2.0
σ = [0, 0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
```

**Calculations:**
```
v(I₀) = 0·0 + 0·0.5 + 8·0.125·1.5 = 1.5
r̃[a]:
  a=0: 0.0 − 1.5 = −1.5
  a=1: 0.5 − 1.5 = −1.0
  a=2..9: 1.5 − 1.5 = 0.0
ΔR = 2 · r̃ = [−3.0, −2.0, 0, 0, 0, 0, 0, 0, 0, 0]
```

**Expected state after t=2:**
```
R[I₀] = [−4.25, −2.75, +0.25, +0.25, +0.25, +0.25, +0.25, +0.25, +0.25, +0.25]
S[I₀] = [0.1, 0.1, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35]
```
Σ S[I₀] = 0.1+0.1+8·0.35 = 3.0 = w₁+w₂ = 1+2 = 3 ✓

**Average strategy after t=2:** S/3 = [0.0333, 0.0333, 0.1167×8]

### Iteration t=3 (Linear CFR, w₃=3)

σ unchanged from t=2 (R positive entries still 0.25).

**Calculations:**
```
v(I₀) = 1.5 (same)
r̃ = [−1.5, −1.0, 0, ..., 0]
ΔR = 3 · r̃ = [−4.5, −3.0, 0, ..., 0]
```

**Expected state after t=3:**
```
R[I₀] = [−8.75, −5.75, +0.25, +0.25, +0.25, +0.25, +0.25, +0.25, +0.25, +0.25]
S[I₀][0] = 0.1, S[I₀][1] = 0.1, S[I₀][2..9] = 0.725 each
```
Σ S[I₀] = 6.0 = w₁+w₂+w₃ ✓

### Closed Form at t=100 (last warm-up iteration, pruning NOT yet active)

Linear CFR regime. σ fixed at σ* = [0,0,0.125,...,0.125] from t=2 onward.

```
R[I₀][0] = −1.25 + (−1.5)·Σₜ₌₂¹⁰⁰ t
          = −1.25 + (−1.5)·(5050−1)
          = −1.25 − 7573.5 = −7574.75

R[I₀][1] = −0.75 + (−1.0)·5049 = −5049.75

R[I₀][2..9] = +0.25 each

S[I₀][0] = S[I₀][1] = 0.1
S[I₀][2..9] = 0.1 + 0.125·5049 = 631.225 each

Σ S[I₀] = 0.2 + 8·631.225 = 5050.0 = 100·101/2 ✓
```

**Critical test:** At t=100, action 0 has R = −7574.75 < −300, but warm-up condition is `t <= 100` so pruning does NOT trigger. Assert no action skipped.

### Pruning Boundary: t=101

**Both actions 0 and 1 are now eligible for pruning** (R[I₀][0] < −300, R[I₀][1] < −300, and t=101 > 100).

**Outcome A (action 0 pruned, prob 0.95):** ΔR[I₀][0] = 0; action 0 not recursed.
```
ΔR = 101 · [0, −1.0, 0, ..., 0]
R[I₀][0] unchanged at −7574.75
R[I₀][1] = −5049.75 − 101 = −5150.75
```

**Outcome B (action 0 not pruned, prob 0.05):**
```
ΔR = 101 · [−1.5, −1.0, 0, ..., 0]
R[I₀][0] = −7574.75 − 151.5 = −7726.25
R[I₀][1] = −5049.75 − 101 = −5150.75
```

### Linear CFR Boundary: t=10,000 → t=10,001

**t=10,000:** wₜ = 10,000 (last Linear CFR iteration).
**t=10,001:** wₜ = 1 (first post-cutoff iteration, 4-order-of-magnitude weight drop).

```
Σ S[I₀][2..9] at t=10,000 (each):
  = 0.1 + 0.125 · Σₜ₌₂¹⁰⁰⁰⁰ t
  = 0.1 + 0.125 · 50,004,999
  = 6,250,624.975

Σ S[I₀] = 0.2 + 8·6,250,624.975 = 50,005,000.0 = 10000·10001/2 ✓

Average raise probability = 6,250,624.975 / 50,005,000 = 0.125000 (iter-1 influence < 10⁻⁸)
```

ΔS at t=10,001 = 1 · σ* = [0, 0, 0.125, ..., 0.125] (weight is 1, not 10,001).

---

## Tasks

### Task 1: Create test infrastructure
- [ ] Create `tests/test_cfr_regret_correctness.py`
- [ ] Implement a `StubGame` that returns hard-coded v values given (state, traverser) — makes downstream recursion deterministic without needing a real deal
- [ ] Implement `make_traverser_state(traverser, legal_actions, v_table)` helper that produces a fake NLHEState with known legal actions and a stub that returns fixed values from recursion

### Task 2: Iteration t=1 regret test
- [ ] Assert R[I₀] = [−1.25, −0.75, +0.25×8] after one traversal with uniform σ and v = [0, 0.5, 1.5×8]
- [ ] Assert S[I₀] = [0.1×10]
- [ ] Assert Σₐ σₐ·(v(a)−v(I₀)) = 0 (weighted regret zero-sum invariant)

### Task 3: Iteration t=2 regret test
- [ ] Assert σ at start of t=2 = [0, 0, 0.125×8] (regret matching from t=1 R)
- [ ] Assert R[I₀] = [−4.25, −2.75, +0.25×8] after t=2 traversal
- [ ] Assert S[I₀] = [0.1, 0.1, 0.35×8]
- [ ] Assert Σ S = 3 (sum of weights)

### Task 4: Iteration t=3 regret test
- [ ] Assert R[I₀] = [−8.75, −5.75, +0.25×8]
- [ ] Assert S[I₀][2..9] = 0.725 each, S total = 6

### Task 5: Closed-form t=100 test
- [ ] Run 100 iterations with fixed RNG seed and stipulated v table
- [ ] Assert R[I₀][0] ≈ −7574.75 (within float tolerance)
- [ ] Assert R[I₀][1] ≈ −5049.75
- [ ] Assert R[I₀][2..9] each ≈ 0.25
- [ ] Assert Σ S[I₀] ≈ 5050 (= 100·101/2)
- [ ] Assert no pruning occurred at t=100 even though R[I₀][0] < −300 (warm-up guard)

### Task 6: Pruning boundary test
- [ ] At t=101 with RNG returning 0.0 (always prune): assert R[I₀][0] unchanged, R[I₀][1] updated
- [ ] At t=101 with RNG returning 0.99 (never prune): assert both R[I₀][0] and R[I₀][1] updated
- [ ] Assert that in both outcomes S[I₀] increments by w₁₀₁·σ* = [0, 0, 101·0.125, ...]
- [ ] Assert σ is identical in both prune/no-prune paths (pruning does not affect σ computation)

### Task 7: Linear CFR weight boundary test
- [ ] Assert `_iteration_weight(10000)` = 10000.0
- [ ] Assert `_iteration_weight(10001)` = 1.0
- [ ] Run one traversal at t=10000, one at t=10001; assert S increment ratio = 10000:1

### Task 8: Non-traverser node immutability test
- [ ] Run a traversal with traverser=P0 on a game where P1 acts first
- [ ] Assert R[I_P1] and S[I_P1] are unchanged after traversal (only R[I_P0] updated)
- [ ] Conversely run traversal with traverser=P1; assert only I_P1 updated

### Task 9: Legal-action isolation test
- [ ] Assert R[I][a] for a ∉ legal(I) remains 0 after 100 iterations
- [ ] Assert that `action_map[I]` never changes across re-visits to the same info-key

### Task 10: Average strategy derivation test
- [ ] After running to t=10000, assert average_strategy[I₀][0] ≈ 0 and average_strategy[I₀][2..9] ≈ 0.125 each
- [ ] Assert uniform fallback: if S[I] = all zeros → get_average_strategy returns 1/|legal| for each legal action

### Task 11: Regret matching unit tests
- [ ] All-negative R → uniform σ
- [ ] All-zero R → uniform σ
- [ ] Mixed R → only positive entries get mass, proportional to r⁺
- [ ] Single positive entry → σ = [0,...,1,...,0]

### Task 12: Sequential vs deterministic reproducibility test
- [ ] Fix `np.random.seed(0)`, run `train(num_iterations=10)`, capture R/S
- [ ] Run again with same seed, assert identical R/S (deterministic)

### Task 13: S sum invariant stress test
- [ ] After any N iterations, assert Σₐ S[I][a] = Σₜ wₜ for each visited info-key I (both linear and post-cutoff regimes)

---

## Threat Model / Edge Cases

| Risk | Mitigation |
|---|---|
| Sign flip in regret update (ev−values instead of values−ev) | Task 2: fold has most-negative regret; any sign flip makes fold most positive |
| Linear CFR weight uses `<` instead of `<=` at cutoff | Task 7: boundary assertions at 10000 and 10001 |
| `np.resize` tiling on short R arrays | Test with R length < NUM_ACTIONS; assert zero-padding not tiling |
| Pruning suppressing S update (it shouldn't; σ[pruned]=0 anyway) | Task 6: explicitly assert S increment is σ-based regardless of prune |
| Non-traverser node getting R update | Task 8: cross-traverser immutability |
| Action map drift (same I, different legal set at re-visit) | Task 9: legal-action isolation |
| Parallel vs sequential weight divergence | Separate: parallel uses per-job RNG; sequential uses global np.random |

---

## Files to Create/Modify

| File | Action | Notes |
|---|---|---|
| `tests/test_cfr_regret_correctness.py` | Create | All tasks above; deterministic via seed control |
| `tests/conftest.py` | Create (if absent) | Shared StubGame fixture |

Do **not** modify any production code in this phase — pure test writing only.

---

## Definition of Done

- [ ] All 13 task groups pass with `pytest tests/test_cfr_regret_correctness.py -v`
- [ ] Every assertion has a comment tracing it to a specific worked example in this plan
- [ ] No test is longer than ~50 lines (split into fixtures if needed)
- [ ] `pytest` exits with code 0 and zero warnings on the modified files

---

## References
- Brown & Sandholm (2019), *Superhuman AI for multiplayer poker* (Pluribus paper)
- `poker_collusion/cfr/trainer.py` — `cfr_traverse`, `_iteration_weight`, `_should_prune`
- `poker_collusion/cfr/strategy.py` — `regret_matching`, `get_average_strategy`
- `poker_collusion/config.py` — `LINEAR_CFR_CUTOFF=10000`, `PRUNE_THRESHOLD=−300`, `PRUNE_WARM_UP_ITERATIONS=100`, `PRUNE_SKIP_PROBABILITY=0.95`
