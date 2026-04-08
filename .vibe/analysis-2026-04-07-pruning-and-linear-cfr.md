# Blueprint v3 underperformance — root cause analysis

**Date:** 2026-04-07
**Subject:** Why `blueprint_claude_v3_5000.pkl` loses to the amateur baseline despite the v2 prune-EV bug being fixed and tests verifying CFR correctness.

---

## TL;DR

The CFR algorithm itself is correct, but the trained artifact is essentially noise stamped with an aggressive policy by random chance. Two compounding issues dominate:

1. **Severe undertraining + lossy abstraction** → ~87% of strategies are still uniform after 5,000 iterations.
2. **Misconfigured `LINEAR_CFR_CUTOFF` and `PRUNE_THRESHOLD`** → the first 1,000 iterations carry ~83% of cumulative regret weight, and pruning freezes actions on a single noisy sample.

This document captures the full investigation, including the math on why the two CFR hyperparameters need to change and what good values look like.

---

## Part 1 — Evidence the blueprint is essentially untrained

### Strategy entropy is near-uniform

Loaded `blueprint_claude_v3_5000.pkl` and computed normalized entropy of every average-strategy distribution:

```
Strategy entropy (normalized 0..1): mean=0.950  median=1.000
% strategies near uniform (entropy>0.85):  87.3%
% strategies decisive (top action >50%):   10.8%
% strategies very decisive (top action >75%):  3.4%
infosets with >90% mass on a single action: 631 / 59855 = 1.1%
```

The *median* infoset's average strategy is **perfectly uniform over its legal actions**. Only 1% of infosets have actually picked an action.

### Visit counts explain the uniformity

```
Total infosets: 59,855  (preflop 47,098 / flop 9,832 / turn 2,925 / river 0)
```

- **River = 0**: across 5,000 iterations × 3 traversers = 15,000 traversals, **no traverser ever made a river decision once**. With 20bb stacks, raising 2–12 bb preflop and a single pot-sized flop bet usually puts someone all-in by the flop, so the trainer never accumulates *any* river data.
- **Preflop root opens (BTN, hand id only, empty history)**: 169 distinct infosets, visited ~30 times each (`15000/3 traversals × 1/3 chance you're BTN ÷ 169 hands`). 30 weighted updates is far too few for regret matching to differentiate hands.

### Concrete BTN-open strategies (canonical hand id from `hole_to_canonical`)

```
hand   0  avg = [.33  .   .01  .   .01  .   .   .   .   .65]   Fold 33% / All-in 65%
hand   1  avg = [ .   .  .54  .   .   .12  .   .  .32 .01]    Raise2 54 / Raise5 12 / Raise12 32
hand   2  avg = [.68  .  .14 .01 .02  .   .  .14  .  .02]    Fold 68 / Raise2 14 / Raise8 14
hand   4  avg = [ .  .02  .  .04  .  .04 .01 .12 .02 .72]    All-in 72%
hand  13  avg = [.75 .01 .05 .06 .01 .07 .01 .01 .01 .01]    Fold 75
hand  14  avg = [.93  .   .   .  .01 .03 .01 .01  .   . ]    Fold 93
hand  19  avg = [ . .03 .23  .  .58  .  .01 .02  .  .12]    Raise3 58
```

There is no monotonic relationship between hand strength and action. Hand 0 (likely 22) open-shoves 65% of the time. Hand 19 (a suited connector) prefers a 3bb open. These distributions are not a converged strategy; they're the snapshot of where regret-matching landed after ~30 weighted updates.

---

## Part 2 — Stepping through one traversal

Command: `python3 scripts/train.py --iterations 1 --debug --load output/blueprint_claude_v3_5000.pkl`

### Traversal 1: P0 (BTN) with Q♣5♦, opp deals SB J♥3♦, BB 6♥2♥

At depth 0 (root preflop, hand id only) the displayed live strategy is:
```
[0]Fold 41%   [2]Raise 2BB 59%   (everything else 0%)
```
On exit the regret table updates:
```
Fold        -9.992
Check/Call   0
Raise 2BB   +6.946
[3..9]       0  ← all 7 raise sizes are PRUNED for this traversal
```
The exploration tree printed by the debugger shows it explicitly:
```
[0]✓Fold      [1] PRUNED      [2]✓Raise 2BB
[3] PRUNED ... [9] PRUNED
```
At iter 5001 with weight 1, the BTN-open root infoset has only 3 of 10 actions explored this iteration because the others were pruned. The 5% revisit rate is adequate at 10⁶ iterations, not at 5×10³.

### Traversal 2: P1 (SB) facing P0 BTN open-shove (Raise 12BB) with 8♠J♥

Live strategy displayed: `[0]Fold 13% [1]Call 87% [9]All-in 0%`. Calling a 12bb open-shove with J8o for 60% of effective stack — clear leak. The blueprint thinks this is +EV because the post-call subgame happens to land on T♠9♣7♣ (open-ended straight draw), and the *single sample* of subgame play returned +32 BB.

### Traversal 3: P2 (BB) facing P0 BTN raise to 3BB, P1 fold, with 7♥6♣

Live strategy: `[1]Call 38% [7]Raise 8BB 11% [8]Raise 12BB 7% [9]All-in 44%`

**44% all-in 3-bet shove with 76o offsuit facing a single min-raise.** Look at the EVs the trainer recorded for the same node:
```
Action       Value     Δ regret_sum
Check/Call   +20.500    +1.9
Raise 8BB    +3.500    -15.1
Raise 12BB  +20.500     ...
All-in       +20.500    ...
```

EVERY non-fold action returns the same +20.5 BB value, because the chance cards are pre-dealt at the start of the hand and one of P2's cards (6♣) makes a winning hand on the runout. **Single-sample external sampling with no variance reduction.** With 5,000 iterations and one sample per node, almost every recorded EV is dominated by which 5 board cards got dealt, not by the strategic merit of the action.

---

## Part 3 — Information abstraction is silently throwing away critical context

In `poker_collusion/abstraction/info_set.py:39-42`:

```python
# Find the start of the current street: the position after the last DEAL.
last_deal = -1
for i, a in enumerate(state.action_history):
    if a == _DEAL: last_deal = i
street_history = state.action_history[last_deal + 1:]
return (round_idx, bucket, tuple(street_history))
```

The infoset key is `(round_idx, bucket, current_street_history)`. **All prior-street action history is dropped.** Comment: "to keep the infoset space tractable."

Consequences:

- A flop infoset for "(flop bucket 17, no actions yet this street)" represents *both* a 3-way limped pot (pot = 3 BB, SPR ≈ 6) *and* a heads-up 4-bet pot (pot = 24 BB, SPR ≈ 0.4). The same regret/strategy entry has to play both.
- The infoset has no encoding of how many opponents are still in the hand. Heads-up vs 3-way is invisible.
- Stack-to-pot ratio is invisible. 20BB-stack NLHE is *all about SPR* — the actions on the flop chart are pot multipliers, but the strategy doesn't know the pot size.
- Position is only implicit via the postflop acting order; on the flop the SB acts first, then BB, then BTN — but the strategy table can't tell the difference between "I'm SB facing two opponents" and "I'm BTN closing action against two checks" because the action history is the same `()` until someone bets.

This is the **biggest lever**. Even with 100× more iterations the algorithm would converge to a *bad* equilibrium because the abstraction is too lossy. Adding `(num_active, pot_in_bb_bucket, position)` to the infoset key would more than 10× the infoset count but is essential. The amateur policy in `evaluation/amateur_policy.py:108-128` reads stack/pot/players directly from `state` — it has every signal the blueprint is missing.

---

## Part 4 — Deep dive: `LINEAR_CFR_CUTOFF` and `PRUNE_THRESHOLD`

These two parameters are *coupled*. You can't reason about one without the other, because the pruning threshold has to be calibrated against the magnitude of cumulative regret, and the Linear-CFR weight schedule is what determines that magnitude.

### Background

#### Linear CFR weighting

From `trainer.py:69-79`:
```python
def _iteration_weight(self, t):
    if self.use_linear_cfr and t <= self.linear_cfr_cutoff:
        return float(t)
    return 1.0
```

- `w_t = t` for `t ≤ cutoff`
- `w_t = 1` afterwards

Theoretical motivation (Brown & Sandholm 2019): early iterations have noisy strategies (uniform-ish) and noisy regret estimates, so their contribution should be downweighted relative to later, more refined iterations. The cutoff is a practical hack to cap the weight so it doesn't grow without bound.

#### Regret pruning

From `trainer.py:188-195`:
```python
def _should_prune(self, info_key, action):
    if t <= warm_up: return False
    regrets = regret_sum.get(info_key, …)
    if regrets[action] < prune_threshold:
        return random < prune_skip_prob   # 0.95
    return False
```

Skip exploring deeply-negative-regret actions 95% of the time. Only a *speed* optimization; never improves convergence quality.

### The math linking the two

Single-iteration regret update at iteration `t` is bounded by `w_t × Δmax`, where `Δmax = 2 × STARTING_STACK_BB = 40` for this game.

Maximum possible cumulative regret magnitude under Linear CFR:
```
|R_t|_max  ≈  Δmax × Σ_{i=1..t} w_i
            =  Δmax × t(t+1)/2
            ≈  Δmax × t²/2
```

Typical magnitude (assuming approximately random walk):
```
σ(R_t)  ≈  Δmax × t^(3/2) / √3
```

(For comparison, *vanilla* CFR has `σ(R_t) ≈ Δmax × √t`.)

**Pruning should kick in when an action's cumulative regret is many `σ(R_t)` below zero**, evaluated at the iteration where pruning turns on (right after warm-up).

### Why current values fail

Current config:
```
LINEAR_CFR_CUTOFF      = 1_000
PRUNE_THRESHOLD        = -300
PRUNE_WARM_UP          = 100
T_MAX_DEFAULT          = 100_000
```

**At iteration 100 (right after warm-up):**
- Max single-iter update: `40 × 100 = 4,000`. **Single bad sample at iter 100 pushes regret 13× past the −300 threshold instantly.**
- Typical regret std: `40 × 100^(3/2) / √3 ≈ 23,000`. The threshold of −300 is `−0.013σ` — basically zero.

**At iteration 1000 (end of linear phase):**
- Max single-iter update: `40 × 1000 = 40,000`. Threshold is **133× smaller** than a single update.
- Typical regret std: `40 × 1000^(3/2) / √3 ≈ 730,000`. Threshold is `−4×10⁻⁴ σ`.

**Empirically confirmed on the loaded blueprint:**
```
regret stats: min=−125,770   max=+147,428   mean=−2.0
% regret entries < -300:   6.6%
% regret entries < -10000: 0.74%
infosets with ≥5 frozen actions: 1.6%
```

So **the threshold isn't a "definitively bad action" filter — it's a "got one mildly unlucky sample" filter.** The high-traffic infosets (preflop opens) get hit hardest because they accumulate large cumulative regrets fast.

### Why the cutoff is also wrong

Sum of weights:
```
S(c, T)  =  c(c+1)/2  +  (T−c)
```

For `c=1000`, `T=100K`:
- Pre-cutoff: `1000·1001/2 = 500,500`
- Post-cutoff: `99,000`
- **Pre-cutoff weight is 5× the post-cutoff weight**

So 1% of training time accounts for 84% of cumulative regret weight. Whatever the algorithm "learns" in iters 1–1000 — when buckets are first being encountered, strategies are near-uniform, EV estimates dominated by chance variance — gets etched in stone for the remaining 99K iterations.

| `c`     | `T`        | `f_linear` (early-phase weight share) |
|---------|------------|---------------------------------------|
| 1,000   | 100,000    | **83.5%**                             |
| 1,000   | 1,000,000  | 33.4%                                 |
| 450     | 100,000    | 50.4% (`c = √(2T)`)                   |
| 100,000 | 100,000    | 99.99% (pure Linear CFR)              |

Larger cutoff means LATE iterations get bigger weights too. Pure Linear CFR (`c = T`) gives the latest 1,000 iters ~200× the weight of the first 1,000 iters — exactly what you want.

### The original Linear CFR has no cutoff

Brown & Sandholm's original Linear CFR formulation has *no cutoff*. They prove that weighting by `t` is optimal for the cumulative regret bound. The cutoff was added later as a numerical safety net for extremely long training runs (10⁹+ iters). For `T ≤ 10⁶` and `Δmax = 40`, **the cutoff should never trigger**. Setting `LINEAR_CFR_CUTOFF = T_MAX` recovers the original Linear CFR theory.

### Recipe for any target `T_MAX`

1. **`LINEAR_CFR_CUTOFF = T_MAX`**. Restores the original Linear CFR weighting.
2. **`PRUNE_WARM_UP ≈ T_MAX / 10`**. Don't prune until at least one full order of magnitude of training has passed.
3. **`PRUNE_THRESHOLD = − Δmax × c² / 2 × κ`**, where `c = LINEAR_CFR_CUTOFF` and `κ ≈ 0.05`. Roughly 5% of the maximum possible cumulative regret magnitude.

### Concrete scaled values

For **`T_MAX = 100,000`** (current default — APPLIED in this commit):
```python
LINEAR_CFR_CUTOFF      = 100_000          # was 1_000
PRUNE_WARM_UP          = 10_000           # was 100
PRUNE_THRESHOLD        = -10_000_000      # was -300
PRUNE_SKIP_PROBABILITY = 0.95             # unchanged
```

For **`T_MAX = 1,000,000`**:
```python
LINEAR_CFR_CUTOFF      = 1_000_000
PRUNE_WARM_UP          = 100_000
PRUNE_THRESHOLD        = -1_000_000_000   # -1e9
```

For **`T_MAX = 10,000,000`**:
```python
LINEAR_CFR_CUTOFF      = 10_000_000
PRUNE_WARM_UP          = 1_000_000
PRUNE_THRESHOLD        = -1e11
```

For comparison: the Pluribus paper uses `−3 × 10⁸` for HUNL with chip stakes of ~10⁴ and total iterations of ~10¹². Normalized by `Δmax² × T`, their threshold is roughly `3 × 10⁻¹³` of max possible cumulative regret. The recommendation of `−10⁷` for `T=10⁵` and `Δmax=40` is `5 × 10⁻⁴` of max — *more conservative* than Pluribus, which is appropriate at small `T` because you can't afford incorrect prunes.

### Alternative: drop pruning entirely

Pruning is a *speed* optimization; it never improves convergence quality, only training throughput. With `T_MAX = 100K` and only ~60K infosets, the per-iteration traversal is fast enough that pruning's wall-clock benefit is small (~2–3×). Re-enable it once basics are working and you're scaling to `T = 10⁶+`.

```bash
python scripts/train.py --iterations 100000 --no-prune \
    --linear-cfr-cutoff 100000 \
    --out output/blueprint_v4.pkl --checkpoint-every 10000
```

### Verification checks

Three checks to run on a trained checkpoint to confirm a setting is correct:

1. **Pruning is rare and biased toward bad actions, not noise.** Count `(regret < threshold).sum() / total_entries`. Should be **<1%**, not 6.6%.
2. **Cumulative regret distribution is roughly symmetric.** `np.percentile(all_regrets, [1, 50, 99])` should look like `[−A, ~0, +A]`. If the 1st percentile is much more negative than the 99th is positive, the threshold is preferentially pruning below-zero actions before they recover.
3. **Strategy entropy decreases monotonically over checkpoints.** Plot mean entropy across iters 1K, 5K, 10K, 50K, 100K. With correct hyperparameters it should drop from ~0.95 to ~0.6. If it stays flat at 0.95 (current state), the algorithm is failing to commit — exactly what happens when noise is being amplified by misconfigured Linear CFR.

The third check would have caught the current state immediately and should be added as a regression test in `tests/`.

---

## Part 5 — Other observations

### Hyperparameter mismatches

`EVAL_HANDS_DEFAULT = 1` (`config.py:53`). In-loop eval is run on a *single hand* by default. Self-play eval at iteration logging is a coin flip, not a signal.

### Tests verify correctness, not performance

The lessons file shows tests were added to catch the prune-EV bug. They confirm the *traversal returns the right number*, but they don't test:

- Converged blueprint exploits known amateur patterns.
- Preflop hand-strength ordering is monotonic (AA opens more than 72o).
- Blueprint reaches every street (`river_infosets > 0`).
- Strategy entropy decreases over iterations.

Any of these would have caught "5000 iters is undertrained" immediately.

---

## Recommendations, in priority order

1. **Train ≥10⁵ iterations before evaluating.** With 5,000 iters the artifact is provably noise.
2. **Add `(num_active_opponents, pot_bucket, position)` to the postflop infoset key** in `info_set.py`. Single biggest correctness lever — the lossy abstraction will *never* converge to a good policy regardless of iteration count.
3. **Apply the new CFR hyperparameters** (done in this commit). Loosen pruning, fix Linear CFR weighting.
4. **Fix `EVAL_HANDS_DEFAULT`** to a real number (5000+).
5. **Add convergence regression tests**: assert river infosets > 0, AA opens more than 72o, mean strategy entropy < 0.85 after N iters.

The bug isn't in the math anymore — it's in the *quantity* of training and the *resolution* of the abstraction.
