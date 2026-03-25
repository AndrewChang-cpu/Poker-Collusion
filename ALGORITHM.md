# Algorithm Documentation: `poker_collusion` Blueprint Strategy

## 1. What This System Does

This codebase trains a **blueprint strategy** for 3-player No-Limit Texas Hold'em using **Monte Carlo Counterfactual Regret Minimization (MCCFR)** with **external sampling**. The algorithm is Pluribus-inspired: it iteratively self-plays millions of hands, accumulating "regret" about past decisions, and converges toward a Nash equilibrium approximation. The final output is a lookup table mapping every situation a player can be in (an "information set") to a probability distribution over actions.

The pipeline has three phases:
1. **Bucket Build** — precompute card abstraction tables
2. **Train** — run MCCFR to produce a blueprint strategy
3. **Evaluate** — measure strategy quality in mbb/g (milli-big-blinds per game)

---

## 2. Game Setup

**Parameters** (`config.py`):
- 3 players: P0 = Button, P1 = Small Blind (0.5 BB), P2 = Big Blind (1 BB)
- Starting stack: 20 BB each
- Initial pot: 1.5 BB (SB + BB)

**Positions and acting order**:
- **Preflop**: P0 (Button) → P1 (SB) → P2 (BB). BB can check if no raise.
- **Postflop** (flop/turn/river): P1 → P2 → P0 (skip folded/all-in players)

**Streets**: Preflop (round 0), Flop (round 1, 3 community cards), Turn (round 2, 1 card), River (round 3, 1 card).

---

## 3. Action Abstraction

The continuous action space of NLHE is collapsed to **10 discrete actions per street** (`abstraction/actions.py`):

### Preflop actions (indices 0–9)

| Index | Meaning |
|-------|---------|
| 0 | Fold |
| 1 | Check / Call |
| 2 | Raise to 2.0 BB |
| 3 | Raise to 2.5 BB |
| 4 | Raise to 3.0 BB |
| 5 | Raise to 4.0 BB |
| 6 | Raise to 5.0 BB |
| 7 | Raise to 8.0 BB |
| 8 | Raise to 12.0 BB |
| 9 | All-in (20 BB) |

These are **absolute BB amounts** for total bet size.

### Postflop actions (indices 0–9)

| Index | Meaning |
|-------|---------|
| 0 | Fold |
| 1 | Check / Call |
| 2 | Bet 0.25× pot |
| 3 | Bet 0.33× pot |
| 4 | Bet 0.50× pot |
| 5 | Bet 0.66× pot |
| 6 | Bet 0.75× pot |
| 7 | Bet 1.0× pot |
| 8 | Bet 1.5× pot |
| 9 | All-in |

These are **pot-relative** — "pot" is calculated as the current pot plus the amount needed to call (`_pot_for_acting`).

### Legality filtering (`get_legal_action_indices`)

Not all 10 actions are legal at every decision point. The filter enforces:

1. **Fold (0)**: legal only when `to_call > 0` (can't fold if not facing a bet)
2. **Check/Call (1)**: Check when `to_call == 0`; Call when `to_call > 0` and `stack >= to_call`
3. **Raise sizes (2–8)**: Legal only if:
   - The resulting total bet ≥ the **minimum raise** (`max_bet + last_raise_amount`)
   - The resulting total bet ≤ the player's stack
   - The amount isn't a duplicate of another already-legal size
4. **All-in (9)**: Legal if stack > 0 and not already represented by another size

If a player can't afford to call (stack < to_call), only All-in (9) is offered alongside Fold (0).

### Chip translation (`action_index_to_chips`)

Converts an abstract action index into `(is_fold, total_bet_this_street)`:
- **Fold**: `(True, current_bet)` — no additional chips
- **Check/Call**: `(False, current_bet + min(to_call, stack))`
- **Preflop raise (2–8)**: `(False, PREFLOP_RAISE_BB[index - 2])` — the value *is* the total bet
- **Postflop bet (2–8)**: `(False, to_call + pot * multiplier)`, capped at stack
- **All-in (9)**: `(False, current_bet + stack)` — everything goes in

---

## 4. Information Abstraction (Card Bucketing)

The information set must abstract the ~2.6 million distinct hold'em hands into a small number of "buckets" so the strategy table is tractable.

### 4.1 Preflop: 169 → 15 buckets

**Build step** (`bucketing_build/preflop_table.py`):
1. Enumerate all 169 canonical hold'em starting hands (13 pairs + 78 suited + 78 offsuit)
2. For each, estimate all-in equity vs a single random opponent hand via Monte Carlo (default 1000 rollouts per hand)
3. Sort all 169 hands by equity
4. Assign to 15 equal-frequency buckets: hand at position `i` gets bucket `i * 15 // 169`

**Runtime lookup** (`abstraction/bucketing.py`):
- Map the 2 hole cards to a canonical ID (0–168) via `_hole_to_canonical`:
  - Pairs → ID 0–12 (by rank)
  - Non-pairs → ID 13–168, indexed by `(high, low, suited/offsuit)`
- Look up the ID in the precomputed table to get bucket 0–14
- **Fallback** (no table): simple `high*13 + low + pair_bonus + suited_bonus` score, linearly mapped to 15 buckets

### 4.2 Postflop: 50 buckets per street (flop/turn/river)

**Build step** (`bucketing_build/postflop_table.py`):
1. Sample many random (hand, board) combinations (default 50,000)
2. For each, estimate equity via Monte Carlo:
   - **Flop**: equity vs 1 random opponent, with random turn+river (default 500 rollouts)
   - **Turn**: equity vs 1 random opponent, with random river
   - **River**: equity vs 2 random opponents (since 3-player), exact board
3. Run **k-means** (k=50) on the 1D equity values
4. Store the 50 cluster centers (just float values, not a full lookup table)

**Runtime lookup**:
- Compute equity of the current (hand, board) via `_estimate_equity` (100 MC rollouts by default)
- Find the nearest cluster center → that center's index is the bucket
- **Fallback** (no cluster centers loaded): use the hand evaluator's category (0=high card through 8=straight flush), linearly mapped to the bucket count

---

## 5. Information Set Key

The information set key is what a player "knows" — their card bucket and the full public action history (`abstraction/info_set.py`):

```python
key = (bucket, tuple(action_history))
```

Where `action_history` is a list of:
- **Integer action indices** (0–9) for player actions
- **The string `"DEAL"`** when community cards are dealt

This is a tuple (hashable), used as a dictionary key in the CFR tables.

**What's NOT in the key**: the player's identity. Position is recoverable from the history because the acting order is deterministic given the street and who has folded/gone all-in.

### Example key

A player holding a medium-strength hand (bucket 7) in a hand where: P0 called preflop (action 1), P1 raised to 3 BB (action 4), P2 called (action 1), P0 called (action 1), then the flop was dealt, and P1 checked (action 1):

```python
(7, (1, 4, 1, 1, "DEAL", 1))
```

On the flop, the bucket would be recomputed for the new board, so this player's actual key at this point would use their *flop* bucket (not the preflop bucket 7). The key is computed fresh at each decision node from the current `state.round_idx` and `state.board`.

---

## 6. The MCCFR Algorithm

### 6.1 Core Data Structures (`cfr/trainer.py`)

```
regret_sum:   dict{ info_key → np.array of length 10 }
strategy_sum: dict{ info_key → np.array of length 10 }
action_map:   dict{ info_key → list of legal action indices }
```

All arrays are length `NUM_ACTIONS = 10` (one slot per abstract action). Only the indices that are legal at a given info set get nonzero entries.

### 6.2 Regret Matching (`cfr/strategy.py`)

Converts accumulated regrets into a strategy (probability distribution):

```
positive[a] = max(regret_sum[info][a], 0)  for each legal action a
if sum(positive) > 0:
    strategy[a] = positive[a] / sum(positive)
else:
    strategy = uniform over legal actions
```

Only the regret entries for the *current legal actions* are extracted and normalized. This means if action 3 is not legal in a particular state, its regret is ignored.

### 6.3 External Sampling Traversal

Each training iteration, for each of the 3 players (the "traverser"), a new hand is dealt and the game tree is traversed:

```
cfr_traverse(state, traverser):
    if terminal:      return payoffs[traverser]
    if chance_node:   sample_chance(state); recurse; return

    player = current_player
    actions = legal_actions(state)
    info_key = get_info_key(state, player)
    strategy = regret_matching(regret_sum[info_key], actions)

    if player == traverser:
        // EXPLORE ALL ACTIONS
        for each action a in actions:
            if should_prune(info_key, a): skip (value = 0)
            apply_action(state, a)
            values[a] = cfr_traverse(state, traverser)
            undo_action()     // step back to explore next action

        ev = strategy · values
        regret_update = values - ev    // counterfactual regret per action
        regret_sum[info_key] += regret_update * weight
        strategy_sum[info_key] += strategy * weight
        return ev

    else:  // opponent
        // SAMPLE ONE ACTION
        a = random_choice(actions, p=strategy)
        apply_action(state, a)
        val = cfr_traverse(state, traverser)
        undo_action()
        return val
```

**Key distinction**: For the traverser, we explore *all* legal actions (full branching). For opponents, we sample *one* action. This is the "external sampling" variant of MCCFR — it gives unbiased regret estimates while keeping the tree exploration tractable.

### 6.4 Linear CFR

When `use_linear_cfr = True` (the default), both regret and strategy updates are weighted by the iteration number `t`:

```
weight = t  (1-indexed iteration)
regret_sum[info][a]   += regret_update[a] * t
strategy_sum[info][a] += strategy[a] * t
```

This causes later iterations (with better strategies) to contribute more to the final average. The theoretical effect is faster convergence — early noisy strategies are downweighted.

### 6.5 Regret-Based Pruning

After a warm-up period (default: 100 iterations), actions with strongly negative cumulative regret are skipped probabilistically:

```
if iteration > 100 and regret_sum[info][action] < -300:
    skip this action with 95% probability
```

When skipped, `values[a] = 0.0` — neither regret nor strategy is updated for that action. This dramatically reduces computation by avoiding deep exploration of actions the algorithm has already learned are bad.

### 6.6 Average Regret Metric

The training loop reports average regret as a convergence diagnostic:

```
For Linear CFR:
    sum_weights = T * (T + 1) / 2     (sum of 1 + 2 + ... + T)
For standard CFR:
    sum_weights = T

avg_regret = (1/|info_sets|) * Σ mean(max(regret[info], 0)) / sum_weights
```

This measures how much "positive regret" remains per info set — lower means closer to equilibrium.

---

## 7. Game Engine Mechanics

### 7.1 State Representation (`env/game_state.py`)

The `NLHEState` object holds:
- `deck`: shuffled 52-card array, `deck_idx` tracks dealing position
- `hole_cards[3]`: two cards per player (dealt from deck[0..5])
- `board[]`: community cards (0, 3, 4, or 5 cards)
- `round_idx`: 0=preflop, 1=flop, 2=turn, 3=river
- `stacks[3]`, `pot`, `bets[3]`: chip tracking (bets are per-street)
- `active[3]`, `all_in[3]`: player status
- `action_history[]`: sequence of action indices (int) and `"DEAL"` strings
- `last_raiser`, `last_raise_amount`: for min-raise enforcement
- `chance_pending`: True when a street has ended and cards need to be dealt
- `undo_stack[]`: snapshots for step-back

### 7.2 Chance Nodes

When a betting round completes (but it's not the river), the engine does **not** deal cards immediately. Instead, it sets `chance_pending = True`. The CFR traversal then sees `is_chance_node(state) == True` and calls `sample_chance(state)`, which:

1. Deals cards from the deck (3 for flop, 1 for turn/river)
2. Appends `"DEAL"` to `action_history`
3. Advances `round_idx`
4. Resets per-street bets to 0
5. Sets `current_player` to the first active non-all-in player in postflop order (P1 → P2 → P0)
6. Pushes undo info onto `undo_stack`

### 7.3 Step-Back / Undo (`game_logic.py:undo_action`)

The mutable engine avoids expensive deep copies by using an undo stack. Each `apply_action` pushes a snapshot:

```python
{"stacks": [...], "pot": ..., "bets": [...], "active": [...],
 "all_in": [...], "last_raiser": ..., "last_raise_amount": ...,
 "current_player": ...}
```

Each `sample_chance` pushes a tuple: `("DEAL", n_cards, bets, last_raiser, last_raise_amount)`.

`undo_action()` pops the top of the stack and restores state. This is O(1) and avoids allocating new state objects — critical for performance since MCCFR calls apply/undo millions of times.

Note: `undo_action` uses a **module-level `_current_state`** variable (set by `apply_action`) so the CFR trainer can call `game.undo_action()` without passing the state explicitly.

### 7.4 Round Completion (`_is_round_complete`)

A betting round ends when:
1. All active non-all-in players have acted at least once this street
2. All active non-all-in players have equal bets
3. If there was a raise, every other active player has acted *after* the raise

The function reconstructs who acted by walking the `action_history` from the last `DEAL` marker, mapping sequential actions to players using the known acting order.

### 7.5 Hand Resolution and Side Pots

**Single winner by fold**: If only 1 player remains active, they win the entire pot.

**Showdown** (`_resolve_side_pots`): When multiple players are active at showdown:
1. Compute each player's total contribution: `20 - remaining_stack`
2. Identify distinct contribution levels (sorted ascending)
3. For each level: the pot slice is `(level - previous_level) × count_of_eligible_contributors`
4. Among the eligible players who are still active (didn't fold), the best hand wins that slice

This correctly handles cases like: P0 all-in for 5 BB, P1 all-in for 20 BB, P2 folds. P0 can only win up to 5 BB from each player (the main pot); P1 gets back the excess (the side pot).

---

## 8. Stepped Example: One MCCFR Iteration

Let's trace a single training iteration where **P0 is the traverser**.

### Setup

```
Iteration t = 500
Deal: P0 gets [A♠ K♥], P1 gets [7♣ 4♦], P2 gets [T♠ 9♠]
Stacks after blinds: P0=20, P1=19.5 (posted 0.5 SB), P2=19 (posted 1 BB)
Pot = 1.5,  bets = [0, 0.5, 1.0]
state.action_history = []
```

### Step 1: P0's preflop decision (traverser — explore ALL)

```
P0's hole cards: [A♠ K♥] → canonical hand: AKo → preflop bucket: 13 (high bucket)
info_key = (13, ())
legal_actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    (fold, call, raise 2BB, 2.5BB, 3BB, 4BB, 5BB, 8BB, 12BB, all-in)
```

The traverser explores every action. Say the current strategy from regret matching is:
```
strategy = [0.01, 0.05, 0.10, 0.15, 0.25, 0.20, 0.10, 0.08, 0.04, 0.02]
```

For **each** of the 10 actions, apply → recurse → undo. Let's follow the branch where **P0 raises to 3 BB (action 4)**:

```
apply_action(state, 4):
    total_bet = 3.0 BB (from PREFLOP_RAISE_BB[4-2] = 3.0)
    add = 3.0 - 0 = 3.0
    P0's stack: 20 → 17
    P0's bets[0]: 0 → 3.0
    pot: 1.5 → 4.5
    last_raiser = 0, last_raise_amount = 3.0
    action_history = [4]
    push snapshot to undo_stack
```

### Step 2: P1's decision (opponent — sample ONE)

```
P1's hole cards: [7♣ 4♦] → preflop bucket: 2 (low bucket)
info_key = (2, (4,))
legal_actions = [0, 1, 5, 6, 7, 8, 9]
    (fold, call 3BB, raise 4BB, 5BB, 8BB, 12BB, all-in)
    Raise sizes 2 (2BB), 3 (2.5BB), 4 (3BB) are < min_raise_total, so filtered out.
```

Suppose regret matching gives:
```
strategy = [0.65, 0.20, 0.05, 0.04, 0.03, 0.02, 0.01]
    (for:    fold,  call,  4BB,  5BB,  8BB,  12BB, all-in)
```

**Sample one action**: dice roll → P1 folds (action 0). This is the opponent, so we only sample once.

```
apply_action(state, 0):
    P1 folds: active[1] = False
    action_history = [4, 0]
```

### Step 3: P2's decision (opponent — sample ONE)

```
P2's hole cards: [T♠ 9♠] → preflop bucket: 8
info_key = (8, (4, 0))
legal_actions = [0, 1, 5, 6, 7, 8, 9]
    Same min-raise logic; small raise sizes filtered
```

Suppose strategy gives P2 calling with 0.40 probability. Dice roll → P2 calls (action 1).

```
apply_action(state, 1):
    to_call = 3.0 - 1.0 = 2.0
    P2's stack: 19 → 17
    P2's bets[2]: 1.0 → 3.0
    pot: 4.5 → 6.5
    action_history = [4, 0, 1]
```

### Step 4: Round complete → Chance node

P0 raised, P1 folded, P2 called. `_is_round_complete` checks:
- Can act: P0 and P2 (P1 folded). But P0 was the last raiser and P2 acted after — round complete.
- `round_idx = 0 < 3`, so `chance_pending = True`.

CFR sees `is_chance_node(state) == True`, calls `sample_chance`:

```
sample_chance(state):
    Deal 3 cards: board = [J♦, 5♠, 2♣]
    action_history = [4, 0, 1, "DEAL"]
    round_idx: 0 → 1
    bets reset to [0, 0, 0]
    current_player = 2 (first active postflop: P1 folded, so P2 acts first)
```

### Step 5: Postflop play (P2 then P0)

**P2 acts** (opponent — sample one):
```
P2's hand [T♠ 9♠] + board [J♦, 5♠, 2♣]:
    equity estimation → bucket 22 (mid-range — has a gutshot and backdoor flush)
info_key = (22, (4, 0, 1, "DEAL"))
legal_actions = [1, 2, 3, 4, 5, 6, 7, 8, 9]   (can't fold, not facing a bet; check=1)
```

Dice → P2 checks (action 1).

**P0 acts** (traverser — explore ALL):
```
P0's hand [A♠ K♥] + board [J♦, 5♠, 2♣]:
    equity estimation → bucket 31 (decent — two overcards)
info_key = (31, (4, 0, 1, "DEAL", 1))
legal_actions = [1, 2, 3, 4, 5, 6, 7, 8, 9]   (check or bet)
```

P0 explores all 9 legal actions. For each, the recursion continues deeper (P2 responds, possibly more streets). Eventually each branch returns a payoff value.

Say the returned values are:
```
values = [-, 2.1, 2.5, 2.8, 3.1, 3.0, 2.9, 2.4, 1.8, 0.5]
    (indices:  1   2    3    4    5    6    7    8    9)
    index 0 (fold) is not legal, so not explored
```

Strategy from regret matching:
```
strategy = [0.15, 0.05, 0.10, 0.15, 0.20, 0.15, 0.10, 0.07, 0.03]
```

### Step 6: Regret and strategy update

```
ev = strategy · values = 0.15*2.1 + 0.05*2.5 + ... ≈ 2.65

regret_update[a] = values[a] - ev   for each legal action
    e.g. action 5 (0.66× pot): 3.0 - 2.65 = +0.35 (betting more was better)
    e.g. action 1 (check):     2.1 - 2.65 = -0.55 (checking was worse)

weight = 500  (Linear CFR: weight = iteration number)

regret_sum[(31, (4,0,1,"DEAL",1))][a] += regret_update[a] * 500
strategy_sum[(31, (4,0,1,"DEAL",1))][a] += strategy[a] * 500
```

After this, `undo_action` is called repeatedly to restore state back to P0's first decision, and the next of P0's 10 preflop actions is explored. Eventually all 10 branches complete, the preflop regrets are updated, and the iteration for traverser=P0 is done.

The process repeats with traverser=P1 and traverser=P2 (fresh deals each time). Then iteration 501 begins.

---

## 9. Final Strategy Extraction

After all iterations complete, the **blueprint strategy** at any info set is:

```
blueprint[info] = strategy_sum[info] / sum(strategy_sum[info])
```

If the sum is zero (unseen info set), fall back to uniform. This weighted average of all strategies played during training converges to a Nash equilibrium approximation. The Linear CFR weighting ensures later (better) iterations dominate.

---

## 10. Evaluation

### Self-play (`evaluate_with_variance`)

All 3 players use the blueprint. Each hand:
1. Deal, then at each decision node sample an action from `blueprint[info_key]`
2. Record payoffs (BB) for each player

**mbb/g** = mean payoff (in BB) × 1000. In a perfectly converged strategy, all 3 players should have mbb/g ≈ 0 (Nash equilibrium in a symmetric game).

### CFR vs Amateur (`evaluate_vs_amateur`)

One player uses the blueprint; the other two use `AmateurPolicy` — a heuristic that:
- Preflop: scores hands by high rank + pair bonus + suited bonus + connected bonus
- Postflop: runs 100 MC rollouts for hand strength
- Maps strength + pot odds to fold/call/raise weights

The CFR player's mbb/g should be positive (winning) against amateurs, demonstrating the strategy is exploiting weaker play.

### Block Bootstrap

Hands are grouped into blocks of 500. Per-block mean payoffs are computed, then standard error is `std(block_means) / sqrt(n_blocks)`. 95% CI = mean ± 1.96 × SE.
