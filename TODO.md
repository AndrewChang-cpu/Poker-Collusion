# Backlog

Prioritized by impact on training correctness, then performance, then maintainability.

---

## P0 — Correctness Bugs (fix before trusting any trained model)

### BUG-1: Undo stack corruption from chance nodes
**File:** `poker_collusion/cfr/trainer.py:97-99`

`sample_chance` pushes a DEAL entry onto `state.undo_stack`, but `cfr_traverse` never calls `undo_action()` after the chance recursion returns. The caller's subsequent `undo_action()` pops the DEAL instead of the caller's own action snapshot, orphaning that snapshot and corrupting the undo stack for the rest of the hand.

This is the root cause of the non-termination bug currently masked by the `depth_limit=500` workaround.

```python
# current (broken)
if self.game.is_chance_node(state):
    new_state = self.game.sample_chance(state)
    return self.cfr_traverse(new_state, traverser, len(new_state.action_history))

# fix: undo the DEAL after recursion returns
if self.game.is_chance_node(state):
    self.game.sample_chance(state)
    val = self.cfr_traverse(state, traverser, len(state.action_history))
    self.game.undo_action()
    return val
```

---

### BUG-2: `_who_acted_this_round` uses a fixed player rotation
**File:** `poker_collusion/env/game_logic.py:171-179`

`_who_acted_this_round` reconstructs which player made each action by cycling through a fixed order `[0, 1, 2]` (preflop) or `[1, 2, 0]` (postflop). When a player folds or goes all-in mid-street, they are skipped by `_advance_to_next_player` but not by this reconstruction. The function then attributes subsequent actions to the wrong players, causing `_is_round_complete` to return incorrect results — either ending rounds too early or never terminating them.

Fix: track `(player, action_index)` pairs in `action_history` instead of bare action indices, or maintain a separate per-street acted set on the state that is updated by `apply_action` and restored by `undo_action`.

---

### BUG-3: `last_raise_amount` is the wrong quantity
**File:** `poker_collusion/env/game_logic.py:92`

```python
state.last_raise_amount = add   # add = total_bet - state.bets[p]
```

`add` is the chips the player moves from their stack, not the raise increment above the previous max bet. Example: BB = 1.0, P0 (with `bets[0] = 0`) raises to 3.0 BB. `add = 3.0 - 0 = 3.0`, so `min_raise_total = 3.0 + 3.0 = 6.0`. Correct poker rules give raise increment = `3.0 - 1.0 = 2.0` and `min_raise_total = 3.0 + 2.0 = 5.0`. This makes min-raise thresholds consistently too high, eliminating legal raise sizes from the action space and distorting the learned strategy.

```python
# fix
prior_max = max(state.bets[q] for q in range(NUM_PLAYERS) if q != p)
state.last_raise_amount = total_bet - prior_max
```

---

### BUG-4: `depth_limit` workaround returns 0.0 (biases model)
**File:** `poker_collusion/cfr/trainer.py:86-89`

The depth limit is a band-aid for BUG-1 and BUG-2. Returning `0.0` when the limit is hit is not neutral — it tells the algorithm that all paths leading here have zero value, which skews regret updates for every info set on the path. Remove this once BUG-1 and BUG-2 are fixed.

---

## P1 — Performance (remove before any serious training run)

### PERF-1: Undo stack printed at every CFR node
**File:** `poker_collusion/cfr/trainer.py:77-80`

```python
try:
    print(depth, [f'{snapshot['stacks']} {snapshot['pot']}' for snapshot in state.undo_stack])
except:
    print('EXCEPTION:', state.undo_stack)
```

This executes at every single node visit — millions of times per training run — and dominates wall time. Delete entirely.

---

### PERF-2: NDJSON logged to disk at every terminal node
**File:** `poker_collusion/cfr/trainer.py:92-93`

```python
entry = {"trainingIteration": ..., "message": "terminal_reached", ...}
write_to_debug(entry)
```

Every terminal state writes a JSON line to disk. For a 100k-iteration run across a deep game tree this is tens of millions of disk writes. Delete or gate behind an explicit debug flag.

---

### PERF-3: `assert` in hot path
**File:** `poker_collusion/cfr/trainer.py:76`

`assert depth == len(state.action_history)` runs on every node. Either remove it after BUG-1/BUG-2 are fixed and the invariant is confirmed, or replace with a conditional check that is only active in a debug mode.

---

## P2 — Code Quality

### QUALITY-1: Hardcoded absolute paths
**File:** `poker_collusion/cfr/trainer.py:21-22`

```python
_CFR_ERROR_LOG = "/Users/aechang/Documents/Coding/Poker-Collusion/logs/cfr_error_traceback.log"
_DEBUG_LOG = "/Users/aechang/Documents/Coding/Poker-Collusion/logs/debug.log"
```

Replace with paths relative to project root (e.g., using `pathlib` from `__file__`).

---

### QUALITY-2: Module-level `_current_state` global makes engine non-reentrant
**File:** `poker_collusion/env/game_logic.py:6`

`_current_state` is set by `apply_action` so `undo_action()` can be called without passing state. This makes it impossible to run two games concurrently and creates an invisible coupling between the engine and the CFR trainer. Pass `state` explicitly to `undo_action(state)` and update the trainer to call `self.game.undo_action(state)`.

---

### QUALITY-3: Duplicate all-in append (dead branch)
**File:** `poker_collusion/abstraction/actions.py:94-97`

```python
if all_in_total not in seen_totals:
    legal.append(9)
else:
    legal.append(9)   # always appends regardless of branch
```

The `if/else` is meaningless — both branches do the same thing. Collapse to a single `legal.append(9)`.

---

### QUALITY-4: Large volume of commented-out code
**File:** `poker_collusion/cfr/trainer.py`

- Line 48: `use_step_back` detection
- Lines 122-123, 144-145: `if self.use_step_back:` guards
- Lines 163-167: recursion limit block
- Lines 184-185: `max_depth_seen` log line
- Lines 241-249: normalization in `load()`

Remove or restore these. Commented-out code at this density obscures the actual logic and makes future edits error-prone.

---

### QUALITY-5: `# TODO: BUGGY` on DEAL undo entry
**File:** `poker_collusion/env/game_logic.py:45`

```python
state.undo_stack.append(("DEAL", n, list(state.bets), state.last_raiser, state.last_raise_amount)) # TODO: BUGGY
```

The defensive `if len(top) >= 5` check in `undo_action` suggests the tuple format has changed at some point. Resolve as part of fixing BUG-1.

---

### QUALITY-6: ALGORITHM.md example has wrong legal actions
**File:** `ALGORITHM.md`, Section 8 Steps 2 and 3

The example shows `legal_actions = [0, 1, 5, 6, 7, 8, 9]` after P0 raises to 3 BB. Due to BUG-3, `min_raise_total = 6.0`, so indices 5 (4 BB) and 6 (5 BB) are actually filtered out. Correct legal actions are `[0, 1, 7, 8, 9]`. Update the example after BUG-3 is fixed (since the fix will change what is actually correct).
