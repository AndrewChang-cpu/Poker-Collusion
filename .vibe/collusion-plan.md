# Collusion plan — single-hand 3-player NLHE

## Infoset: skip / out-of-hand tokens (planned change)

**Problem:** `get_info_key` uses `(round_idx, bucket, street_history)` where `street_history` is only **action indices** (0–9) on the current street (`poker_collusion/abstraction/info_set.py`). Folds use **`0`**. The engine still advances `current_player` correctly when seats are inactive (`_advance_to_next_player` skips `active=False` in `poker_collusion/env/game_logic.py`), but **that structure is not in the key**. So the same abstract tuple can arise with **different** active-player sets — different **next actor** after folds — and merge infosets incorrectly.

**Change:** Extend the street-history representation with a **distinct sentinel** (not action `0`) meaning "no decision here / seat skipped because already out" (or an equivalently precise rule), emitted whenever the rotation **skips** a seat. Example pattern (conceptual): `(1, SKIP, 1, 5, SKIP)` — `SKIP` is **not** "player folded now"; fold-as-choice stays `0`.

**Requirements:**

- Define **exactly** when `SKIP` is appended so it stays **consistent** with `_advance_to_next_player` / postflop first actor in `sample_chance`.
- Use a type that cannot collide with legal actions (e.g. string token, dedicated int, or parallel `actor_history`-style encoding — pick one and document).
- Expect **new infoset space**; old `.pkl` blueprints are **not** compatible without migration.

**Scope:** Fixes **active-player / rotation aliasing** in the key only — not bucket coarseness, not dropping prior-street context from the key.

## Engine facts (implementation)

- One hand per episode; terminal payoffs: `get_payoffs` → net BB per seat (`poker_collusion/env/game_state.py`).
- **Seat roles are fixed to indices:** `deal_new_hand` always sets **P0 = BTN**, **P1 = SB**, **P2 = BB**; preflop order `0→1→2`, postflop `1→2→0` (`poker_collusion/config.py`). There is **no button rotation** in the current env.
- CFR today returns `payoffs[traverser]` only (`poker_collusion/cfr/trainer.py`); team training requires a **scalar \(J\)** (or CFV tied to \(J\)) for traversers in `{i,j}`.

## Default seating (v1)

- **Colluders:** seats **0 and 1** (BTN + SB). **Frozen solo bot:** seat **2** (BB).
- **Rationale:** matches `deal_new_hand` / `play.py` labels; solo player is always BB in this layout; postflop acting order is SB(1) → BB(2) → BTN(0), so the bot sits **between** the two colluders postflop (sandwich in postflop order).
- **Sensitivity (optional, separate runs):** fixed remappings without engine rotation — e.g. team `{0,2}` vs solo `1` (BTN+BB vs SB), team `{1,2}` vs solo `0` (blinds vs BTN). Each is a different **policy-to-seat** assignment, not a within-codebase "rotate button" feature.

## Team objective

- **Primary \(J\) (utilitarian):** \(J = u_i + u_j\) for team seats \(i,j\).
- **Training rule of thumb:** at terminal, compute `payoffs` as now; derive \(J\); for nodes where `player == traverser` and `traverser` is a colluder, CFR regret/values use the **team scalar** (team MCCFR), not `payoffs[traverser]` alone. Non-team traverser unchanged. Execution at runtime: still two tables keyed by each player's `get_info_key(state, player)` (no card sharing).
- **Allied non-traverser nodes:** when `traverser = 0` and the tree reaches a node where seat 1 (the other colluder) acts, treat it as a **standard opponent node** — sample one action from seat 1's current training strategy, no special logic. Coordination emerges because both colluders optimize the same \(J\) during their respective traversals, converging jointly via iterated best response on a shared objective.

### Alternative \(J\) (separate checkpoints, same frozen bot + default seating)

| Name | \(J\) | Notes |
|------|-------|--------|
| Max-min | \(\min(u_i, u_j)\) | |
| Smooth compromise | e.g. \(u_i + u_j + \lambda \min(u_i, u_j)\) | fix \(\lambda\), report |
| Risk-shaped | \(\phi(u_i) + \phi(u_j)\), \(\phi\) concave | e.g. \(\log(c+u)\); choose \(c\) so domain covers min terminal \(u\) |

**Reporting:** headline metric for all runs = **utilitarian** team mbb/g \((u_i + u_j)\) and solo mbb/g so objectives are comparable.

## Frozen opponent (v1 scope)

- Load a **single** selfish 3p blueprint (same abstraction as training). **Freeze** strategy for seat **2** (no regret updates for that seat during team training; always use frozen `get_average_strategy` or stored probs at seat 2 nodes).
- Document: path to `.pkl`, training iterations, and that seat 2 was never updated after freeze.

## Infoset / seat in the key

- Until the **skip-token** change lands, `get_info_key` is as above; **fold / active-set aliasing** remains possible (see first section).
- `bucket` is from the acting `player`'s cards; colluders vs frozen bot all use the same keying scheme.
- With **fixed** P0=BTN / P1=SB / P2=BB, full-ring paths often pin the actor from prefix length + order; **player/role in the key** is **not** needed for rotation experiments — seat sensitivity is achieved via separate training runs with different `team_seats` assignments.

## Evaluation

- **Primary:** utilitarian team mbb/g for seats \(i,j\); solo mbb/g for seat 2.
- **Control:** independent 3p CFR baseline under the same seat layout (no team \(J\)).
- **Optional:** per-seat BB/hand; simple action-frequency deltas vs baseline (detectability).

---

## Later (not in scope for v1 collusion)

- **Retrained solo:** unfreeze seat 2 and train against the team (adaptation); separate numbers from frozen bot.
- Multi-hand sessions, stack carryover, session \(J\), discounting, bust/ICM-style terms.
- Engine feature: **rotate button / permute seats** each hand so one policy sees all roles (if you want full positional fairness without separate layouts).

**When you get there (multi-hand):** session termination; infoset with stack vector; regret on session return vs per-hand shaping; compute cost.

---

## Resolved decisions

1. **v1 seating:** Ship **team `{0,1}` vs frozen `2`** first. Sensitivity layouts as separate labeled training runs.
2. **Frozen weights:** Use existing `blueprint_X.pkl` from repo. User passes path via `--frozen-strategy` arg.
3. **Team training scope:** Update regrets only for team seats. Always sample frozen policy at seat 2. Skip `traverser=2` entirely in `train()` loop.
4. **Objective rollout:** Utilitarian \(J\) first. Alternative objectives as follow-up runs once plumbing works.
5. **Infoset:** Implement skip-tokens **before** collusion training. Retrain baseline with fix. `player`/`role` is **not** needed in the key for rotation experiments.
6. **Togglability:** All collusion features are opt-in. Without team flags, existing selfish CFR works unchanged.

---

# Implementation plan

## Phase A: Skip-token infoset fix

**Goal:** Disambiguate infoset keys when players have folded on prior streets.

**Approach:** The engine already tracks `actor_history` (who acted at each step) alongside `action_history`. Use actor-tagged pairs `(actor, action)` in the street history component of the key. This is equivalent to the SKIP sentinel concept — when a player has folded, no `(player, action)` pair appears for them, so two games with different fold patterns from earlier streets produce different key tuples.

### File changes

**`poker_collusion/abstraction/info_set.py`** — `get_info_key`:
- Current key: `(round_idx, bucket, tuple(street_actions))`
- New key: `(round_idx, bucket, tuple(zip(street_actors, street_actions)))`
- Slice `state.actor_history` the same way as `action_history` (everything after the last `"DEAL"` index).
- The `"DEAL"` sentinel appears in both `action_history` and `actor_history` at the same positions; use the same `last_deal` index for both.

**No engine changes.** `game_logic.py` already appends to both `action_history` and `actor_history` in `apply_action`. `sample_chance` appends `"DEAL"` to `action_history`; verify it also appends a corresponding sentinel to `actor_history` (if not, add one — e.g. `"DEAL"` or `-1`).

**Compatibility:** Old `.pkl` blueprints are **not** compatible. Retrain baseline after this change.

### Validation
- Unit test: construct two game states that currently produce the same key but have different fold histories from prior streets. Assert they produce different keys after the fix.
- Smoke: `python scripts/train.py -n 100 --out /dev/null` runs without error.

---

## Phase B: Team MCCFR training

**Goal:** Train two colluding agents against a frozen opponent, with all features togglable.

### B1: Config additions (`poker_collusion/config.py`)

```python
# Collusion defaults (all None = selfish mode)
TEAM_SEATS: Optional[List[int]] = None      # e.g. [0, 1]
FROZEN_SEATS: Optional[List[int]] = None     # e.g. [2]
```

### B2: CFRTrainer changes (`poker_collusion/cfr/trainer.py`)

**Constructor** — add optional params (all default to `None` / selfish mode):
```python
def __init__(self, ...,
    team_seats: Optional[List[int]] = None,
    frozen_trainer: Optional['CFRTrainer'] = None,
):
    self.team_seats = set(team_seats) if team_seats else set()
    self.frozen_trainer = frozen_trainer
```

**Terminal return** — `cfr_traverse` line 110-113 and `_cfr_traverse_local` line 230-231:
```python
if self.game.is_terminal(state):
    payoffs = self.game.get_payoffs(state)
    if self.team_seats and traverser in self.team_seats:
        return sum(payoffs[s] for s in self.team_seats)
    return payoffs[traverser]
```

**Opponent node strategy** — `cfr_traverse` line 179-186 and `_cfr_traverse_local` line 285-288. When the current player is the frozen seat, sample from the frozen blueprint instead of training regrets:
```python
else:  # player != traverser
    if self.frozen_trainer and player not in self.team_seats:
        strategy = self.frozen_trainer.get_average_strategy(info_key, actions)
        if strategy is None:
            strategy = np.ones(len(actions)) / len(actions)
    # else: strategy already computed from self.get_strategy above
    action_idx = np.random.choice(len(actions), p=strategy)
    ...
```

**`train()` loop** — line 419-425. Skip frozen seat as traverser:
```python
for traverser in range(self.num_players):
    if self.frozen_trainer and traverser not in self.team_seats:
        continue  # frozen seat — no updates
    ...
```

**`train_parallel()` loop** — line 377-379. Same traverser filter in job generation:
```python
traversers = [i for i in range(self.num_players)
              if not self.frozen_trainer or i in self.team_seats]
jobs = [
    (self.game.deal_new_hand(), traversers[i % len(traversers)], weight, rngs[i])
    for i in range(batch_size)
]
```

**`_cfr_traverse_local`** — mirror the terminal return and opponent strategy changes from `cfr_traverse`. The frozen trainer lookup uses `self.frozen_trainer.get_average_strategy(info_key, actions)` which reads `strategy_sum` — safe for read-only access across threads.

### B3: CLI changes (`scripts/train.py`)

New arguments:
```
--team-seats 0,1          # comma-separated seat indices for team
--frozen-strategy PATH    # path to frozen opponent .pkl
```

When `--team-seats` is provided:
- `--frozen-strategy` is **required** (error if missing).
- Load frozen blueprint as a second `CFRTrainer` via `frozen_trainer.load(path)`.
- Pass `team_seats` and `frozen_trainer` to the training `CFRTrainer`.

When `--team-seats` is **not** provided:
- Existing selfish behavior. `--frozen-strategy` is ignored if present.

### B4: Save/load metadata

Extend `CFRTrainer.save()` to persist `team_seats` in the pickle dict when non-empty. `load()` restores it. This lets evaluation know the checkpoint is a team strategy.

---

## Phase C: Team evaluation

### C1: `scripts/evaluate.py` additions

New mode: `--team-eval` with required `--team-strategy PATH` and `--frozen-strategy PATH`:
- Load team checkpoint for seats in `team_seats` (from saved metadata).
- Load frozen baseline for remaining seat(s).
- Build `policies` list: team trainer for team seats, frozen trainer for frozen seat.
- Call `evaluate_strategies(game, policies, ...)`.
- Print: team mbb/g = `mbb[seat_i] + mbb[seat_j]`, solo mbb/g = `mbb[frozen_seat]`.

Control run (for comparison): `evaluate_strategies` with the original selfish blueprint in all 3 seats (existing functionality — already works).

### C2: Team mbb/g reporting (`poker_collusion/evaluation/mbbg.py`)

Add a utility function:
```python
def summarize_team(mbb_mean, mbb_se, team_seats, seat_labels=("BTN","SB","BB")):
    """Print team aggregate and solo opponent mbb/g from per-seat arrays."""
```

Called by `evaluate.py` after `evaluate_strategies` returns per-seat results.

---

## Phase D: Validation

### D1: Correctness checks
- **Frozen immutability:** After team training, load the frozen `.pkl` and the frozen trainer's `regret_sum` / `strategy_sum`. Assert byte-identical to the original (no writes leaked).
- **Team terminal value:** Instrument `cfr_traverse` with a debug flag. For a known terminal state, assert the returned value = `payoffs[0] + payoffs[1]` (not `payoffs[traverser]` alone).
- **Traverser skip:** Assert that after N iterations of team training, `regret_sum` contains zero keys where the info key could only belong to seat 2 (frozen seat never traverses, never accumulates regrets).

### D2: Smoke tests
- `python scripts/train.py -n 100 --out /tmp/team.pkl --team-seats 0,1 --frozen-strategy output/blueprint.pkl` completes without error.
- `python scripts/evaluate.py --team-eval --team-strategy /tmp/team.pkl --frozen-strategy output/blueprint.pkl --hands 1000` produces non-zero team mbb/g.

### D3: Expected results
- Team mbb/g should be **positive** (colluders extract value from frozen opponent).
- Solo mbb/g should be **negative** (mirror of team extraction, modulo noise).
- Team mbb/g should exceed independent baseline mbb/g (the whole point of the study).

---

## Implementation order

1. **Phase A** (skip-token) — standalone, no collusion dependency. Retrain baseline after.
2. **Phase B** (team MCCFR) — core algorithmic change. Depends on retrained baseline from A.
3. **Phase C** (evaluation) — needs team checkpoint from B to test.
4. **Phase D** (validation) — runs after B and C are functional.

Phases A and B2/B3/B4 are parallelizable if the skip-token baseline is available. Phase C can be built concurrently with late Phase B since it only needs the CLI plumbing and `evaluate_strategies` already exists.
