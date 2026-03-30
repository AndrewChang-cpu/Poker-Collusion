# Pluribus Implementation Plan (3-Player NLHE)

## Scope

This plan covers the remaining work to implement the full Pluribus algorithm
for 3-player no-limit Texas hold'em with 20 BB stacks. It is divided into two
phases:

- **Phase 1**: Offline blueprint training gaps
- **Phase 2**: Online real-time search (the core Pluribus innovation)

Existing infrastructure (MCCFR with external sampling, Linear CFR, regret
pruning, bucketing, evaluation harness) is treated as complete and correct.

---

## Phase 1: Offline Blueprint Training Gaps

### 1.1 — Drop Preflop Information Abstraction

**What**: Remove bucket lookup on the preflop street and instead use the
canonical 169-hand representation directly as the infoset key component.

**Why**: The paper states: *"Pluribus only plays according to this blueprint
strategy in the first betting round (of four), where the number of decision
points is small enough that the blueprint strategy can afford to not use
information abstraction."* With 3 players and 20 BB stacks, the preflop tree
is small enough to enumerate all 169 canonical hands without bucketing.

169 canonical hands = 13 pairs + 78 suited combos + 78 offsuit combos,
encoded as a single integer `[0, 168]` via a precomputed rank+suited lookup.

**Definition of done**:
- Preflop infosets use canonical hand IDs, not bucket IDs
- The number of distinct preflop infoset keys equals `169 × (number of
  distinct preflop action histories)` — verifiable by inspecting
  `strategy_sum` after a training run
- Blueprint retrains successfully from scratch with no errors
- Postflop bucketing is unchanged

---

### 1.2 — Linear CFR Hard Cutoff

**What**: Add a configurable iteration cutoff after which Linear CFR
discounting stops and training continues as plain CFR (weight = 1 per
iteration). Default cutoff: 10,000 iterations. Overridable via CLI.

**Why**: The paper states: *"We stop the discounting after that because the
time cost of doing the multiplications with the discount factor is not worth
the benefit later on."*

Behavior: for `t ≤ cutoff`, iteration weight = `t`; for `t > cutoff`,
weight = 1. Applies to both regret and strategy sum updates.

**Definition of done**:
- For iterations beyond the cutoff, the per-iteration weight is exactly 1
- The cutoff value is configurable in `config.py` and overridable from the
  CLI — hardcoding it anywhere else is not acceptable
- Training runs that cross the cutoff boundary produce no errors and
  checkpoint/resume correctly

---

## Phase 2: Online Real-Time Search

This is the core Pluribus innovation. After the preflop round ends, Pluribus
abandons the blueprint and instead solves a depth-limited subgame in real time
at each decision point.

The blueprint is still used: (a) directly on preflop, (b) as one of the four
continuation strategies at subgame leaf nodes.

---

### 2.1 — Reach Probability Tracking

**What**: Each bot instance maintains a running probability vector over all
1326 possible hole card pairs, updated after each action the bot takes during
a hand. This state is owned by the bot and reset at the start of each hand.

**Why**: Subgame construction requires knowing the probability that the bot
would have reached the current game state while holding each possible hand.
This is necessary to balance the strategy across all hands simultaneously —
without it the bot could be exploited. The paper states: *"Pluribus keeps
track of the probability it would have reached the current situation with
each possible hand according to its strategy... it will first calculate how
it would act with every possible hand, being careful to balance its strategy
across all the hands."*

At hand start: uniform over all hands consistent with the known deck (hands
containing cards already visible — board or actual hole cards — are zeroed
out). After each action the bot takes: each hand's probability is multiplied
by the probability that hand would have taken that action at its infoset,
then renormalized.

**Definition of done**:
- Reach prob vector has exactly 1326 entries (one per canonical hole pair)
- Impossible hands (using cards visible to the bot) are always zero
- After a sequence of actions, the sum of all reach probs equals 1.0
- Reach probs are reset to uniform at the start of each new hand
- Reach probs update correctly when the bot holds different hands in
  isolation tests (simulate bot holding hand A and hand B through the same
  action sequence; the resulting distributions should differ appropriately)

---

### 2.2 — Subgame Construction

**What**: A subgame is a modified game tree rooted at the current game state,
used by the real-time CFR solver. Its root is a chance node that branches
once per feasible hole card pair, weighted by the bot's normalized reach
probabilities. The game tree under each branch uses the same 10-action
abstraction as the blueprint. Information abstraction (bucketing) is applied
only to future streets — the current street being searched uses actual card
identities. Leaf nodes are reached at the depth limit or at game-terminal
states.

Depth limit is configurable in units of betting rounds remaining. Setting it
≥ 4 effectively disables it (at most 3 streets remain after preflop). Default
is 3 (search to end of game), which is feasible given 20 BB shallow stacks.

**Definition of done**:
- The subgame root correctly enumerates all feasible hands with weights
  proportional to reach probs; infeasible hands have weight zero
- Current-street nodes use exact card identities; future-street nodes use
  blueprint buckets
- Depth limit is respected: no node in the subgame tree exceeds the
  configured depth
- A subgame built from the same game state and reach probs produces
  identical structure on repeated calls (deterministic construction)

---

### 2.3 — Continuation Strategies

**What**: At each leaf node of the subgame, instead of a fixed terminal
value, each player simultaneously and independently chooses among k=4
continuation strategies. The leaf value is estimated by rolling out the
remainder of the game under the chosen strategies and averaging over multiple
rollouts.

The 4 continuation strategies per player:
1. **Blueprint** — the trained average strategy
2. **Fold-biased** — blueprint with fold probability multiplied by
   `BIAS_FACTOR`, renormalized
3. **Call-biased** — blueprint with call probability multiplied by
   `BIAS_FACTOR`, renormalized
4. **Raise-biased** — blueprint with all raise/bet probabilities multiplied
   by `BIAS_FACTOR`, renormalized

`BIAS_FACTOR` and the number of MC rollouts per leaf are configurable
in `config.py`.

**Why this works**: An unbalanced subgame strategy (e.g., always betting big)
is punished by an opponent choosing the fold-biased continuation. The bot is
forced to find a strategy robust to all four opponent continuations.

**Definition of done**:
- All 4 continuation strategies produce valid probability distributions over
  legal actions (non-negative, sum to 1) for any infoset
- Leaf value estimates converge as rollout count increases (verify by running
  with 10, 100, and 1000 rollouts on the same leaf state and checking that
  variance decreases)
- A game-terminal node returns the exact payoff vector regardless of
  continuation strategy (continuation strategies are only applied at
  non-terminal depth-limit leaves)

---

### 2.4 — Depth-Limited Subgame CFR Solver

**What**: A CFR solver that runs on the subgame for a fixed number of
iterations and returns a strategy for each feasible hand the bot could hold.

The algorithm is Monte Carlo Linear CFR — identical to the blueprint trainer
— with these differences:
- Runs for a fixed iteration count (`SUBGAME_CFR_ITERATIONS` in config),
  not until convergence
- No regret pruning
- All feasible hands are solved jointly in each traversal, weighted by reach
  probabilities
- Leaf values come from the continuation strategy mechanism (2.3), not
  terminal payoffs
- Regret and strategy state is local to each solve call and discarded
  afterward

Output: for each feasible hand, a probability distribution over legal actions.
The bot executes the action for the hand it actually holds.

**Definition of done**:
- The solver completes in finite time for any valid game state and reach prob
  vector
- Output distributions are valid (non-negative, sum to 1) for every feasible
  hand
- Running the solver on the same inputs twice produces identical output
  (deterministic given a fixed RNG seed)
- A pure blueprint bot and a search bot (with the blueprint as the only
  continuation strategy and depth limit = full game) produce strategies that
  are close in distribution on the same infosets, confirming the solver
  is consistent with the blueprint

---

### 2.5 — PluribusBot: Online Play Agent

**What**: A self-contained bot object that uses the blueprint on preflop and
the subgame solver on all postflop streets. It encapsulates reach probability
state for the current hand and exposes three methods: reset at hand start,
select an action, and observe an opponent action.

On preflop: the bot samples directly from the blueprint average strategy for
its actual hand. On postflop: the bot runs the subgame solver using its
current reach probs, then selects the action for its actual hand from the
resulting strategy.

**Definition of done**:
- The bot produces a valid action index for every game state where it is the
  active player
- Preflop actions come exclusively from the blueprint; postflop actions come
  exclusively from the subgame solver
- Reach probs are correctly initialized at hand start and correctly updated
  after each action the bot takes
- Two bot instances with different seats but the same blueprint and config
  behave symmetrically on equivalent game states

---

### 2.6 — Multi-Bot Play Harness

**What**: A play loop that runs N hands between multiple bot instances
(any mix of `PluribusBot` and existing `AmateurPolicy` opponents), collects
per-seat payoffs, and reports mbb/g with block-bootstrap variance.

At hand start: deal cards and initialize each bot for the new hand. At each
decision point: query the active bot for an action. After each action:
notify all bots. At hand end: record payoffs. Reuse existing variance
estimation utilities.

A CLI script (`scripts/play.py`) exposes this as a runnable command with
configurable blueprint path, hand count, opponent type, and search parameters
(iterations, depth limit, bias factor).

**Definition of done**:
- A match between three `PluribusBot` instances runs to completion without
  errors for any number of hands
- A match between one `PluribusBot` and two `AmateurPolicy` opponents runs
  correctly
- Per-seat mbb/g and standard errors are reported and are finite/non-NaN
- The search-augmented bot achieves a higher mbb/g than the pure blueprint
  bot against `AmateurPolicy` opponents (directional correctness check, not
  a hard threshold — variance is expected)

---

## Implementation Order

1. **1.1** — Drop preflop bucketing
2. **1.2** — Linear CFR hard cutoff
3. **Retrain blueprint** after Phase 1 changes
4. **2.1** — Reach probability tracking
5. **2.3** — Continuation strategies
6. **2.2** — Subgame construction
7. **2.4** — Subgame CFR solver
8. **2.5** — PluribusBot agent
9. **2.6** — Multi-bot play harness + CLI

Steps 4–9 are sequentially dependent. Each step should be verified against
its definition of done before proceeding.

---

## New Config Parameters

| Parameter | Default | Section |
|---|---|---|
| `LINEAR_CFR_CUTOFF` | `10_000` | 1.2 |
| `SUBGAME_DEPTH_LIMIT` | `3` | 2.2 |
| `BIAS_FACTOR` | `4.0` | 2.3 |
| `SUBGAME_LEAF_ROLLOUTS` | `10` | 2.3 |
| `SUBGAME_CFR_ITERATIONS` | `200` | 2.4 |

---

## Open Questions / Design Decisions to Revisit

- **SUBGAME_CFR_ITERATIONS default**: Start at 200; tune empirically by
  comparing mbb/g of search bot vs pure blueprint bot against amateurs.
- **SUBGAME_LEAF_ROLLOUTS**: At 20 BB stacks with depth limit = 3 (full
  game), leaf nodes are always terminal and this parameter is irrelevant.
  It only matters if the depth limit is reduced below 3.
- **Warm-starting subgame CFR from blueprint**: Not in the paper; defer
  unless convergence is too slow at the default iteration count.
- **Off-tree bet handling**: Not needed for bot-vs-bot play (shared action
  abstraction). Pseudoharmonic mapping would be required if playing against
  humans or external bots with arbitrary bet sizes.
