# Pluribus Implementation — Completed

## Phase 1: Offline Blueprint Training Gaps

- [x] **1.1 Drop Preflop Information Abstraction** — Preflop infosets now use
  canonical 169-hand IDs directly instead of 15-bucket mapping. Postflop
  bucketing unchanged.
- [x] **1.2 Linear CFR Hard Cutoff** — `LINEAR_CFR_CUTOFF` config param
  (default 10,000). After cutoff, iteration weight = 1 (no discounting).
  Configurable via `--linear-cfr-cutoff` CLI flag. Checkpoint/resume preserves
  the cutoff value.

## Phase 2: Online Real-Time Search

- [x] **2.1 Reach Probability Tracking** — `ReachTracker` maintains 1326-entry
  probability vector over all hole-card pairs. Bijective pair_index encoding.
  Reset at hand start, updated after each bot action, renormalized.
- [x] **2.3 Continuation Strategies** — 4 strategies per player at depth-limit
  leaves: Blueprint, Fold-biased, Call-biased, Raise-biased. BIAS_FACTOR
  configurable.
- [x] **2.2 Subgame Construction** — Implicit in solver traversal. Current
  street uses exact card identity; future streets use blueprint buckets.
  Depth limit in betting rounds.
- [x] **2.4 Subgame CFR Solver** — MC Linear CFR on subgame, no pruning,
  local regret/strategy tables. Bucket caching for fast equity lookups.
- [x] **2.5 PluribusBot** — Preflop: blueprint average strategy. Postflop:
  subgame solver. Reach probs maintained and updated per action.
- [x] **2.6 Multi-Bot Play Harness** — `scripts/play.py` CLI with self-play,
  vs-amateur, and seat rotation modes. Block-bootstrap mbb/g reporting.

## New Files

- `poker_collusion/search/__init__.py`
- `poker_collusion/search/reach.py` — Reach probability tracker
- `poker_collusion/search/continuation.py` — Continuation strategies + leaf rollout
- `poker_collusion/search/solver.py` — Depth-limited subgame CFR solver
- `poker_collusion/search/bot.py` — PluribusBot agent
- `poker_collusion/search/play.py` — Multi-bot play harness
- `scripts/play.py` — CLI for match play

## Modified Files

- `poker_collusion/config.py` — Added 5 new config params
- `poker_collusion/abstraction/bucketing.py` — Made `hole_to_canonical` public;
  fixed equity runout card count bug
- `poker_collusion/abstraction/info_set.py` — Preflop uses canonical hand ID
- `poker_collusion/abstraction/__init__.py` — Exports `hole_to_canonical`
- `poker_collusion/cfr/trainer.py` — `_iteration_weight()` helper, cutoff logic,
  save/load cutoff
- `poker_collusion/cfr/strategy.py` — Added `get_average_strategy` alias to Strategy
- `scripts/train.py` — `--linear-cfr-cutoff` CLI flag

## Performance Notes

Postflop hands with search take ~100s/hand due to MC equity estimation in
bucketing (100 rollouts per unique (hole, board) pair). For production use,
replace the MC bucketing with precomputed lookup tables or a neural equity
estimator.
