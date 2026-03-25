# Copilot Instructions

## Project Overview

3-player No-Limit Texas Hold'em (NLHE) blueprint strategy via Monte Carlo Counterfactual Regret Minimization (MCCFR). The goal is to compute game-theoretic equilibrium strategies and study collusion between two players against a third. Specs: 20 BB stacks, 3 players (P0=Button, P1=SB, P2=BB), 10 abstract actions per round.

## Build & Run Commands

```bash
# Install dependencies (Python 3.8+)
pip install numpy scikit-learn tqdm

# 1. Precompute bucket abstraction tables (run once, outputs to data/)
python scripts/build_buckets.py --postflop-samples 5000 --postflop-rollouts 200

# 2. Train blueprint
python scripts/train.py --iterations 10000 --out output/blueprint.pkl

# Resume training from checkpoint
python scripts/train.py --load output/blueprint.pkl --iterations 5000 --out output/blueprint.pkl

# Save checkpoints every N iterations
python scripts/train.py --iterations 10000 --out output/blueprint.pkl --checkpoint-every 2000

# 3. Evaluate a blueprint
python scripts/evaluate.py --strategy output/blueprint.pkl --hands 50000

# Validate on small game (Kuhn poker, should converge in ~50k iters)
python main.py kuhn
```

There is no test suite. Use `python main.py kuhn` to sanity-check the MCCFR implementation.

## Architecture

The canonical implementation lives in `poker_collusion/` and `scripts/`. Root-level files (`cfr.py`, `nlhe3p.py`, `kuhn3p.py`, `evaluate.py`, `main.py`) are legacy/reference only.

**Data flow:**
1. `scripts/build_buckets.py` → precomputes `data/{preflop,flop,turn,river}_buckets.pkl`
2. `scripts/train.py` → runs MCCFR using `poker_collusion/cfr/trainer.py`, saves blueprint pickle
3. `scripts/evaluate.py` → loads blueprint, plays self-play hands, reports mbb/g with block bootstrap CI

**Package layout:**
- `poker_collusion/env/` — game engine (`game_state.py`, `game_logic.py`, `hand_eval.py`)
- `poker_collusion/abstraction/` — action abstraction (10 actions), hand-strength bucketing, info set key construction
- `poker_collusion/bucketing_build/` — MC equity + k-means precomputation for bucket tables
- `poker_collusion/cfr/` — MCCFR trainer with Linear CFR weighting and regret pruning
- `poker_collusion/evaluation/` — mbb/g self-play evaluation with block bootstrap standard error
- `poker_collusion/config.py` — all hyperparameters (stack size, bucket counts, CFR params)

## Key Conventions

### Mutable step-back game engine
`apply_action(state, action_idx)` **mutates** state in-place and pushes a snapshot to `state.undo_stack`. `undo_action(state)` pops and restores the snapshot. The CFR traversal uses this to branch the game tree without copying state. Never call `apply_action` without a matching `undo_action` when you need to explore multiple branches.

### Info set key format
```
"{round_idx}|{bucket}|{action1,action2,...,DEAL,...}"
# e.g. "1|23|5,1,DEAL,3,1,2"
```
`round_idx` is 0–3 (preflop/flop/turn/river). `DEAL` sentinels in the action history mark street boundaries. Keys are constructed by `poker_collusion/abstraction/info_set.py`.

### Action indexing
Actions 0–9 are consistent across all streets:
- 0 = Fold, 1 = Check/Call, 2–8 = Raise/Bet sizes, 9 = All-in
- Preflop raises: absolute BB amounts `[2.0, 2.5, 3.0, 4.0, 5.0, 8.0, 12.0]`
- Postflop bets: pot multipliers `[0.25, 0.33, 0.5, 0.66, 0.75, 1.0, 1.5]`
- Duplicate total-bet amounts are deduplicated in `get_legal_action_indices()`

### Hand-strength buckets
- Preflop: 169 canonical hands → 15 buckets (k-means on MC equity)
- Postflop: 50 buckets per street (flop/turn/river), loaded from `data/` at runtime
- If bucket tables don't exist, `bucketing.py` falls back to a simple equity approximation

### CFR trainer storage
`trainer.regret_sum`, `trainer.strategy_sum`, and `trainer.action_map` are plain dicts keyed by info set string. Each value is a `numpy` array sized to the number of legal actions at that info set. Persisted via `pickle`.

### Evaluation metric
Performance is reported as **mbb/g** (millibig blinds per game = avg BB profit × 1000). Variance is estimated via **block bootstrap** (blocks of 500 hands) to account for within-session correlation—not naive SE.

### Milestone 2 (collusion)
To implement collusion: modify `CFRTrainer` so that traversers 0 and 1 optimize a **shared objective** (sum of P0 + P1 payoffs) rather than individual payoffs. The game engine and abstraction layers are unchanged.
