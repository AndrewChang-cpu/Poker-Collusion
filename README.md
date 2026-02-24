# Poker Collusion: Pluribus-Style Blueprint via MCCFR

3-player No-Limit Texas Hold'em blueprint strategy via Monte Carlo Counterfactual Regret Minimization (MCCFR), matching the spec in `PROJECT_FORMULATION.md`.

## Project Structure

```
poker_collusion/
├── __init__.py
├── config.py              # Game params (20 BB), bucket counts, CFR hyperparams
├── env/                   # Game environment
│   ├── game_state.py      # State, deal_new_hand, payoffs
│   ├── game_logic.py      # Legal actions, apply_action, undo_action, chance nodes
│   └── hand_eval.py       # 5-card evaluator, side pot resolution
├── abstraction/
│   ├── actions.py         # 10 actions per round, legality filtering
│   ├── bucketing.py       # Bucket lookup (precomputed tables)
│   └── info_set.py        # Info set key = (bucket, action_history)
├── bucketing_build/       # Precompute bucket tables
│   ├── preflop_table.py   # 169 canonical -> 15 buckets
│   └── postflop_table.py  # Sample + MC equity + k-means -> 50 per street
├── cfr/
│   ├── trainer.py         # MCCFR external sampling, Linear CFR, pruning
│   └── strategy.py       # Regret matching, average strategy
└── evaluation/
    └── mbbg.py            # Self-play, mbb/g, block bootstrap SE

scripts/
├── build_buckets.py       # Build preflop + postflop bucket tables
├── train.py               # Run MCCFR, save blueprint
└── evaluate.py            # Load blueprint, report mbb/g
```

## Usage

From the project root:

```bash
# 1. Build bucket tables (run once; optional for quick tests — fallback bucketing is used)
python scripts/build_buckets.py --postflop-samples 5000 --postflop-rollouts 200

# 2. Train blueprint
python scripts/train.py --iterations 10000 --out output/blueprint.pkl

# Optional: save a checkpoint every N iterations (e.g. every 2000)
python scripts/train.py --iterations 10000 --out output/blueprint.pkl --checkpoint-every 2000
# Writes output/blueprint_2000.pkl, output/blueprint_4000.pkl, ... and final to output/blueprint.pkl

# Optional: resume from a saved strategy (runs --iterations more, then overwrites --out)
python scripts/train.py --load output/blueprint.pkl --iterations 5000 --out output/blueprint.pkl

# 3. Evaluate
python scripts/evaluate.py --strategy output/blueprint.pkl --hands 50000
```

## Game Parameters (from formulation)

- 3 players: P0 = Button, P1 = SB, P2 = BB
- 20 BB per player; SB 0.5 BB, BB 1 BB
- Preflop order: 0 → 1 → 2; postflop order: 1 → 2 → 0
- 10 abstract actions per round (preflop: fold, check/call, 7 raise sizes, all-in; postflop: fold, check/call, 7 pot-relative bets, all-in)
- Full side-pot resolution at showdown

## Dependencies

- Python 3.8+
- NumPy
- scikit-learn (optional; for k-means when building postflop bucket tables)

## Notes

- The original `cfr.py`, `nlhe3p.py`, `kuhn3p.py`, and `evaluate.py` at the repo root remain for reference; the canonical implementation is in `poker_collusion/` and `scripts/`.
- For Milestone 2 (cooperation), use a modified trainer that optimizes a shared objective for two of the three players.
