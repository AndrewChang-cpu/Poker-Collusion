# Poker Collusion: Pluribus-Style Blueprint via MCCFR (+ Online Search)

3-player No-Limit Texas Hold'em blueprint strategy via Monte Carlo Counterfactual Regret Minimization (MCCFR), matching the spec in `PROJECT_FORMULATION.md`.

This repo also includes a Pluribus-style **online real-time search** agent (`PluribusBot`) that plays **blueprint on preflop** and uses a **depth-limited subgame CFR solver** postflop.

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
├── search/                # Pluribus-style online search (postflop)
    ├── reach.py           # ReachTracker (1326 hole-card pair probs)
    ├── continuation.py    # Continuation strategies at depth-limit leaves
    ├── solver.py          # Depth-limited subgame CFR solver (bucket caching)
    ├── bot.py             # PluribusBot (blueprint preflop, search postflop)
    └── play.py            # Multi-bot play harness + mbb/g reporting
└── evaluation/
    └── mbbg.py            # Self-play, mbb/g, block bootstrap SE

scripts/
├── build_buckets.py       # Build preflop + postflop bucket tables
├── train.py               # Run MCCFR, save blueprint
├── evaluate.py            # Load blueprint, report mbb/g
└── play.py                # PluribusBot match play (self-play / vs amateur / rotate)
```

## Usage

From the project root:

```bash
# 1. Build bucket tables (run once; optional for quick tests — fallback bucketing is used)
python scripts/build_buckets.py --postflop-samples 5000 --postflop-rollouts 200

# 2. Train blueprint
python scripts/train.py --iterations 10000 --out output/blueprint.pkl

# Optional: stop Linear CFR discounting after a cutoff (default: 10,000)
python scripts/train.py --iterations 10000 --out output/blueprint.pkl --linear-cfr-cutoff 10000

# Optional: save a checkpoint every N iterations (e.g. every 2000)
python scripts/train.py --iterations 10000 --out output/blueprint.pkl --checkpoint-every 2000
# Writes output/blueprint_2000.pkl, output/blueprint_4000.pkl, ... and final to output/blueprint.pkl

# Optional: resume from a saved strategy (runs --iterations more, then overwrites --out)
python scripts/train.py --load output/blueprint.pkl --iterations 5000 --out output/blueprint.pkl

# 3. Evaluate
python scripts/evaluate.py --strategy output/blueprint.pkl --hands 50000

# 4. Online play (PluribusBot)
# 3 PluribusBot self-play
python scripts/play.py --strategy output/blueprint.pkl --hands 100

# 1 PluribusBot vs 2 amateurs
python scripts/play.py --vs-amateur --strategy output/blueprint.pkl --hands 100

# Rotate PluribusBot through all seats vs amateurs
python scripts/play.py --vs-amateur --rotate --strategy output/blueprint.pkl --hands 100

# Tune search parameters
python scripts/play.py --vs-amateur --strategy output/blueprint.pkl --hands 100 \
    --cfr-iters 200 --depth-limit 3 --leaf-rollouts 10 --bias-factor 4.0

# Plot evaluation curve over multiple .pkl files
python scripts/eval_curve.py output/blueprint_claude_v3_1800.pkl output/blueprint_claude_v3_1850.pkl output/blueprint_claude_v3_1900.pkl output/blueprint_claude_v3_1950.pkl output/blueprint_claude_v3_2000.pkl --hands 1000 --workers 8 --ci se

# 5. Team MCCFR (collusion training)
# Train seats 0,1 as a team against a frozen opponent blueprint at seat 2
python3t scripts/train.py --log-interval 1 --out output/blueprint_v4.pkl \
    --checkpoint-every 100 --workers 8 --batch-size 24 \
    --team-seats 0,1 --frozen-strategy output/blueprint_claude_v3_32100.pkl \
    --team-objective utilitarian
# --team-objective choices: utilitarian (default), maxmin, smooth, risk

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
