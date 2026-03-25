# Project Context

- Name: Poker Collusion
- Stack:
  - Language: Python (package-style project)
  - Core libraries: `numpy`, `tqdm`
  - Optional/used for bucketing build: `scikit-learn` (k-means), `pickle` (strategy/bucket persistence)
  - Runtime style: CLI scripts (`scripts/*.py`) driving package modules (`poker_collusion/*`)
- Architecture:
  - This is a research/training codebase for 3-player NLHE using MCCFR with abstractions.
  - `scripts/` are entrypoints (`build_buckets`, `train`, `evaluate`).
  - `poker_collusion/env` implements mutable game state + step-back transitions (`apply_action`/`undo_action`).
  - `poker_collusion/abstraction` maps raw game states to abstract actions and bucketed infosets.
  - `poker_collusion/cfr` implements MCCFR traversal, regret/strategy storage, pruning, and persistence.
  - `poker_collusion/evaluation` runs self-play and CFR-vs-amateur evaluation using mbb/g with block-based variance estimates.
  - `data/` stores bucket tables (`*.pkl`), `output/` stores trained blueprint snapshots (`*.pkl`), `logs/` stores debug/error logs.

## Project Architecture & Directory Map

- `poker_collusion/` — primary Python package for engine, abstraction, training, evaluation, and config.
- `poker_collusion/config.py` — central constants (game params, abstraction sizes, CFR hyperparams, eval defaults).
- `poker_collusion/env/` — game engine/state transitions:
  - `game_state.py`: `NLHEState`, initial dealing, payoff extraction
  - `game_logic.py`: legal-action routing, apply/undo, street progression, terminal resolution
  - `hand_eval.py`: hand ranking and showdown utilities
- `poker_collusion/abstraction/` — abstraction layer:
  - `actions.py`: fixed 10-action abstraction and legality filtering
  - `bucketing.py`: bucket lookup/loading with runtime fallback
  - `info_set.py`: infoset key generation `(bucket, action_history)`
- `poker_collusion/bucketing_build/` — offline table builders for preflop/flop/turn/river abstractions.
- `poker_collusion/cfr/` — MCCFR implementation:
  - `trainer.py`: traversal, regret updates, pruning, checkpoints/save/load
  - `strategy.py`: regret matching and average strategy helpers
  - `debug.py`: CFR debug instrumentation
- `poker_collusion/evaluation/` — evaluation runners and baseline opponent policy.
- `scripts/` — executable workflows:
  - `build_buckets.py` (precompute abstraction tables)
  - `train.py` (train/resume/checkpoint blueprint)
  - `evaluate.py` (self-play or vs amateur, optional seat rotation)
- `data/` — generated bucket table artifacts.
- `output/` — generated trained strategy artifacts.
- `logs/` — runtime diagnostics and traceback logs.
- `.github/copilot-instructions.md` — project guidance (not CI workflow config).
- `.vibe/` — agent memory/config docs (including this file).

## Anti-Patterns & "Never Do This"

- Never bypass `poker_collusion/config.py` by hardcoding game constants (stack sizes, bucket counts, action count, prune params) inside engine/trainer logic.
- Never mutate game state in tree traversal without a matching undo path; every decision/chance mutation in CFR traversal must be reversible or traversal correctness breaks.
- Never create alternate action encodings outside `abstraction/actions.py`; action indices `0..9` are a shared contract across env, trainer, and evaluation.
- Never build infoset keys ad hoc; always use `abstraction/info_set.py` so strategy/regret dictionaries remain consistent.
- Never commit generated heavy artifacts unless explicitly needed (`output/*.pkl`, large `data/*.pkl`, debug logs) — they are runtime outputs, not source of truth.
- Never rely on module-global hidden state patterns for new features (e.g., global current state coupling) when explicit state passing is feasible; it reduces reentrancy and introduces subtle bugs.
- Never treat TODO-marked correctness workarounds (depth caps, debug hacks, temporary fallbacks) as final behavior; verify whether they are masking upstream bugs before extending them.

## Git & Workflow Standards

Observed in repo:
- No `CONTRIBUTING.md` present.
- Default branch is `main`; remote branch seen: `parallelization`.
- Commit style is informal and mixed (e.g., `Added ...`, `add ...`, `merge`), not conventional-commit formatted.
- Repository currently includes both source changes and doc/config updates in-flight; generated assets have historically been committed in some commits.

Recommended default standard (since no formal guide exists):
- Branch naming: `feature/<topic>`, `fix/<topic>`, `chore/<topic>`, `docs/<topic>`.
- Commit message format: imperative, concise, scoped when useful (e.g., `fix(cfr): undo chance node after traversal`).
- Keep commits focused: separate algorithm/code changes from generated model artifacts.
- For PRs: include problem statement, behavior change, commands run, and before/after evidence for training/eval-impacting changes.

## Definition of Done (DoD)

Before marking a task complete, ensure all applicable checks pass:

- Code is consistent with package architecture (`scripts/` as entrypoints, core logic in `poker_collusion/*`).
- No new hardcoded environment-specific paths; file paths resolve from repo-root/module paths.
- CLI workflows still run for touched areas:
  - bucket build (if abstraction code changed),
  - training smoke run (if CFR/env changed),
  - evaluation smoke run (if policy/eval changed).
- No regressions in mutable step-back flow (apply/undo/chance transitions) for CFR traversal paths touched.
- Lint/type/test: project has no formal suite; at minimum run targeted command-based smoke checks and verify no runtime exceptions in modified flow.
- Docs/config updated when behavior or commands changed (`README.md`, `.vibe/project-context.md`, or related docs).
- No accidental inclusion of large generated artifacts/logs unless task explicitly requires them.

## Useful Project Commands

Install deps:
- `pip install -r requirements.txt`

Build abstraction tables:
- `python scripts/build_buckets.py --postflop-samples 5000 --postflop-rollouts 200`

Train blueprint:
- `python scripts/train.py --iterations 10000 --out output/blueprint.pkl`

Resume training from checkpoint:
- `python scripts/train.py --load output/blueprint.pkl --iterations 5000 --out output/blueprint.pkl`

Train with periodic checkpoints:
- `python scripts/train.py --iterations 10000 --out output/blueprint.pkl --checkpoint-every 2000`

Evaluate self-play:
- `python scripts/evaluate.py --strategy output/blueprint.pkl --hands 50000`

Evaluate CFR vs amateur (single seat):
- `python scripts/evaluate.py --vs-amateur --strategy output/blueprint.pkl --hands 10000 --cfr-seat 0`

Evaluate CFR vs amateur (seat rotation):
- `python scripts/evaluate.py --vs-amateur --rotate --strategy output/blueprint.pkl --hands 10000`
