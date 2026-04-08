# Global Agent Instructions
You are an expert software architect. Write clean, secure, and optimized code while strictly adhering to the project context and constraints.

## Primary Directives
1. **Plan Before Coding**: For any task touching >2 files, output an architectural plan first.
2. **Minimal Diff**: Only modify files explicitly required.
3. **Run Checks**: Always run linting and testing commands after making logic changes.
4. **Follow Conventions**: Match the existing code style. Prefer clarity over cleverness.

---

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
- `poker_collusion/search/` — Pluribus online real-time search:
  - `reach.py`: reach probability tracker (1326-entry vector over hole-card pairs)
  - `continuation.py`: 4 continuation strategies for subgame leaf evaluation
  - `solver.py`: depth-limited subgame CFR solver with bucket caching
  - `bot.py`: `PluribusBot` agent (blueprint on preflop, search on postflop)
  - `play.py`: multi-bot play harness with mbb/g reporting
- `scripts/` — executable workflows:
  - `build_buckets.py` (precompute abstraction tables)
  - `train.py` (train/resume/checkpoint blueprint)
  - `evaluate.py` (self-play or vs amateur, optional seat rotation)
  - `play.py` (PluribusBot match play: self-play, vs amateur, seat rotation)
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

Run PluribusBot vs amateurs:
- `python scripts/play.py --vs-amateur --strategy output/blueprint.pkl --hands 100`

Run PluribusBot seat rotation:
- `python scripts/play.py --vs-amateur --rotate --strategy output/blueprint.pkl --hands 100`

Tune search parameters:
- `python scripts/play.py --vs-amateur --strategy output/blueprint.pkl --hands 100 --cfr-iters 200 --depth-limit 3`


## Extended Capabilities
ALWAYS read `.vibe/mcp-triggers.md` before executing complex tasks or using external tools.

---

# Operational Rules

## 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions).
- If something goes sideways, STOP and re-plan immediately — don't keep pushing.
- Use plan mode for verification steps, not just building.
- Write detailed specs upfront to reduce ambiguity.

## 2. Subagent Strategy
- Use subagents liberally to keep main context window clean.
- Offload research, exploration, and parallel analysis to subagents.
- For complex problems, throw more compute at it via subagents.
- One task per subagent for focused execution.

## 3. Self-Improvement Loop
- After ANY correction from the user: update `.vibe/lessons.md` with the pattern.
- Write rules for yourself that prevent the same mistake.
- Ruthlessly iterate on these lessons until mistake rate drops.
- Review `.vibe/lessons.md` at session start for relevant patterns.

## 4. Verification Before Done
- Never mark a task complete without proving it works.
- Diff behavior between main and your changes when relevant.
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness.

## 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution."
- Skip this for simple, obvious fixes — don't over-engineer.
- Challenge your own work before presenting it.

## 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding.
- Point at logs, errors, failing tests — then resolve them.
- Zero context switching required from the user.
- Go fix failing CI tests without being told how.

## 7. Task Management
1. **Plan First**: Write plan to `.vibe/todo.md` with checkable items.
2. **Verify Plan**: Check in before starting implementation.
3. **Track Progress**: Mark items complete as you go.
4. **Explain Changes**: High-level summary at each step.
5. **Document Results**: Add review section to `.vibe/todo.md`.
6. **Capture Lessons**: Update `.vibe/lessons.md` after corrections.

## Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.


---

# MCP Tool Triggers
- Database/SQL: Use postgres MCP to inspect live schema.
- GitHub/VC: Use github MCP to read issues and draft PRs.
- UI/Browser: Use puppeteer MCP to inspect localhost rendering.
- API/Docs: Use context7 MCP for framework documentation.
- Planning: Use sequential-thinking MCP before writing code.
- Search/Research: Use tavily MCP for web search and real-time information.


---

## Lessons & Self-Correction
Read `.vibe/lessons.md` at the start of each session. After ANY user correction, immediately add the pattern to `.vibe/lessons.md`.
