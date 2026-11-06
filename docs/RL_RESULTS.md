# MAPPO team trial: results and hacks (2026-11-06)

Setup: `docs/RL_ALGORITHM.md`. BTN+SB team (MAPPO, shared actor, centralized
critic) vs BB frozen to the 5M-iteration MCCFR baseline
(`models/leduc_baseline_5000k.pkl`, compacted by `poker_collusion/rl/frozen.py`).
M5 MacBook Air CPU, 4096 hands/update, lr 3e-4, ent 0.01, clip 0.2.
Logs and checkpoints (every 100 updates): `output_rl/<run>/`.
Eval: 20k hands, fixed seed, team = BTN+SB BB/hand.

Reference: all three seats at the MCCFR baseline give the team **+0.155 ±0.008**.

| Run | Objective / opponent fallback | Updates (hands) | Team BB/hand by checkpoint |
| --- | --- | --- | --- |
| `mappo_team01_s0` | utilitarian / uniform | 406 (1.7M), stopped to free CPU | 0.98 @100-200, 1.38 @300, 3.35 @400 |
| `mappo_team01_s1` | utilitarian / uniform | 1000 (4.1M) | 0.99 @200-600, 1.35 @700, 3.80 @800, 5.38 @1000 |
| `ppo_indiv01_s0` | individual (control) / uniform | 1000 (4.1M) | 0.78 @100-600, 1.02 @1000 |
| `mappo_smooth01_s0` | sum + min / uniform | 1000 (4.1M) | 0.74-0.77 throughout |
| `mappo_ctx01_s0` | utilitarian / card-aware | 668 (2.7M), stopped to free CPU | 0.70 @100, 1.24 @200, 2.43 @600 |

## Hacks the policy found on its own

1. **Chip dumping (in-spec collusion).** By update 100 the BTN shoves ~25% of
   hands and the SB calls the shove with any card, including the weakest
   (continue rate 1.0 over 1,428 spots). The NE BB folds to two all-ins, so the
   team wins BB's blind while chips move only between teammates. Per-seat
   numbers look like a bad SB (-1.6 BB/hand) beside a great BTN (+2.6). Allowed
   by the utilitarian reward; the `smooth` objective (sum + min, already in the
   repo's MCCFR config) removes it (continue rate 0.0 over 435 spots).
2. **Opponent-table coverage farming.** The frozen MCCFR policy returns uniform
   random at infosets NE self-play never reached (`CFRTrainer` behavior). The
   team learned raise/limp lines that land the BB there: s0 checkpoint 300 drew
   40% of team EV from the 7% of hands with a fallback; checkpoint 400 drew 94%
   from 45% of hands (+3.35 BB/hand). Seed 1 reproduced it after update 600 and ended at +5.5 BB/hand with 67% of hands on a fallback and 97% of EV from them. The
   individual-objective control also started drifting into it by update 700.
3. **The card-aware fallback is also exploitable** (`mappo_ctx01_s0`): 24% of
   hands hit a fallback at update 600, 29% of EV. Any fixed fallback is a
   target; the real fix is an opponent that responds (co-evolution / PSRO).
   Some EV outside fallback hands may come from barely-visited infosets with
   noisy averages; not measured.

No hidden-card leakage, chip creation, termination or reward-delivery bug was
observed in the Leduc engine or wrapper (200k random hands + TouchStone suite).

## Bugs found in existing code (see TODO.md)

- BUG-5 (fixed): `--team-objective maxmin|smooth|risk` trained on individual payoff.
- BUG-6: `--decentralized` evaluation crashes on the current info-key format.
- BUG-7: uniform fallback above; MCCFR team results vs the frozen BB may share it (unverified).
- TEST-1: pre-existing failing test (wrong `train()` signature).

## TouchStone

Suite: `touchstone_suite/` (adapter shells into `.venv`; probes in
`poker_collusion/rl/probes.py`). Run `touchstone check touchstone_suite
--timeout 300`. `output_rl/current` is the checkpoint the checkpoint cases test.
Lifecycle evidence and product feedback:
`touchstone-field-trials/feedback/2026-11-06-poker.md`.
