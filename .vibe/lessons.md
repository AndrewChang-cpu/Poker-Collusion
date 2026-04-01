# Lessons Learned

Review this file at the start of every session. Update it after ANY correction from the user.
Write rules for yourself that prevent the same mistake. Ruthlessly iterate until mistake rate drops.

## Format

**Pattern**: [what happened]
**Rule**: [what to do / not do going forward]
**Why**: [root cause or user reasoning]

---

## Active Lessons

**Pattern**: CFR trained with pruning EV bug (values[pruned]=0 included in EV) produced a strategy that lost to amateur baseline. The fix is to renormalize strategy over non-pruned actions before computing EV. The bug was originally fixed in cfr_traverse (serial) but left unfixed in _cfr_traverse_local (parallel) — both paths must be kept in sync.
**Rule**: When actions are pruned in cfr_traverse or _cfr_traverse_local, compute EV only over non-pruned actions (renormalize strategy). Setting pruned values to 0 and including them in EV inflates/deflates the EV incorrectly. Blueprints trained with the bug must be discarded and retrained. When fixing a bug that exists in both serial and parallel traversal paths, always fix both.
**Why**: Pruned actions have historically bad values (negative regret), so setting their traversal value to 0 biases EV. This cascades into corrupted regret updates. The parallel path is a copy of the serial path with thread-local delta dicts — any algorithmic fix to one must be mirrored in the other.

<!-- Add new lessons above this line. Remove lessons that have been internalized into config files. -->

**Pattern**: User asked about `regret_sum` action interpretability; I answered about `strategy_sum` and distribution sampling instead of mapping regret vector entries to action indices/legal actions.
**Rule**: When asked about interpretability of a specific table (`regret_sum` vs `strategy_sum`), explicitly state (a) how actions are encoded (vector index 0..9), (b) how legality is represented (`action_map`), and (c) how to read values for just the legal actions—before discussing derived policies.
**Why**: I generalized “infoset -> action” to strategy behavior and missed the user’s concrete question: “where are the actions for regret entries?”
