# Lessons Learned

Review this file at the start of every session. Update it after ANY correction from the user.
Write rules for yourself that prevent the same mistake. Ruthlessly iterate until mistake rate drops.

## Format

**Pattern**: [what happened]
**Rule**: [what to do / not do going forward]
**Why**: [root cause or user reasoning]

---

## Active Lessons

**Pattern**: CFR trained with pruning EV bug (values[pruned]=0 included in EV) produced a strategy that lost to amateur baseline. The fix is to renormalize strategy over non-pruned actions before computing EV.
**Rule**: When actions are pruned in cfr_traverse, compute EV only over non-pruned actions (renormalize strategy). Setting pruned values to 0 and including them in EV inflates/deflates the EV incorrectly. Blueprints trained with the bug must be discarded and retrained.
**Why**: Pruned actions have historically bad values (negative regret), so setting their traversal value to 0 biases EV. This cascades into corrupted regret updates across the linear phase (which dominates 99.5% of the average strategy when cutoff=1000 and total_iters=3450).

<!-- Add new lessons above this line. Remove lessons that have been internalized into config files. -->

**Pattern**: User asked about `regret_sum` action interpretability; I answered about `strategy_sum` and distribution sampling instead of mapping regret vector entries to action indices/legal actions.
**Rule**: When asked about interpretability of a specific table (`regret_sum` vs `strategy_sum`), explicitly state (a) how actions are encoded (vector index 0..9), (b) how legality is represented (`action_map`), and (c) how to read values for just the legal actions—before discussing derived policies.
**Why**: I generalized “infoset -> action” to strategy behavior and missed the user’s concrete question: “where are the actions for regret entries?”
