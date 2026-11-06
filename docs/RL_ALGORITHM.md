# Deep RL policy for team Leduc: algorithm choice

## Setting

3-player Leduc (this repo's engine), seats 0 (BTN) and 1 (SB) form a team that
maximizes summed payoff; seat 2 (BB) is frozen to the 5M-iteration MCCFR
self-play average strategy. Fixing the opponent makes the learning problem a
cooperative, partially observable two-agent problem: a *team best response* to
a fixed opponent, the same question the existing MCCFR team runs ask
(`outputv2/leduc_obs_signal_*`, `leduc_ctde_*`). It is not an equilibrium
solve.

## Choice: MAPPO (shared PPO actor, centralized critic)

- Decentralized actor on own observation (own card, board, public betting,
  seat id), shared across both team seats. Centralized critic additionally sees
  every hole card during training only: centralized training, decentralized
  execution, the repo's "CTDE" rung, without the hand-written prior
  marginalization of `team_policy.py`.
- Action masking over the engine's legal actions; illegal actions raise.
- One terminal reward per hand, gamma = 1, so returns are the hand's team payoff
  (scaled 0.1); advantage = return - critic(global state).

Why:

1. **PG methods are competitive in imperfect-information games.** Rudolph et
   al. (ICLR 2026) tuned PPO, PPG and MMD against NFSP, PSRO, ESCHER and R-NaD
   with exact exploitability on five games and found the generic PG methods
   match or beat the specialized ones.
   [arXiv:2502.08938](https://arxiv.org/abs/2502.08938)
2. **PPO with a centralized value function is a strong cooperative MARL
   baseline**, including on the partially observable Hanabi benchmark
   (Yu et al., MAPPO). [arXiv:2103.01955](https://arxiv.org/abs/2103.01955)
3. **Our problem is a best response, not an equilibrium.** NFSP
   (Heinrich & Silver 2016, [arXiv:1603.01121](https://arxiv.org/abs/1603.01121))
   and PSRO (Lanctot et al. 2017, [arXiv:1711.00832](https://arxiv.org/abs/1711.00832))
   earn their extra machinery by averaging over a population to approach
   equilibrium. Against a fixed opponent that averaging buys nothing. PSRO is
   the natural *next* step if the BB is also trained: Team-PSRO (McAleer et al.,
   NeurIPS 2023) uses cooperative RL best responses to reach TMECor.
   [project page](https://www.mit.edu/~gfarina/2023/team_psro_neurips23)
4. **The team solution concept matches the repo.** Farina et al. (NeurIPS
   2018) model ex-ante coordinating colluders as one imperfect-recall
   meta-player; a shared actor trained on team reward is the RL analogue.
   [paper](https://papers.neurips.cc/paper_files/paper/2018/file/c17028c9b6e0c5deaad29665d582284a-Paper.pdf)

Alternatives not chosen: MMD (Sokota et al., ICLR 2023,
[arXiv:2206.05825](https://arxiv.org/abs/2206.05825)) is equally supported by
(1) and worth adding if the opponent is trained; Deep CFR
([arXiv:1811.00164](https://arxiv.org/abs/1811.00164)) duplicates the repo's
tabular MCCFR, which is exact enough for Leduc.

## Known limitations

- Best response to one frozen opponent: high team EV here measures
  exploitation of that opponent, not robustness. No exploitability is computed.
- Rudolph et al. stress entropy-coefficient tuning; we used 0.01 with no sweep
  and entropy collapses within ~200 updates.
- Own small PPO (CleanRL-style) instead of a library, because the engine is
  turn-based with in-wrapper opponents; RLCard/OpenSpiel wrappers would need
  this repo's 10-action abstraction re-implemented.
