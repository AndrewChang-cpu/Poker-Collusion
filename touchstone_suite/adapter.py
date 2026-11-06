"""TouchStone adapter for the Leduc team RL environment.

TouchStone runs this file with its own interpreter from a temporary copy, so
the probes run in the project's venv via subprocess. Set POKER_COLLUSION_ROOT
if the repo lives elsewhere.

Case input = {"probe": NAME, ...}; observations are the probe's return dict
(poker_collusion/rl/probes.py):
- obs_counterfactual {hands, seed, checkpoint?}: resamples cards hidden from the
  acting seat -> obs_max_diff (actor obs change), critic_changed_frac (control),
  cfr_key_changed, psychic_key_changed_by_opponent_card / _by_teammate_card.
- conservation {hands, seed, opponent: uniform|frozen, checkpoint?} ->
  terminated, hands, max_abs_payoff_sum, min_final_stack, max_learner_steps.
- team_reward {component: env|cfr_trainer, objective, payoffs[3], team_seats,
  seat} -> reward computed by that component.
- terminal_delivery {hands, seed} -> ended_by_opponent, reward_mismatches.
- illegal_action {seed} -> raised, state_changed.
- checkpoint_replay {checkpoint, hands, seed} -> team_bb, team_se, seat_bb,
  max_abs_payoff_sum, opp_miss_rate, opp_fold_rate, reraise_vs_teammate,
  fold_to_teammate_raise, learner_decisions,
  weakest_card_facing_teammate_shove, weakest_card_continue_rate (chip-dump
  signature: continuing preflop with the weakest card after a teammate shove).
- decentralized_policy {hands, seed}: CTDE deployment wrapper -> errors,
  first_error, true_key_queried_frac, output_max_diff (teammate card resampled).
Every observation also carries code_version (git describe --dirty).
"""

import json
import os
import subprocess

ROOT = os.environ.get("POKER_COLLUSION_ROOT", os.path.expanduser("~/Documents/Coding/Poker-Collusion"))


def setup(case):
    return {"root": ROOT}


def exercise(context, inputs):
    proc = subprocess.run(
        [os.path.join(ROOT, ".venv", "bin", "python"), "-m", "poker_collusion.rl.probes"],
        input=json.dumps(inputs), capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ROOT}, timeout=280,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    context["observation"] = json.loads(proc.stdout)
    rev = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True, cwd=ROOT)
    context["observation"]["code_version"] = rev.stdout.strip()  # adapter hash alone misses env changes


def collect(context):
    return context["observation"]


def cleanup(context):
    pass
