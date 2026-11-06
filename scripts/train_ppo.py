"""
Train a MAPPO team (default BTN+SB) against a frozen MCCFR opponent.

  python scripts/train_ppo.py --opponent models/leduc_baseline_5000k_avg.pkl \
      --out output_rl/mappo_team01 --updates 500
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from poker_collusion.rl.env import LeducTeamEnv
from poker_collusion.rl.frozen import FrozenPolicy, compact_checkpoint
from poker_collusion.rl.ppo import PPOConfig, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", required=True, help="compact avg-strategy pickle (or full MCCFR pkl with --compact)")
    ap.add_argument("--compact", action="store_true", help="compact --opponent first (writes <name>_avg.pkl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--team-seats", default="0,1")
    ap.add_argument("--team-objective", default="utilitarian", choices=["utilitarian", "individual"])
    for name, default in vars(PPOConfig()).items():
        ap.add_argument(f"--{name.replace('_', '-')}", type=type(default), default=default)
    args = ap.parse_args()

    opp_path = args.opponent
    if args.compact:
        opp_path = os.path.splitext(args.opponent)[0] + "_avg.pkl"
        compact_checkpoint(args.opponent, opp_path)
    seats = [int(s) for s in args.team_seats.split(",")]
    cfg = PPOConfig(**{k: getattr(args, k) for k in vars(PPOConfig())})
    opp = FrozenPolicy(opp_path)
    env = LeducTeamEnv(seats, opp, args.team_objective)
    eval_env = LeducTeamEnv(seats, opp, args.team_objective)
    train(env, cfg, args.out, eval_env=eval_env)


if __name__ == "__main__":
    main()
