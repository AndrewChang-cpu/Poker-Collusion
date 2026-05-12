#!/bin/bash
set -e
mkdir -p outputv2/logs

# ── Experiment 1: Nash convergence matrix (25 evals, seat 0 = row, seats 1&2 = col) ──

python3 scripts/evaluate.py --strategies models/leduc_baseline_100k.pkl models/leduc_baseline_100k.pkl models/leduc_baseline_100k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_100k_vs_100k.log
python3 scripts/evaluate.py --strategies models/leduc_baseline_100k.pkl models/leduc_baseline_1000k.pkl models/leduc_baseline_1000k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_100k_vs_1000k.log
python3 scripts/evaluate.py --strategies models/leduc_baseline_100k.pkl models/leduc_baseline_2000k.pkl models/leduc_baseline_2000k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_100k_vs_2000k.log
python3 scripts/evaluate.py --strategies models/leduc_baseline_100k.pkl models/leduc_baseline_3000k.pkl models/leduc_baseline_3000k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_100k_vs_3000k.log
python3 scripts/evaluate.py --strategies models/leduc_baseline_100k.pkl models/leduc_baseline_4000k.pkl models/leduc_baseline_4000k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_100k_vs_4000k.log

python3 scripts/evaluate.py --strategies models/leduc_baseline_1000k.pkl models/leduc_baseline_100k.pkl models/leduc_baseline_100k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_1000k_vs_100k.log
python3 scripts/evaluate.py --strategies models/leduc_baseline_1000k.pkl models/leduc_baseline_1000k.pkl models/leduc_baseline_1000k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_1000k_vs_1000k.log
python3 scripts/evaluate.py --strategies models/leduc_baseline_1000k.pkl models/leduc_baseline_2000k.pkl models/leduc_baseline_2000k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_1000k_vs_2000k.log
python3 scripts/evaluate.py --strategies models/leduc_baseline_1000k.pkl models/leduc_baseline_3000k.pkl models/leduc_baseline_3000k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_1000k_vs_3000k.log
python3 scripts/evaluate.py --strategies models/leduc_baseline_1000k.pkl models/leduc_baseline_4000k.pkl models/leduc_baseline_4000k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_1000k_vs_4000k.log

python3 scripts/evaluate.py --strategies models/leduc_baseline_2000k.pkl models/leduc_baseline_100k.pkl models/leduc_baseline_100k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_2000k_vs_100k.log
python3 scripts/evaluate.py --strategies models/leduc_baseline_2000k.pkl models/leduc_baseline_1000k.pkl models/leduc_baseline_1000k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_2000k_vs_1000k.log
python3 scripts/evaluate.py --strategies models/leduc_baseline_2000k.pkl models/leduc_baseline_2000k.pkl models/leduc_baseline_2000k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_2000k_vs_2000k.log
python3 scripts/evaluate.py --strategies models/leduc_baseline_2000k.pkl models/leduc_baseline_3000k.pkl models/leduc_baseline_3000k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_2000k_vs_3000k.log
python3 scripts/evaluate.py --strategies models/leduc_baseline_2000k.pkl models/leduc_baseline_4000k.pkl models/leduc_baseline_4000k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_2000k_vs_4000k.log

python3 scripts/evaluate.py --strategies models/leduc_baseline_3000k.pkl models/leduc_baseline_100k.pkl models/leduc_baseline_100k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_3000k_vs_100k.log
python3 scripts/evaluate.py --strategies models/leduc_baseline_3000k.pkl models/leduc_baseline_1000k.pkl models/leduc_baseline_1000k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_3000k_vs_1000k.log
python3 scripts/evaluate.py --strategies models/leduc_baseline_3000k.pkl models/leduc_baseline_2000k.pkl models/leduc_baseline_2000k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_3000k_vs_2000k.log
python3 scripts/evaluate.py --strategies models/leduc_baseline_3000k.pkl models/leduc_baseline_3000k.pkl models/leduc_baseline_3000k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_3000k_vs_3000k.log
python3 scripts/evaluate.py --strategies models/leduc_baseline_3000k.pkl models/leduc_baseline_4000k.pkl models/leduc_baseline_4000k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_3000k_vs_4000k.log

python3 scripts/evaluate.py --strategies models/leduc_baseline_4000k.pkl models/leduc_baseline_100k.pkl models/leduc_baseline_100k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_4000k_vs_100k.log
python3 scripts/evaluate.py --strategies models/leduc_baseline_4000k.pkl models/leduc_baseline_1000k.pkl models/leduc_baseline_1000k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_4000k_vs_1000k.log
python3 scripts/evaluate.py --strategies models/leduc_baseline_4000k.pkl models/leduc_baseline_2000k.pkl models/leduc_baseline_2000k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_4000k_vs_2000k.log
python3 scripts/evaluate.py --strategies models/leduc_baseline_4000k.pkl models/leduc_baseline_3000k.pkl models/leduc_baseline_3000k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_4000k_vs_3000k.log
python3 scripts/evaluate.py --strategies models/leduc_baseline_4000k.pkl models/leduc_baseline_4000k.pkl models/leduc_baseline_4000k.pkl --hands 100000 2>&1 | tee outputv2/logs/nash_4000k_vs_4000k.log

# ── Experiment 2: Observable signaling — train all 3 from scratch ───────────────────

python3 scripts/train.py --team-seats 0,1 --frozen-strategy models/leduc_baseline_5000k.pkl --iterations 5000000 --out outputv2/leduc_obs_signal_01_1m.pkl --plot-every 50000 2>&1 | tee outputv2/logs/train_obs_signal_01.log
python3 scripts/train.py --team-seats 0,2 --frozen-strategy models/leduc_baseline_5000k.pkl --iterations 5000000 --out outputv2/leduc_obs_signal_02_1m.pkl --plot-every 50000 2>&1 | tee outputv2/logs/train_obs_signal_02.log
python3 scripts/train.py --team-seats 1,2 --frozen-strategy models/leduc_baseline_5000k.pkl --iterations 5000000 --out outputv2/leduc_obs_signal_12_1m.pkl --plot-every 50000 2>&1 | tee outputv2/logs/train_obs_signal_12.log

# Exp 2 evals
python3 scripts/evaluate.py --team-eval --team-strategy outputv2/leduc_obs_signal_01_1m.pkl --frozen-strategy models/leduc_baseline_5000k.pkl --hands 100000 2>&1 | tee outputv2/logs/eval_obs_signal_01.log
python3 scripts/evaluate.py --team-eval --team-strategy outputv2/leduc_obs_signal_02_1m.pkl --frozen-strategy models/leduc_baseline_5000k.pkl --hands 100000 2>&1 | tee outputv2/logs/eval_obs_signal_02.log
python3 scripts/evaluate.py --team-eval --team-strategy outputv2/leduc_obs_signal_12_1m.pkl --frozen-strategy models/leduc_baseline_5000k.pkl --hands 100000 2>&1 | tee outputv2/logs/eval_obs_signal_12.log

# ── Experiment 3 & 4: CTDE — train all 3 from scratch ───────────────────────────────

python3 scripts/train.py --team-seats 0,1 --shared-info --frozen-strategy models/leduc_baseline_5000k.pkl --iterations 5000000 --out outputv2/leduc_ctde_01_1m.pkl --plot-every 50000 2>&1 | tee outputv2/logs/train_ctde_01.log
python3 scripts/train.py --team-seats 0,2 --shared-info --frozen-strategy models/leduc_baseline_5000k.pkl --iterations 5000000 --out outputv2/leduc_ctde_02_1m.pkl --plot-every 50000 2>&1 | tee outputv2/logs/train_ctde_02.log
python3 scripts/train.py --team-seats 1,2 --shared-info --frozen-strategy models/leduc_baseline_5000k.pkl --iterations 5000000 --out outputv2/leduc_ctde_12_1m.pkl --plot-every 50000 2>&1 | tee outputv2/logs/train_ctde_12.log

# Exp 3 evals: CTDE = decentralized execution (--no-shared-info at eval)
python3 scripts/evaluate.py --team-eval --team-strategy outputv2/leduc_ctde_01_1m.pkl --frozen-strategy models/leduc_baseline_5000k.pkl --no-shared-info --hands 100000 2>&1 | tee outputv2/logs/eval_ctde_01.log
python3 scripts/evaluate.py --team-eval --team-strategy outputv2/leduc_ctde_02_1m.pkl --frozen-strategy models/leduc_baseline_5000k.pkl --no-shared-info --hands 100000 2>&1 | tee outputv2/logs/eval_ctde_02.log
python3 scripts/evaluate.py --team-eval --team-strategy outputv2/leduc_ctde_12_1m.pkl --frozen-strategy models/leduc_baseline_5000k.pkl --no-shared-info --hands 100000 2>&1 | tee outputv2/logs/eval_ctde_12.log

# Exp 4 evals: Free communication = same models, shared info honored at eval
python3 scripts/evaluate.py --team-eval --team-strategy outputv2/leduc_ctde_01_1m.pkl --frozen-strategy models/leduc_baseline_5000k.pkl --hands 100000 2>&1 | tee outputv2/logs/eval_freecomm_01.log
python3 scripts/evaluate.py --team-eval --team-strategy outputv2/leduc_ctde_02_1m.pkl --frozen-strategy models/leduc_baseline_5000k.pkl --hands 100000 2>&1 | tee outputv2/logs/eval_freecomm_02.log
python3 scripts/evaluate.py --team-eval --team-strategy outputv2/leduc_ctde_12_1m.pkl --frozen-strategy models/leduc_baseline_5000k.pkl --hands 100000 2>&1 | tee outputv2/logs/eval_freecomm_12.log

# ── Experiment 5: Coevolution from scratch (outputs team + victim + curve txt) ──────

python3 scripts/train.py --coevolve --victim-seat 2 --iterations 4000000 --out outputv2/coev_team_01.pkl --victim-out outputv2/coev_victim_2.pkl --plot-every 100000 2>&1 | tee outputv2/logs/train_coev_01.log
python3 scripts/train.py --coevolve --victim-seat 1 --iterations 4000000 --out outputv2/coev_team_02.pkl --victim-out outputv2/coev_victim_1.pkl --plot-every 100000 2>&1 | tee outputv2/logs/train_coev_02.log
python3 scripts/train.py --coevolve --victim-seat 0 --iterations 4000000 --out outputv2/coev_team_12.pkl --victim-out outputv2/coev_victim_0.pkl --plot-every 100000 2>&1 | tee outputv2/logs/train_coev_12.log
