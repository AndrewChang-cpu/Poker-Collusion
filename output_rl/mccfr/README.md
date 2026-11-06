# Free-comm MCCFR team for the BUG-6 eval rerun (2026-11-06)

`scripts/train.py --iterations 200000 --team-seats 0,1 --frozen-strategy models/leduc_baseline_5000k.pkl --team-objective utilitarian --shared-info --out output_rl/mccfr/free_comm_200k.pkl` (~19 min; pkl not committed, 83 MB).

`scripts/evaluate.py --team-eval <mode> --team-strategy output_rl/mccfr/free_comm_200k.pkl --frozen-strategy models/leduc_baseline_5000k.pkl --hands 20000`, team BTN+SB mbb/g:

| Mode | Before fix | After fix |
| --- | --- | --- |
| `--shared-info` (free comm) | +263.4 ± 34.2 | unchanged |
| `--decentralized` | `ValueError` on first team decision | -675.9 ± 58.1 |

Reference: NE self-play team +155 ± 8. Marginalizing the free-comm strategy over the
teammate's card loses ~940 mbb/g versus free comm; whether that is under-training
(200k vs the 5M baselines) or the prior-only method (ignores teammate actions) is untested.
