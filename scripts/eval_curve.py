#!/usr/bin/env python3
"""
Evaluate a sequence of blueprint pkl files vs the amateur strategy (CFR in
seat 0) and plot mbb/g as a learning curve.

Usage:
  python scripts/eval_curve.py output/bp_50.pkl output/bp_100.pkl ... \\
      [--hands 1000] [--workers 4] [--ci {95,se}]

Each pkl is evaluated in a separate process. Results are plotted in the order
the files were passed and saved as eval_curve.png next to this script.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import multiprocessing as mp
import os
import sys
from typing import List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Worker (runs in a child process)
# ---------------------------------------------------------------------------

def _evaluate_one(args: Tuple[int, str, int]) -> Tuple[int, float, float]:
    """Load one pkl and evaluate vs amateur in seat 0. Returns (idx, mbb, se)."""
    idx, pkl_path, num_hands = args

    # Suppress tqdm progress bars and the per-eval table printed by evaluate_vs_amateur.
    os.environ["TQDM_DISABLE"] = "1"

    import pickle
    import numpy as np

    # ── Validate ────────────────────────────────────────────────────────────
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"pkl not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid pkl (expected dict, got {type(data).__name__}): {pkl_path}")

    required = {"strategy_sum", "action_map"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Invalid pkl (missing keys {missing}): {pkl_path}")

    # ── Build game + strategy ───────────────────────────────────────────────
    import types
    from poker_collusion.cfr.strategy import Strategy
    from poker_collusion.evaluation.mbbg import evaluate_vs_amateur
    import poker_collusion.env as _env

    game = types.SimpleNamespace(
        deal_new_hand      = _env.deal_new_hand,
        get_current_player = _env.get_current_player,
        get_legal_actions  = _env.get_legal_actions,
        get_info_key       = _env.get_info_key,
        is_terminal        = _env.is_terminal,
        get_payoffs        = _env.get_payoffs,
        apply_action       = _env.apply_action,
        is_chance_node     = _env.is_chance_node,
        sample_chance      = _env.sample_chance,
    )

    strategy = Strategy(
        strategy_sum=data["strategy_sum"],
        action_map=data["action_map"],
    )

    # ── Evaluate (suppress printed output) ─────────────────────────────────
    silent = io.StringIO()
    with contextlib.redirect_stdout(silent):
        mbb_mean, mbb_se = evaluate_vs_amateur(
            game, strategy, num_hands=num_hands, cfr_seat=0
        )

    return idx, float(mbb_mean[0]), float(mbb_se[0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot mbb/g learning curve across a sequence of blueprint pkls"
    )
    ap.add_argument("pkls", nargs="+", help="Blueprint pkl files in evaluation order")
    ap.add_argument(
        "--hands", type=int, default=1000, metavar="N",
        help="Evaluation hands per pkl (default: 1000)",
    )
    ap.add_argument(
        "--workers", type=int, default=None, metavar="N",
        help="Parallel worker processes (default: number of pkls)",
    )
    ap.add_argument(
        "--ci", choices=["95", "se"], default="95",
        help="Error band: '95' for 95%% CI, 'se' for ±1 SE (default: 95)",
    )
    args = ap.parse_args()

    # Resolve paths relative to repo root
    pkl_paths: List[str] = [
        p if os.path.isabs(p) else os.path.join(ROOT, p)
        for p in args.pkls
    ]

    # Fail fast: check all files exist before spinning up workers
    bad = [p for p in pkl_paths if not os.path.isfile(p)]
    if bad:
        for p in bad:
            print(f"Error: not found: {p}", file=sys.stderr)
        sys.exit(1)

    n_workers = args.workers or len(pkl_paths)
    jobs = [(i, p, args.hands) for i, p in enumerate(pkl_paths)]

    print(
        f"Evaluating {len(pkl_paths)} checkpoint(s), "
        f"{args.hands} hands each, {n_workers} worker(s)..."
    )

    # ── Run evaluations in parallel ─────────────────────────────────────────
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    with mp.Pool(processes=n_workers) as pool:
        if use_tqdm:
            raw = list(
                tqdm(
                    pool.imap_unordered(_evaluate_one, jobs),
                    total=len(jobs),
                    desc="Evaluating",
                )
            )
        else:
            raw = pool.map(_evaluate_one, jobs)

    # Sort back into submission order
    raw.sort(key=lambda r: r[0])
    mbbs = [r[1] for r in raw]
    ses  = [r[2] for r in raw]

    # ── Print table ─────────────────────────────────────────────────────────
    print(f"\n{'File':<40} {'mbb/g':>8}  {'± SE':>8}")
    print("-" * 60)
    for i, (p, m, s) in enumerate(zip(pkl_paths, mbbs, ses)):
        print(f"{os.path.basename(p):<40} {m:>8.1f}  {s:>8.1f}")

    # ── Plot ────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\nmatplotlib not installed — skipping plot. Run: pip install matplotlib")
        return

    mbbs_arr = np.array(mbbs)
    ses_arr  = np.array(ses)
    xs = list(range(len(pkl_paths)))

    band = 1.96 * ses_arr if args.ci == "95" else ses_arr
    band_label = "95% CI" if args.ci == "95" else "±1 SE"

    fig, ax = plt.subplots(figsize=(max(8, len(pkl_paths) * 0.4 + 2), 5))

    ax.plot(xs, mbbs_arr, marker="o", markersize=4, linewidth=1.5,
            color="steelblue", label="mbb/g (CFR seat 0 vs amateur)")
    ax.fill_between(xs, mbbs_arr - band, mbbs_arr + band,
                    alpha=0.25, color="steelblue", label=band_label)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    # X-axis tick labels: show filename stems; thin out if many files
    stems = [os.path.splitext(os.path.basename(p))[0] for p in pkl_paths]
    step = max(1, math.ceil(len(stems) / 20))
    visible = list(range(0, len(stems), step))
    ax.set_xticks(visible)
    ax.set_xticklabels([stems[i] for i in visible], rotation=45, ha="right", fontsize=8)

    ax.set_xlabel("Checkpoint (order passed in)")
    ax.set_ylabel("mbb/g")
    ax.set_title(
        f"CFR vs Amateur — seat 0, {args.hands} hands per checkpoint  ({band_label})"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_curve.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()



"""
baseline (self play)
BTN    blueprint_v4_15000.pkl       200        148.2        [428.5, 1009.5]
SB     blueprint_v4_15000.pkl       -200         211.2        [-386.2, 441.8]
BB     blueprint_v4_15000.pkl       0       156.9        [-1054.3, -439.3]

team
BTN    blueprint_v4_15000.pkl       200        148.2        [428.5, 1009.5]
SB     blueprint_v4_15000.pkl       100         211.2        [-386.2, 441.8]
BB     blueprint_v4_15000.pkl       -300       156.9        [-1054.3, -439.3]
"""