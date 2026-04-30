"""
Plot all training curve .txt files from a given directory, grouped by experiment type.
Usage:
  python scripts/plot_all_curves.py --input-dir outputv2 --out-dir outputv2/curves
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def load_curve(path):
    iterations, btn, sb, bb = [], [], [], []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            iterations.append(int(parts[0]))
            btn.append(float(parts[1]))
            sb.append(float(parts[2]))
            bb.append(float(parts[3]))
    return iterations, btn, sb, bb


SEAT_STYLES = {
    "BTN (seat 0)": ("#1f77b4", "o"),
    "SB (seat 1)":  ("#ff7f0e", "s"),
    "BB (seat 2)":  ("#d62728", "^"),
}

COMBO_LABELS = {"01": "Team [0,1] vs seat 2", "02": "Team [0,2] vs seat 1", "12": "Team [1,2] vs seat 0"}


def plot_group(files_by_combo, title, out_path):
    combos = sorted(files_by_combo.keys())
    fig, axes = plt.subplots(1, len(combos), figsize=(6 * len(combos), 5), sharey=False)
    if len(combos) == 1:
        axes = [axes]

    for ax, combo in zip(axes, combos):
        path = files_by_combo[combo]
        iters, btn, sb, bb = load_curve(path)
        iters_m = [x / 1e6 for x in iters]
        for (label, (color, marker)), vals in zip(SEAT_STYLES.items(), [btn, sb, bb]):
            ax.plot(iters_m, vals, color=color, marker=marker, markersize=3, label=label)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax.set_title(COMBO_LABELS.get(combo, combo), fontsize=11, fontweight="bold")
        ax.set_xlabel("Iterations (millions)")
        ax.set_ylabel("mbb/g")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputv2")
    parser.add_argument("--out-dir",   default="outputv2/curves")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    groups = {
        "obs_signal": {},
        "ctde":       {},
        "coev":       {},
    }

    for fname in os.listdir(args.input_dir):
        if not fname.endswith("_curve.txt"):
            continue
        path = os.path.join(args.input_dir, fname)
        for combo in ("01", "02", "12"):
            if f"obs_signal_{combo}" in fname:
                groups["obs_signal"][combo] = path
            elif f"ctde_{combo}" in fname:
                groups["ctde"][combo] = path
            elif f"coev_team_{combo}" in fname or f"coev_{combo}" in fname:
                groups["coev"][combo] = path

    specs = [
        ("obs_signal", "Observable Signaling — per-seat mbb/g vs frozen 4M opponent", "obs_signal_curves.png"),
        ("ctde",       "CTDE / Free Comm Training — per-seat mbb/g vs frozen 4M opponent",   "ctde_curves.png"),
        ("coev",       "Coevolution — per-seat mbb/g (team + victim co-training)",            "coev_curves.png"),
    ]

    for key, title, out_fname in specs:
        files = groups[key]
        if not files:
            print(f"No files found for group '{key}', skipping.")
            continue
        plot_group(files, title, os.path.join(args.out_dir, out_fname))


if __name__ == "__main__":
    main()
