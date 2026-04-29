"""
Generate convergence curve plots from curve .txt files.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def load_curve(path):
    iterations, btn, sb, bb = [], [], [], []
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            iterations.append(int(parts[0]))
            btn.append(float(parts[1]))
            sb.append(float(parts[2]))
            bb.append(float(parts[3]))
    return iterations, btn, sb, bb


def plot_single(ax, path, label, color_btn, color_sb, color_bb, show_legend=True):
    iters, btn, sb, bb = load_curve(path)
    iters_m = [x / 1e6 for x in iters]
    ax.plot(iters_m, btn, color=color_btn, marker="o", markersize=3, label="BTN (seat 0)")
    ax.plot(iters_m, sb,  color=color_sb,  marker="s", markersize=3, label="SB (seat 1)")
    ax.plot(iters_m, bb,  color=color_bb,  marker="^", markersize=3, label="BB (seat 2)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlabel("Training iterations (millions)")
    ax.set_ylabel("mbb/g (self-play)")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    if show_legend:
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs-signal", default="output/leduc_obs_signal_1m_curve.txt")
    parser.add_argument("--ctde",       default="output/leduc_ctde_1m_curve.txt")
    parser.add_argument("--out",        default="output/convergence_curves.png")
    parser.add_argument("--combined",   action="store_true",
                        help="Also plot BTN+SB team sum on a separate axis")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    plot_single(axes[0], args.obs_signal,
                "Observable Signaling",
                "#1f77b4", "#ff7f0e", "#d62728")

    plot_single(axes[1], args.ctde,
                "CTDE (Full-Comm Training, Standard Deployment)",
                "#1f77b4", "#ff7f0e", "#d62728", show_legend=False)

    fig.suptitle("Per-Seat mbb/g Convergence (vs Frozen NE Opponent)", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved to {args.out}")

    if args.combined:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        for path, label, color in [
            (args.obs_signal, "Observable Signaling", "#1f77b4"),
            (args.ctde, "CTDE", "#ff7f0e"),
        ]:
            iters, btn, sb, bb = load_curve(path)
            iters_m = [x / 1e6 for x in iters]
            team_sum = [b + s for b, s in zip(btn, sb)]
            ax2.plot(iters_m, team_sum, label=label, color=color, marker="o", markersize=3)
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
        ax2.set_xlabel("Training iterations (millions)")
        ax2.set_ylabel("Team mbb/g (BTN + SB)")
        ax2.set_title("Team Combined mbb/g Convergence")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        out2 = args.out.replace(".png", "_team.png")
        fig2.savefig(out2, dpi=150)
        print(f"Saved team plot to {out2}")


if __name__ == "__main__":
    main()
