"""
Strategy analysis: extract behavioral patterns from trained pkl files.
Produces three figures for the presentation slide 13.

Usage:
  python scripts/analyze_strategies.py
"""
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from poker_collusion.cfr.trainer import CFRTrainer
from poker_collusion.config import NUM_PLAYERS

OUTPUT_DIR = os.path.join(ROOT, "output")

PKLS = {
    "obs_signal":      os.path.join(OUTPUT_DIR, "leduc_obs_signal_1m.pkl"),
    "ctde":            os.path.join(OUTPUT_DIR, "leduc_ctde_1m.pkl"),
    "victim_static":   os.path.join(OUTPUT_DIR, "leduc_victim_no_comm_2.pkl"),
    "victim_coev":     os.path.join(OUTPUT_DIR, "coev_victim_2.pkl"),
}

# Action indices
FOLD   = 0
CALL   = 1
RAISES = list(range(2, 9))
ALLIN  = 9


def load_strategy_sum(path):
    """Return strategy_sum dict from a pkl file."""
    print(f"  Loading {os.path.basename(path)}...")
    trainer = CFRTrainer.__new__(CFRTrainer)
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return data.get("strategy_sum", data)
    # If it's a trainer object saved directly
    return trainer.strategy_sum if hasattr(trainer, "strategy_sum") else data


def load_trainer(path):
    print(f"  Loading {os.path.basename(path)}...")
    # Use a dummy game module for loading
    class DummyGame:
        pass
    trainer = CFRTrainer(DummyGame(), num_players=NUM_PLAYERS)
    trainer.load(path)
    return trainer


def normalized(arr):
    s = arr.sum()
    if s <= 0:
        return np.ones(len(arr)) / len(arr)
    return arr / s


def compute_raise_freq_by_bucket(strategy_sum, round_idx):
    """Compute mean raise frequency grouped by bucket for a given round."""
    bucket_raise = {}
    bucket_count = {}
    for key, arr in strategy_sum.items():
        if key[0] != round_idx:
            continue
        bucket = key[1]
        if isinstance(bucket, tuple):
            bucket = bucket[0]
        prob = normalized(arr)
        raise_freq = prob[RAISES].sum() + prob[ALLIN]
        bucket_raise[bucket] = bucket_raise.get(bucket, 0.0) + raise_freq
        bucket_count[bucket] = bucket_count.get(bucket, 0) + 1
    buckets = sorted(bucket_raise.keys())
    freqs = [bucket_raise[b] / bucket_count[b] for b in buckets]
    return buckets, freqs


def compute_fold_freq_by_bucket(strategy_sum, round_idx):
    """Compute mean fold frequency grouped by bucket for a given round."""
    bucket_fold = {}
    bucket_count = {}
    for key, arr in strategy_sum.items():
        if key[0] != round_idx:
            continue
        bucket = key[1]
        if isinstance(bucket, tuple):
            bucket = bucket[0]
        prob = normalized(arr)
        fold_freq = prob[FOLD]
        bucket_fold[bucket] = bucket_fold.get(bucket, 0.0) + fold_freq
        bucket_count[bucket] = bucket_count.get(bucket, 0) + 1
    buckets = sorted(bucket_fold.keys())
    freqs = [bucket_fold[b] / bucket_count[b] for b in buckets]
    return buckets, freqs


def compute_l1_divergence_by_bucket(ss_a, ss_b, round_idx):
    """Mean L1 distance between strategies a and b, grouped by bucket."""
    bucket_l1 = {}
    bucket_count = {}
    shared_keys = set(ss_a.keys()) & set(ss_b.keys())
    for key in shared_keys:
        if key[0] != round_idx:
            continue
        bucket = key[1]
        if isinstance(bucket, tuple):
            bucket = bucket[0]
        pa = normalized(ss_a[key])
        pb = normalized(ss_b[key])
        # Align lengths
        n = min(len(pa), len(pb))
        l1 = np.abs(pa[:n] - pb[:n]).sum()
        bucket_l1[bucket] = bucket_l1.get(bucket, 0.0) + l1
        bucket_count[bucket] = bucket_count.get(bucket, 0) + 1
    buckets = sorted(bucket_l1.keys())
    divergences = [bucket_l1[b] / bucket_count[b] for b in buckets]
    return buckets, divergences


def main():
    print("Loading strategy files...")
    trainers = {}
    for name, path in PKLS.items():
        if not os.path.exists(path):
            print(f"  MISSING: {path} — skipping")
            continue
        trainers[name] = load_trainer(path)
    print(f"Loaded: {list(trainers.keys())}\n")

    fig_paths = []

    # ── Figure 1: Raise frequency by bucket ──────────────────────────────────
    if "obs_signal" in trainers and "ctde" in trainers:
        print("Computing raise frequency by bucket...")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for ax, round_idx, round_name in zip(axes, [0, 1], ["Preflop", "Postflop"]):
            b_obs, f_obs = compute_raise_freq_by_bucket(
                trainers["obs_signal"].strategy_sum, round_idx)
            b_ctde, f_ctde = compute_raise_freq_by_bucket(
                trainers["ctde"].strategy_sum, round_idx)

            ax.plot(b_obs, f_obs, marker='o', markersize=4, label="Observable Signaling", color="#1a7a4a")
            ax.plot(b_ctde, f_ctde, marker='s', markersize=4, label="CTDE", color="#c0392b", linestyle='--')
            ax.set_title(round_name)
            ax.set_xlabel("Hand Bucket (low = weak, high = strong)")
            ax.set_ylabel("Raise Frequency")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

        fig.suptitle("Raise Frequency by Hand Bucket\nObservable Signaling vs. CTDE (Team Seats)", fontsize=12)
        fig.tight_layout()
        out = os.path.join(OUTPUT_DIR, "analysis_raise_freq.png")
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  Saved: {out}")
        fig_paths.append(out)
        plt.close(fig)
    else:
        print("Skipping raise frequency (missing pkls)")

    # ── Figure 2: Victim fold rate by bucket ─────────────────────────────────
    if "victim_static" in trainers and "victim_coev" in trainers:
        print("Computing victim fold rate by bucket...")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for ax, round_idx, round_name in zip(axes, [0, 1], ["Preflop", "Postflop"]):
            b_s, f_s = compute_fold_freq_by_bucket(
                trainers["victim_static"].strategy_sum, round_idx)
            b_c, f_c = compute_fold_freq_by_bucket(
                trainers["victim_coev"].strategy_sum, round_idx)

            ax.plot(b_s, f_s, marker='o', markersize=4, label="Static team opponent", color="#2563eb")
            ax.plot(b_c, f_c, marker='s', markersize=4, label="Co-evolved team opponent", color="#c0392b", linestyle='--')
            ax.set_title(round_name)
            ax.set_xlabel("Hand Bucket (low = weak, high = strong)")
            ax.set_ylabel("Fold Frequency")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

        fig.suptitle("Victim Fold Rate by Hand Bucket\nStatic Team Opponent vs. Co-Evolved Team Opponent", fontsize=12)
        fig.tight_layout()
        out = os.path.join(OUTPUT_DIR, "analysis_victim_fold.png")
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  Saved: {out}")
        fig_paths.append(out)
        plt.close(fig)
    else:
        print("Skipping victim fold rate (missing pkls)")

    # ── Figure 3: Strategy divergence obs_signal vs ctde ─────────────────────
    if "obs_signal" in trainers and "ctde" in trainers:
        print("Computing strategy divergence (L1)...")
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        for ax, round_idx, round_name in zip(axes, [0, 1], ["Preflop", "Postflop"]):
            buckets, divergences = compute_l1_divergence_by_bucket(
                trainers["obs_signal"].strategy_sum,
                trainers["ctde"].strategy_sum,
                round_idx,
            )
            ax.bar(buckets, divergences, color="#16213e", alpha=0.8)
            ax.set_title(round_name)
            ax.set_xlabel("Hand Bucket (low = weak, high = strong)")
            ax.set_ylabel("Mean L1 Distance")
            ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle("Strategy Divergence: Observable Signaling vs. CTDE\nMean L1 Distance per Hand Bucket", fontsize=12)
        fig.tight_layout()
        out = os.path.join(OUTPUT_DIR, "analysis_l1_divergence.png")
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  Saved: {out}")
        fig_paths.append(out)
        plt.close(fig)
    else:
        print("Skipping L1 divergence (missing pkls)")

    print(f"\nDone. {len(fig_paths)} figures saved:")
    for p in fig_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
