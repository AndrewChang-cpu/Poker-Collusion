#!/usr/bin/env python3
"""
Build preflop and postflop bucket tables; save to data/.
Run from project root: python scripts/build_buckets.py
"""

import os
import sys
import argparse

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from poker_collusion.config import (
    DEFAULT_BUCKET_DIR,
    PREFLOP_BUCKETS_FILE,
    FLOP_BUCKETS_FILE,
    TURN_BUCKETS_FILE,
    RIVER_BUCKETS_FILE,
    PREFLOP_BUCKETS,
    FLOP_BUCKETS,
    TURN_BUCKETS,
    RIVER_BUCKETS,
)
from poker_collusion.bucketing_build.preflop_table import build_preflop_table
from poker_collusion.bucketing_build.postflop_table import (
    build_flop_table,
    build_turn_table,
    build_river_table,
)


def main():
    ap = argparse.ArgumentParser(description="Build bucket tables for NLHE abstraction")
    ap.add_argument("--out-dir", default=DEFAULT_BUCKET_DIR, help="Output directory")
    ap.add_argument("--preflop-rollouts", type=int, default=500, help="MC rollouts per preflop hand")
    ap.add_argument("--postflop-samples", type=int, default=5000, help="Postflop samples (reduce for quick test)")
    ap.add_argument("--postflop-rollouts", type=int, default=200, help="MC rollouts per postflop sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    import pickle

    print("Building preflop table (169 canonical -> {} buckets)...".format(PREFLOP_BUCKETS))
    preflop = build_preflop_table(n_rollouts=args.preflop_rollouts)
    path = os.path.join(out_dir, PREFLOP_BUCKETS_FILE)
    with open(path, "wb") as f:
        pickle.dump(preflop, f)
    print(f"  Saved {path}")

    print("Building flop table (k-means {} clusters)...".format(FLOP_BUCKETS))
    flop_centers = build_flop_table(
        n_samples=args.postflop_samples,
        n_rollouts=args.postflop_rollouts,
        n_clusters=FLOP_BUCKETS,
        seed=args.seed,
    )
    path = os.path.join(out_dir, FLOP_BUCKETS_FILE)
    with open(path, "wb") as f:
        pickle.dump(flop_centers, f)
    print(f"  Saved {path}")

    print("Building turn table...")
    turn_centers = build_turn_table(
        n_samples=args.postflop_samples,
        n_rollouts=args.postflop_rollouts,
        n_clusters=TURN_BUCKETS,
        seed=args.seed,
    )
    path = os.path.join(out_dir, TURN_BUCKETS_FILE)
    with open(path, "wb") as f:
        pickle.dump(turn_centers, f)
    print(f"  Saved {path}")

    print("Building river table...")
    river_centers = build_river_table(
        n_samples=args.postflop_samples,
        n_rollouts=args.postflop_rollouts,
        n_clusters=RIVER_BUCKETS,
        seed=args.seed,
    )
    path = os.path.join(out_dir, RIVER_BUCKETS_FILE)
    with open(path, "wb") as f:
        pickle.dump(river_centers, f)
    print(f"  Saved {path}")

    print("Done.")


if __name__ == "__main__":
    main()
