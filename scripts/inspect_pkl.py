#!/usr/bin/env python3
"""
Inspect a saved .pkl artifact (blueprint / checkpoint).

This repo uses pickle for strategy persistence. The common formats are:
- CFRTrainer.save(): dict with keys: regret_sum, strategy_sum, action_map, iteration, ...
- Strategy.load():   reads dict keys: strategy_sum, action_map (subset of above)

WARNING: pickle can execute arbitrary code when loading. Only open files you trust.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _is_mapping(x: Any) -> bool:
    return isinstance(x, dict)


def _fmt_int(x: int) -> str:
    return f"{x:,}"


def _safe_items(d: Dict[Any, Any], n: int) -> List[Tuple[Any, Any]]:
    if n <= 0:
        return []
    items = list(d.items())
    items.sort(key=lambda kv: repr(kv[0]))
    return items[:n]


def _to_floats(x: Any) -> List[float]:
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    try:
        # numpy arrays (or other array-likes) without importing numpy
        return [float(v) for v in x]  # type: ignore[assignment]
    except TypeError:
        return [float(x)]


def _avg_probs_from_strategy_sum(
    strategy_sum_row: Any, legal_actions: List[int]
) -> List[float]:
    row = _to_floats(strategy_sum_row)
    if not legal_actions:
        return []
    sub = [row[a] if a < len(row) else 0.0 for a in legal_actions]
    tot = float(sum(sub))
    if tot <= 0:
        return [1.0 / float(len(legal_actions)) for _ in legal_actions]
    return [v / tot for v in sub]


def _count_nonzero(xs: Sequence[float]) -> int:
    return sum(1 for v in xs if v != 0.0)


def _topk_indices(xs: Sequence[float], k: int) -> List[int]:
    return sorted(range(len(xs)), key=lambda i: xs[i], reverse=True)[:k]


def _summarize_table(
    title: str,
    table: Dict[Any, Any],
    action_map: Optional[Dict[Any, List[int]]],
    sample: int,
) -> None:
    print(f"\n== {title} ==")
    print(f"Entries: {_fmt_int(len(table))}")
    if not table:
        return

    shown = _safe_items(table, sample)
    for i, (k, v) in enumerate(shown):
        key_repr = repr(k)
        print(f"\n[{i}] key: {key_repr[:200]}{'...' if len(key_repr) > 200 else ''}")

        row = _to_floats(v)
        nz = _count_nonzero(row)
        s = float(sum(row))
        print(f"    row: len={len(row)}, nonzero={nz}, sum={s:.6g}")
        print(f"    vector: {row}")

        if action_map is not None and k in action_map:
            legal = list(action_map[k])
            legal_values = {int(a): float(row[a]) if a < len(row) else 0.0 for a in legal}
            print(f"    legal_actions: {legal}")
            print(f"    legal_values: {legal_values}")

            # For strategy_sum rows, show normalized action probabilities and argmax.
            probs = _avg_probs_from_strategy_sum(row, legal)
            top = _topk_indices(probs, min(5, len(probs)))
            top_str = ", ".join(f"a{legal[j]}={probs[j]:.3f}" for j in top)
            print(f"    top_probs: {top_str}")
            full = {int(a): float(p) for a, p in zip(legal, probs)}
            best_a = max(full.items(), key=lambda kv: kv[1])[0] if full else None
            print(f"    action_probs: {full}")
            print(f"    argmax_action: {best_a}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect a saved pickle (.pkl) artifact")
    ap.add_argument("path", help="Path to .pkl (relative to repo root or absolute)")
    ap.add_argument("--sample", type=int, default=3, help="How many sample infosets to print per table")
    ap.add_argument("--no-samples", action="store_true", help="Print only top-level metadata")
    args = ap.parse_args()

    path = args.path
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.isfile(full):
        print(f"Not found: {full}")
        sys.exit(1)

    try:
        with open(full, "rb") as f:
            obj = pickle.load(f)
    except ModuleNotFoundError as e:
        # Common: pickles contain numpy arrays and require numpy to unpickle.
        missing = getattr(e, "name", None) or str(e)
        print(f"Failed to load pickle because a module is missing: {missing}")
        print("This strategy/checkpoint pickle likely contains NumPy arrays.")
        print("Fix: install numpy in the Python environment you're using, then re-run:")
        print("  python3 -m pip install numpy")
        sys.exit(2)

    print("=" * 60)
    print("Pickle inspection")
    print("=" * 60)
    print(f"File: {path}")
    print(f"Resolved: {full}")
    print(f"Type: {type(obj)}")

    if not _is_mapping(obj):
        # Could be a raw object; just print repr slice.
        r = repr(obj)
        print(f"Repr: {r[:500]}{'...' if len(r) > 500 else ''}")
        return

    data: Dict[str, Any] = obj  # type: ignore[assignment]
    keys = sorted(list(data.keys()), key=str)
    print(f"Top-level keys: {keys}")

    if "iteration" in data:
        print(f"iteration: {data.get('iteration')}")
    if "linear_cfr_cutoff" in data:
        print(f"linear_cfr_cutoff: {data.get('linear_cfr_cutoff')}")

    action_map = data.get("action_map")
    if action_map is not None and not _is_mapping(action_map):
        print(f"action_map: unexpected type {type(action_map)} (expected dict)")
        action_map = None

    if args.no_samples:
        return

    sample = int(args.sample)
    if "strategy_sum" in data and _is_mapping(data["strategy_sum"]):
        _summarize_table(
            "strategy_sum (infoset -> action weight vector)",
            data["strategy_sum"],
            action_map,
            sample,
        )
    if "regret_sum" in data and _is_mapping(data["regret_sum"]):
        _summarize_table(
            "regret_sum (infoset -> regret vector)",
            data["regret_sum"],
            action_map,
            sample,
        )


if __name__ == "__main__":
    main()

