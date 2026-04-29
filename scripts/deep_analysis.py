"""
Deep strategy analysis: extract behavioral patterns from trained pkl files.
Compares NE baseline, observable signaling, and CTDE.

Usage:
  python3 scripts/deep_analysis.py
"""
import os, sys, pickle, numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from poker_collusion.cfr.trainer import CFRTrainer
from poker_collusion.config import NUM_PLAYERS

OUTPUT_DIR = os.path.join(ROOT, "output")
OLD_OUTPUT = os.path.join(ROOT, "old output")

PKLS = {
    "ne_baseline":  os.path.join(OLD_OUTPUT, "leduc_baseline_1000k.pkl"),
    "obs_signal":   os.path.join(OUTPUT_DIR, "leduc_obs_signal_1m.pkl"),
    "ctde":         os.path.join(OUTPUT_DIR, "leduc_ctde_1m.pkl"),
}

# Action indices
FOLD, CALL = 0, 1
RAISE_SIZES = list(range(2, 9))   # 7 raise sizes
ALLIN = 9
NUM_ACTIONS = 10

RANK_NAMES = {0: "J", 1: "Q", 2: "K", 3: "A"}


def load_trainer(path):
    print(f"  Loading {os.path.basename(path)}...")
    class G: pass
    t = CFRTrainer(G(), num_players=NUM_PLAYERS)
    t.load(path)
    return t


def avg_strategy(arr):
    s = arr.sum()
    if s <= 0:
        return np.ones(NUM_ACTIONS) / NUM_ACTIONS
    return arr / s


def parse_key(key):
    """Return (round_idx, bucket, history)."""
    return key[0], key[1], key[2]


def bucket_to_info(bucket, round_idx):
    """Return human-readable hand description."""
    if isinstance(bucket, tuple):
        b = bucket[0]
    else:
        b = bucket
    if round_idx == 0:
        return RANK_NAMES.get(b, str(b))
    else:
        hole = b // 4
        board = b % 4
        pair = " (PAIR)" if hole == board else ""
        return f"{RANK_NAMES.get(hole,'?')} hole / {RANK_NAMES.get(board,'?')} board{pair}"


def history_contains_raise(history):
    """True if any previous actor raised (action >= 2)."""
    for event in history:
        if isinstance(event, (list, tuple)) and len(event) == 2:
            actor, action = event
            if action >= 2:
                return True
    return False


def acting_player_from_history(history, round_idx):
    """Infer which player is about to act from history length."""
    # Preflop order: 0,1,2; postflop order: 1,2,0
    preflop_order = [0, 1, 2]
    postflop_order = [1, 2, 0]
    order = preflop_order if round_idx == 0 else postflop_order
    n_actions = sum(1 for e in history if isinstance(e, (list, tuple)) and len(e) == 2)
    return order[n_actions % 3]


# ─────────────────────────────────────────────────────────────────────────────
def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print("Loading strategies...")
    trainers = {}
    for name, path in PKLS.items():
        if os.path.exists(path):
            trainers[name] = load_trainer(path)
        else:
            print(f"  MISSING: {path}")

    names = list(trainers.keys())
    labels = {"ne_baseline": "NE Baseline", "obs_signal": "Obs Signaling", "ctde": "CTDE"}

    # ── 1. AGGREGATE ACTION FREQUENCIES ──────────────────────────────────────
    section("1. AGGREGATE ACTION FREQUENCIES (all info sets)")
    for name, trainer in trainers.items():
        totals = np.zeros(NUM_ACTIONS)
        count = 0
        for arr in trainer.strategy_sum.values():
            prob = avg_strategy(arr)
            totals += prob
            count += 1
        totals /= count
        raise_total = totals[RAISE_SIZES].sum()
        print(f"\n{labels[name]}:")
        print(f"  Fold:       {totals[FOLD]:.3f}")
        print(f"  Call:       {totals[CALL]:.3f}")
        print(f"  Raise (any):{raise_total:.3f}  {[f'{totals[i]:.3f}' for i in RAISE_SIZES]}")
        print(f"  All-in:     {totals[ALLIN]:.3f}")

    # ── 2. ACTION FREQUENCIES BY ROUND ───────────────────────────────────────
    section("2. ACTION FREQUENCIES BY ROUND")
    for round_idx, round_name in [(0, "Preflop"), (1, "Postflop")]:
        print(f"\n--- {round_name} ---")
        for name, trainer in trainers.items():
            totals = np.zeros(NUM_ACTIONS)
            count = 0
            for key, arr in trainer.strategy_sum.items():
                if key[0] != round_idx:
                    continue
                totals += avg_strategy(arr)
                count += 1
            if count == 0:
                continue
            totals /= count
            raise_total = totals[RAISE_SIZES].sum()
            print(f"  {labels[name]:20s}  fold={totals[FOLD]:.3f}  call={totals[CALL]:.3f}  raise={raise_total:.3f}  all-in={totals[ALLIN]:.3f}")

    # ── 3. RAISE FREQUENCY BY BUCKET ─────────────────────────────────────────
    section("3. RAISE + FOLD FREQUENCY BY HAND BUCKET")
    for round_idx, round_name in [(0, "Preflop"), (1, "Postflop")]:
        print(f"\n--- {round_name} ---")
        # collect per bucket
        bucket_data = {}  # name -> bucket -> [raise_freq, fold_freq, count]
        all_buckets = set()
        for name, trainer in trainers.items():
            bucket_data[name] = defaultdict(lambda: [0.0, 0.0, 0])
            for key, arr in trainer.strategy_sum.items():
                if key[0] != round_idx:
                    continue
                b = key[1][0] if isinstance(key[1], tuple) else key[1]
                prob = avg_strategy(arr)
                bucket_data[name][b][0] += prob[RAISE_SIZES].sum() + prob[ALLIN]
                bucket_data[name][b][1] += prob[FOLD]
                bucket_data[name][b][2] += 1
                all_buckets.add(b)

        for b in sorted(all_buckets):
            hand = bucket_to_info((b,), round_idx)
            row = f"  Bucket {b:2d} ({hand:25s})"
            for name in names:
                d = bucket_data[name][b]
                if d[2] > 0:
                    rf = d[0] / d[2]
                    ff = d[1] / d[2]
                    row += f"  {labels[name][:12]}: raise={rf:.3f} fold={ff:.3f}"
            print(row)

    # ── 4. BET SIZING PREFERENCES ────────────────────────────────────────────
    section("4. BET SIZING PREFERENCES (given raise decision)")
    # Preflop sizes: {2,2.5,3,4,5,8,12} BB; Postflop: {0.25,0.33,0.5,0.66,0.75,1,1.5} pot
    preflop_labels = ["2BB", "2.5BB", "3BB", "4BB", "5BB", "8BB", "12BB"]
    postflop_labels = ["25%", "33%", "50%", "66%", "75%", "100%", "150%"]

    for round_idx, round_name, size_labels in [
        (0, "Preflop", preflop_labels),
        (1, "Postflop", postflop_labels),
    ]:
        print(f"\n--- {round_name} (conditional on raising) ---")
        for name, trainer in trainers.items():
            totals = np.zeros(7)
            for key, arr in trainer.strategy_sum.items():
                if key[0] != round_idx:
                    continue
                prob = avg_strategy(arr)
                raise_mass = sum(prob[i] for i in RAISE_SIZES)
                if raise_mass > 1e-6:
                    for j, i in enumerate(RAISE_SIZES):
                        totals[j] += prob[i] / raise_mass
            totals /= max(1, sum(1 for k in trainer.strategy_sum if k[0] == round_idx))
            # normalize to get conditional distribution
            s = totals.sum()
            if s > 0:
                totals /= s
            sizes_str = "  ".join(f"{size_labels[j]}:{totals[j]:.2f}" for j in range(7))
            print(f"  {labels[name]:20s}  {sizes_str}")

    # ── 5. SQUEEZE DETECTION ─────────────────────────────────────────────────
    section("5. SQUEEZE DETECTION: P1 re-raise rate given P0 already raised (preflop)")
    # Look for preflop info sets where history shows P0 has already raised
    # and the current actor is P1 (SB)
    for name, trainer in trainers.items():
        total_p1_preflop = 0
        p1_raise_after_p0_raise = 0
        p1_total_after_p0_raise = 0
        p1_raise_no_prior = 0
        p1_total_no_prior = 0

        for key, arr in trainer.strategy_sum.items():
            round_idx, bucket, history = parse_key(key)
            if round_idx != 0:
                continue
            # Count action events in history
            action_events = [e for e in history if isinstance(e, (list, tuple)) and len(e) == 2]
            n = len(action_events)
            # Preflop order: P0, P1, P2, (P0, P1, P2 ...)
            # After n actions, it's player preflop_order[n % 3]
            preflop_order = [0, 1, 2]
            actor = preflop_order[n % 3]

            if actor != 1:
                continue  # only look at P1's decisions

            prob = avg_strategy(arr)
            raise_freq = prob[RAISE_SIZES].sum() + prob[ALLIN]
            p0_raised = any(
                (isinstance(e, (list, tuple)) and len(e) == 2 and e[0] == 0 and e[1] >= 2)
                for e in action_events
            )

            if p0_raised:
                p1_raise_after_p0_raise += raise_freq
                p1_total_after_p0_raise += 1
            else:
                p1_raise_no_prior += raise_freq
                p1_total_no_prior += 1

        r_after = p1_raise_after_p0_raise / p1_total_after_p0_raise if p1_total_after_p0_raise else 0
        r_before = p1_raise_no_prior / p1_total_no_prior if p1_total_no_prior else 0
        print(f"\n{labels[name]}:")
        print(f"  P1 raise rate WITHOUT prior P0 raise: {r_before:.3f}  (n={p1_total_no_prior})")
        print(f"  P1 raise rate AFTER P0 raised:        {r_after:.3f}  (n={p1_total_after_p0_raise})")
        squeeze_lift = r_after - r_before
        print(f"  Squeeze lift (after - before):        {squeeze_lift:+.3f}")

    # ── 6. P0 vs P1 STRATEGY DIVERGENCE (within team) ────────────────────────
    section("6. SEAT ASYMMETRY: Do P0 and P1 play differently within each strategy?")
    # Compare raise frequencies for P0 vs P1 at same bucket+round (aggregated over histories)
    for name, trainer in trainers.items():
        seat_raise = defaultdict(lambda: [0.0, 0])
        for key, arr in trainer.strategy_sum.items():
            round_idx, bucket, history = parse_key(key)
            action_events = [e for e in history if isinstance(e, (list, tuple)) and len(e) == 2]
            n = len(action_events)
            order = [0, 1, 2] if round_idx == 0 else [1, 2, 0]
            actor = order[n % 3]
            prob = avg_strategy(arr)
            raise_freq = prob[RAISE_SIZES].sum() + prob[ALLIN]
            seat_raise[actor][0] += raise_freq
            seat_raise[actor][1] += 1

        print(f"\n{labels[name]} — avg raise frequency by seat:")
        for seat in [0, 1, 2]:
            rf = seat_raise[seat][0] / seat_raise[seat][1] if seat_raise[seat][1] else 0
            seat_label = ["BTN (P0)", "SB (P1)", "BB (P2)"][seat]
            print(f"  {seat_label}: {rf:.3f}  (n={seat_raise[seat][1]})")

    # ── 7. POSTFLOP PAIR vs HIGH CARD BEHAVIOR ────────────────────────────────
    section("7. POSTFLOP: PAIR vs HIGH CARD BEHAVIOR")
    for name, trainer in trainers.items():
        pair_raise, pair_fold, pair_n = 0.0, 0.0, 0
        hc_raise, hc_fold, hc_n = 0.0, 0.0, 0
        for key, arr in trainer.strategy_sum.items():
            if key[0] != 1:
                continue
            b = key[1][0] if isinstance(key[1], tuple) else key[1]
            hole, board = b // 4, b % 4
            prob = avg_strategy(arr)
            rf = prob[RAISE_SIZES].sum() + prob[ALLIN]
            ff = prob[FOLD]
            if hole == board:
                pair_raise += rf; pair_fold += ff; pair_n += 1
            else:
                hc_raise += rf; hc_fold += ff; hc_n += 1

        pr = pair_raise / pair_n if pair_n else 0
        pf = pair_fold / pair_n if pair_n else 0
        hr = hc_raise / hc_n if hc_n else 0
        hf = hc_fold / hc_n if hc_n else 0
        print(f"\n{labels[name]}:")
        print(f"  Pair hands:      raise={pr:.3f}  fold={pf:.3f}  (n={pair_n})")
        print(f"  High card hands: raise={hr:.3f}  fold={hf:.3f}  (n={hc_n})")
        print(f"  Pair vs HC raise delta: {pr-hr:+.3f}")

    # ── 8. STRATEGY DIVERGENCE FROM NE ───────────────────────────────────────
    section("8. STRATEGY DIVERGENCE FROM NE BASELINE (mean L1 per round)")
    if "ne_baseline" in trainers:
        ne_ss = trainers["ne_baseline"].strategy_sum
        for name in ["obs_signal", "ctde"]:
            if name not in trainers:
                continue
            other_ss = trainers[name].strategy_sum
            shared = set(ne_ss.keys()) & set(other_ss.keys())
            l1_pre, l1_post, n_pre, n_post = 0.0, 0.0, 0, 0
            for key in shared:
                p_ne = avg_strategy(ne_ss[key])
                p_ot = avg_strategy(other_ss[key])
                n = min(len(p_ne), len(p_ot))
                l1 = np.abs(p_ne[:n] - p_ot[:n]).sum()
                if key[0] == 0:
                    l1_pre += l1; n_pre += 1
                else:
                    l1_post += l1; n_post += 1
            print(f"\n{labels[name]} vs NE:")
            print(f"  Preflop  mean L1: {l1_pre/n_pre:.4f}  (n={n_pre})")
            print(f"  Postflop mean L1: {l1_post/n_post:.4f}  (n={n_post})")
            print(f"  Shared keys: preflop={n_pre}, postflop={n_post}")
            total_ne = len([k for k in ne_ss if k[0] == 0])
            total_ot = len([k for k in other_ss if k[0] == 0])
            print(f"  NE preflop keys: {total_ne}, {labels[name]} preflop keys: {total_ot}")

    print("\n" + "=" * 70)
    print("Analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
