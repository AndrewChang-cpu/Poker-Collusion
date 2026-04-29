"""
Signaling mechanism analysis: how does P0 signal to P1 through preflop actions?

Isolates:
  - P0's first-action distribution by bucket (NE vs obs_signal)
  - P1's response to each P0 action, by P1's bucket (NE vs obs_signal)
  - Mutual information between P0 action and P1 action
"""
import os, sys, numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from poker_collusion.cfr.trainer import CFRTrainer
from poker_collusion.config import NUM_PLAYERS

OUTPUT_DIR  = os.path.join(ROOT, "output")
OLD_OUTPUT  = os.path.join(ROOT, "old output")

PKLS = {
    "ne_baseline": os.path.join(OLD_OUTPUT, "leduc_baseline_1000k.pkl"),
    "obs_signal":  os.path.join(OUTPUT_DIR, "leduc_obs_signal_1m.pkl"),
    "ctde":        os.path.join(OUTPUT_DIR, "leduc_ctde_1m.pkl"),
}

FOLD, CALL = 0, 1
RAISES = list(range(2, 9))
ALLIN  = 9
RANK   = {0: "J", 1: "Q", 2: "K", 3: "A"}


def load(path):
    print(f"  Loading {os.path.basename(path)}...")
    class G: pass
    t = CFRTrainer(G(), num_players=NUM_PLAYERS)
    t.load(path)
    return t


def norm(arr):
    s = arr.sum()
    return arr / s if s > 0 else np.ones(len(arr)) / len(arr)


def action_events(history):
    return [e for e in history if isinstance(e, (list, tuple)) and len(e) == 2]


def section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print("Loading...")
    trainers = {k: load(p) for k, p in PKLS.items() if os.path.exists(p)}
    labels = {"ne_baseline": "NE Baseline", "obs_signal": "Obs Signaling", "ctde": "CTDE"}

    # ── 1. P0 FIRST-ACTION DISTRIBUTION BY BUCKET ────────────────────────────
    # Filter: round=0, zero prior action events → P0 is first to act
    section("1. P0 FIRST-ACTION DISTRIBUTION BY BUCKET (preflop, no prior actions)")
    print("   (raise = any of the 7 raise sizes + all-in)")

    p0_data = {}  # name -> bucket -> strategy array
    for name, trainer in trainers.items():
        p0_data[name] = defaultdict(list)
        for key, arr in trainer.strategy_sum.items():
            if key[0] != 0:
                continue
            evts = action_events(key[2])
            if len(evts) != 0:
                continue   # only want P0's very first decision
            b = key[1][0] if isinstance(key[1], tuple) else key[1]
            p0_data[name][b].append(norm(arr))

    # Average over multiple info sets at the same bucket
    for name in trainers:
        print(f"\n{labels[name]}:")
        print(f"  {'Hand':6s}  {'Fold':6s}  {'Call':6s}  {'Raise':6s}  {'All-in':7s}")
        for b in sorted(p0_data[name]):
            arrs = np.array(p0_data[name][b])
            avg = arrs.mean(axis=0)
            raise_total = avg[RAISES].sum()
            print(f"  {RANK[b]:6s}  {avg[FOLD]:.3f}   {avg[CALL]:.3f}   {raise_total:.3f}   {avg[ALLIN]:.3f}")

    # ── 2. P0 FIRST-ACTION: SIGNAL BANDWIDTH ─────────────────────────────────
    # How much does P0's action distribution VARY across buckets?
    # High variance = informative signal. Compare NE vs obs_signal.
    section("2. SIGNAL BANDWIDTH: how much does P0's action vary across buckets?")
    print("   TV distance = max difference in raise rate between strongest/weakest hand")
    for name in trainers:
        buckets = sorted(p0_data[name].keys())
        raise_by_bucket = {}
        for b in buckets:
            arrs = np.array(p0_data[name][b])
            avg = arrs.mean(axis=0)
            raise_by_bucket[b] = avg[RAISES].sum() + avg[ALLIN]
        lo = raise_by_bucket[min(buckets)]
        hi = raise_by_bucket[max(buckets)]
        spread = hi - lo
        print(f"\n{labels[name]}:")
        for b in buckets:
            bar = "#" * int(raise_by_bucket[b] * 40)
            print(f"  {RANK[b]} raise rate: {raise_by_bucket[b]:.3f}  {bar}")
        print(f"  Spread (A - J): {spread:+.3f}")

    # ── 3. P1 RESPONSE TO EACH P0 ACTION, BY P1 BUCKET ───────────────────────
    # Filter: round=0, exactly 1 prior action event, that event is P0 (actor=0)
    section("3. P1 RESPONSE TO P0's ACTION, BY P1 HAND (preflop)")

    # We bucket P0's action into 3 categories: fold, call, raise/all-in
    def p0_category(action):
        if action == FOLD: return "P0_fold"
        if action == CALL: return "P0_call"
        return "P0_raise"

    p1_data = {}  # name -> p0_cat -> p1_bucket -> [raise_rate, fold_rate, count]
    for name, trainer in trainers.items():
        p1_data[name] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))
        for key, arr in trainer.strategy_sum.items():
            if key[0] != 0:
                continue
            evts = action_events(key[2])
            if len(evts) != 1:
                continue
            actor, action = evts[0]
            if actor != 0:
                continue   # must be P0's action
            p0_cat = p0_category(action)
            b = key[1][0] if isinstance(key[1], tuple) else key[1]
            prob = norm(arr)
            p1_data[name][p0_cat][b][0] += prob[RAISES].sum() + prob[ALLIN]
            p1_data[name][p0_cat][b][1] += prob[FOLD]
            p1_data[name][p0_cat][b][2] += 1

    cats = ["P0_fold", "P0_call", "P0_raise"]
    for name in trainers:
        print(f"\n{labels[name]}:")
        for cat in cats:
            print(f"\n  When {cat}:")
            print(f"    {'P1 hand':8s}  {'raise':6s}  {'fold':6s}  {'n':6s}")
            for b in sorted(p1_data[name][cat]):
                d = p1_data[name][cat][b]
                if d[2] == 0:
                    continue
                rf = d[0] / d[2]
                ff = d[1] / d[2]
                print(f"    {RANK[b]:8s}  {rf:.3f}   {ff:.3f}   {d[2]}")

    # ── 4. COORDINATION LIFT: obs_signal vs NE, ctde vs NE ───────────────────
    section("4. COORDINATION LIFT: P1 raise rate (obs_signal vs NE, ctde vs NE)")
    print("   Positive = raises MORE than NE in this situation")
    print("   Key: does CTDE show same lift? If not, signaling is obs_signal-specific.")
    for cat in cats:
        print(f"\n  When {cat}:")
        print(f"    {'P1 hand':8s}  {'NE':7s}  {'Obs':7s}  {'Obs lift':9s}  {'CTDE':7s}  {'CTDE lift':9s}")
        all_buckets = (set(p1_data.get("ne_baseline", {}).get(cat, {}).keys()) |
                       set(p1_data.get("obs_signal",  {}).get(cat, {}).keys()) |
                       set(p1_data.get("ctde",        {}).get(cat, {}).keys()))
        for b in sorted(all_buckets):
            ne_d   = p1_data.get("ne_baseline", {}).get(cat, {}).get(b, [0,0,0])
            obs_d  = p1_data.get("obs_signal",  {}).get(cat, {}).get(b, [0,0,0])
            ctde_d = p1_data.get("ctde",        {}).get(cat, {}).get(b, [0,0,0])
            ne_r   = ne_d[0]   / ne_d[2]   if ne_d[2]   > 0 else float("nan")
            obs_r  = obs_d[0]  / obs_d[2]  if obs_d[2]  > 0 else float("nan")
            ctde_r = ctde_d[0] / ctde_d[2] if ctde_d[2] > 0 else float("nan")
            obs_lift  = obs_r  - ne_r
            ctde_lift = ctde_r - ne_r
            flag = " <--" if abs(obs_lift) > 0.05 or abs(ctde_lift) > 0.05 else ""
            print(f"    {RANK[b]:8s}  {ne_r:7.3f}  {obs_r:7.3f}  {obs_lift:+9.3f}  {ctde_r:7.3f}  {ctde_lift:+9.3f}{flag}")

    # ── 4b. CTDE P1 RAISE RATE CONDITIONED ON P1'S OWN HAND AFTER P0 RAISED ──
    # Hypothesis: CTDE P1 raise rate tracks P1's own hand (has joint info).
    # Obs signal P1 raise rate is flat — P1 ignores own hand, responds to signal.
    section("4b. SIGNAL VS. JOINT INFO: Does P1's own hand predict its raise rate?")
    print("   After P0 raised: is P1's raise rate flat (signaling) or hand-stratified (joint info)?")
    print("   Flat = P1 ignores own hand = pure signal response")
    print("   Stratified = P1 uses own hand = direct joint coordination")
    cat = "P0_raise"
    print(f"\n  {'P1 hand':8s}  {'NE':7s}  {'Obs':7s}  {'CTDE':7s}")
    all_buckets = (set(p1_data.get("ne_baseline", {}).get(cat, {}).keys()) |
                   set(p1_data.get("obs_signal",  {}).get(cat, {}).keys()) |
                   set(p1_data.get("ctde",        {}).get(cat, {}).keys()))
    obs_rates, ctde_rates = [], []
    for b in sorted(all_buckets):
        ne_d   = p1_data.get("ne_baseline", {}).get(cat, {}).get(b, [0,0,0])
        obs_d  = p1_data.get("obs_signal",  {}).get(cat, {}).get(b, [0,0,0])
        ctde_d = p1_data.get("ctde",        {}).get(cat, {}).get(b, [0,0,0])
        ne_r   = ne_d[0]   / ne_d[2]   if ne_d[2]   > 0 else float("nan")
        obs_r  = obs_d[0]  / obs_d[2]  if obs_d[2]  > 0 else float("nan")
        ctde_r = ctde_d[0] / ctde_d[2] if ctde_d[2] > 0 else float("nan")
        print(f"  {RANK[b]:8s}  {ne_r:7.3f}  {obs_r:7.3f}  {ctde_r:7.3f}")
        if not np.isnan(obs_r):  obs_rates.append(obs_r)
        if not np.isnan(ctde_r): ctde_rates.append(ctde_r)
    if len(obs_rates) > 1:
        print(f"\n  Obs raise rate std across P1 hands (flat=0): {np.std(obs_rates):.4f}")
    if len(ctde_rates) > 1:
        print(f"  CTDE raise rate std across P1 hands (flat=0): {np.std(ctde_rates):.4f}")

    # ── 5. MUTUAL INFORMATION: P0 action <-> P1 action ───────────────────────
    # Compute joint distribution P(P0_action, P1_action) and MI
    section("5. MUTUAL INFORMATION between P0 action and P1 action (preflop)")
    print("   Higher MI = more coordination between teammates")

    for name, trainer in trainers.items():
        # P0 marginal (3 categories: fold/call/raise) × P1 marginal
        joint = defaultdict(lambda: defaultdict(float))
        p0_marginal = defaultdict(float)
        p1_marginal = defaultdict(float)
        total_weight = 0.0

        # Get P0 distributions first
        p0_dist_by_bucket = {}
        for key, arr in trainer.strategy_sum.items():
            if key[0] != 0: continue
            evts = action_events(key[2])
            if len(evts) != 0: continue
            b = key[1][0] if isinstance(key[1], tuple) else key[1]
            if b not in p0_dist_by_bucket:
                p0_dist_by_bucket[b] = norm(arr)
            else:
                p0_dist_by_bucket[b] = (p0_dist_by_bucket[b] + norm(arr)) / 2

        # Get P1 distributions given P0 action
        p1_given_p0 = defaultdict(lambda: defaultdict(list))
        for key, arr in trainer.strategy_sum.items():
            if key[0] != 0: continue
            evts = action_events(key[2])
            if len(evts) != 1: continue
            actor, action = evts[0]
            if actor != 0: continue
            p0_cat = p0_category(action)
            b = key[1][0] if isinstance(key[1], tuple) else key[1]
            p1_given_p0[p0_cat][b].append(norm(arr))

        # Compute MI using uniform bucket prior
        buckets = sorted(p0_dist_by_bucket.keys())
        n_buckets = len(buckets)
        if n_buckets == 0:
            continue

        # Build joint: P(p0_action, p1_action) = mean over buckets of P0(action|bucket) * P1(action|p0_action, bucket)
        joint_counts = np.zeros((3, 3))  # [fold/call/raise] x [fold/call/raise]
        action_cats = ["fold", "call", "raise"]

        def action_to_cat_idx(a):
            if a == FOLD: return 0
            if a == CALL: return 1
            return 2

        for b in buckets:
            if b not in p0_dist_by_bucket:
                continue
            p0_prob = p0_dist_by_bucket[b]
            # P0 category probs
            p0_cat_prob = [p0_prob[FOLD], p0_prob[CALL], p0_prob[RAISES].sum() + p0_prob[ALLIN]]

            for ci, cat in enumerate(["P0_fold", "P0_call", "P0_raise"]):
                if cat not in p1_given_p0 or b not in p1_given_p0[cat]:
                    continue
                p1_arrs = p1_given_p0[cat][b]
                p1_avg = np.array(p1_arrs).mean(axis=0)
                p1_cat_prob = [p1_avg[FOLD], p1_avg[CALL], p1_avg[RAISES].sum() + p1_avg[ALLIN]]
                for ri in range(3):
                    for pi in range(3):
                        joint_counts[ci, ri] += p0_cat_prob[ci] * p1_cat_prob[pi]

        joint_counts /= joint_counts.sum()
        p0_marg = joint_counts.sum(axis=1)
        p1_marg = joint_counts.sum(axis=0)

        mi = 0.0
        for i in range(3):
            for j in range(3):
                if joint_counts[i,j] > 1e-10 and p0_marg[i] > 1e-10 and p1_marg[j] > 1e-10:
                    mi += joint_counts[i,j] * np.log2(joint_counts[i,j] / (p0_marg[i] * p1_marg[j]))

        print(f"\n{labels[name]}:")
        print(f"  Mutual information P0<->P1: {mi:.4f} bits")
        print(f"  Joint distribution P(P0_action, P1_action):")
        print(f"  {'':12s}  {'P1 fold':10s}  {'P1 call':10s}  {'P1 raise':10s}")
        for i, cat in enumerate(["P0 fold", "P0 call", "P0 raise"]):
            row = "  " + f"{cat:12s}"
            for j in range(3):
                row += f"  {joint_counts[i,j]:.4f}    "
            print(row)

    # ── 6. WHY CTDE UNDERPERFORMS: POSTFLOP HAND STRENGTH EXPLOITATION ───────
    section("6. WHY CTDE UNDERPERFORMS: POSTFLOP RAISE RATE BY HAND STRENGTH")
    print("   Does CTDE collapse pair vs. high-card distinction postflop?")
    print("   NE/obs_signal should raise much more with pairs; CTDE may not.")
    RANK_POST = {}
    for hole in range(4):
        for board in range(4):
            b = hole * 4 + board
            pair = hole == board
            RANK_POST[b] = (f"{RANK[hole]}/{RANK[board]}", pair)

    for name, trainer in trainers.items():
        pair_r, pair_n = 0.0, 0
        hc_r, hc_n = 0.0, 0
        for key, arr in trainer.strategy_sum.items():
            if key[0] != 1:
                continue
            b = key[1][0] if isinstance(key[1], tuple) else key[1]
            if b not in RANK_POST:
                continue
            _, is_pair = RANK_POST[b]
            prob = norm(arr)
            rf = prob[RAISES].sum() + prob[ALLIN]
            if is_pair:
                pair_r += rf; pair_n += 1
            else:
                hc_r += rf; hc_n += 1
        pr = pair_r / pair_n if pair_n else 0
        hr = hc_r / hc_n if hc_n else 0
        print(f"\n{labels[name]}:")
        print(f"  Pair raise rate:      {pr:.3f}  (n={pair_n})")
        print(f"  High-card raise rate: {hr:.3f}  (n={hc_n})")
        print(f"  Delta (pair - HC):    {pr - hr:+.3f}  {'<-- COLLAPSED' if abs(pr-hr) < 0.05 else ''}")

    # ── 7. CTDE POSTFLOP PASSIVITY: CALL vs RAISE RATES ──────────────────────
    section("7. CTDE POSTFLOP PASSIVITY: action frequencies by round")
    print("   Does CTDE call more and raise less postflop, limiting profit extraction?")
    for round_idx, rname in [(0, "Preflop"), (1, "Postflop")]:
        print(f"\n  {rname}:")
        print(f"  {'Strategy':20s}  {'fold':7s}  {'call':7s}  {'raise':7s}  {'all-in':7s}")
        for name, trainer in trainers.items():
            totals = np.zeros(10)
            count = 0
            for key, arr in trainer.strategy_sum.items():
                if key[0] != round_idx:
                    continue
                totals += norm(arr)
                count += 1
            if count == 0:
                continue
            totals /= count
            rsum = totals[RAISES].sum()
            print(f"  {labels[name]:20s}  {totals[FOLD]:.3f}    {totals[CALL]:.3f}    {rsum:.3f}    {totals[ALLIN]:.3f}")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
