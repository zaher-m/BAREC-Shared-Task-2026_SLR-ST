"""Two more cheap end-game ideas, same A/B protocol on clean test.

1. BAGGED THRESHOLDS. The threshold fit is 18 parameters on ~3.6k points, so it has real
   variance. Average the thresholds over bootstrap resamples to cut it.
   Result: no gain. The QWK optimum in threshold space is sharp, so averaging over
   resamples smooths across it instead of toward it.

2. THREE-WAY BLENDS. The 50/50 V6+AD blend leads because the two families are decorrelated.
   Does a third group (large models only, or the older V5 ensemble) decorrelate further?
   Result: no. Every third group is correlated with one of the halves and waters down the
   equal split that was doing the work.
"""
import json

import numpy as np

from slra_st import paths
from slra_st.calibration import QWKThresholdOptimizer, labels_from_thresholds
from slra_st.ensembling import ab_splits, combine, load_pool, weight_vector
from slra_st.metrics import fast_qwk

pool = load_pool("test")
tags, X, y = pool.tags, pool.X, pool.y
N = len(y)


def scores_of(config):
    cfg = json.load(open(paths.ensemble(config)))
    return combine(X, weight_vector(tags, cfg["members"], cfg["weights"]))


s_v6, s_ad, s_v5 = scores_of("v6"), scores_of("all_data"), scores_of("v5")
large = [1.0 if "large" in t else 0.0 for t in tags]
s_lg = combine(X, large)
print(f"pool={len(tags)}  large-group members={int(sum(large))}")

s_blend = 0.5 * s_v6 + 0.5 * s_ad


def th_single(s, yy):
    return QWKThresholdOptimizer().fit(s, yy).thresholds_


def th_bagged(s, yy, n_boot=8, seed=0):
    r = np.random.RandomState(seed)
    acc = []
    for _ in range(n_boot):
        idx = r.randint(0, len(yy), len(yy))
        acc.append(QWKThresholdOptimizer().fit(s[idx], yy[idx]).thresholds_)
    return np.mean(acc, axis=0)


print()
print("=" * 72)
print("1. BAGGED vs SINGLE thresholds (on the 50/50 blend)")
print("=" * 72)
rs, rb = [], []
for i, (fit, ev) in enumerate(ab_splits(N, reps=6)):
    rs.append(fast_qwk(y[ev] - 1, labels_from_thresholds(s_blend[ev], th_single(s_blend[fit], y[fit])) - 1))
    # i // 2 -> one bootstrap seed per permutation, shared by both directions
    rb.append(fast_qwk(y[ev] - 1, labels_from_thresholds(s_blend[ev], th_bagged(s_blend[fit], y[fit], seed=i // 2)) - 1))
print(f"  single thresholds  {np.mean(rs)*100:.3f} +- {np.std(rs)*100:.3f}")
print(f"  bagged thresholds  {np.mean(rb)*100:.3f} +- {np.std(rb)*100:.3f}")

cands = {
    "0.5*V6 + 0.5*AD (current best)": s_blend,
    "1/3 V6 + 1/3 AD + 1/3 large": (s_v6 + s_ad + s_lg) / 3,
    "0.4 V6 + 0.4 AD + 0.2 large": 0.4 * s_v6 + 0.4 * s_ad + 0.2 * s_lg,
    "1/3 V6 + 1/3 AD + 1/3 V5": (s_v6 + s_ad + s_v5) / 3,
    "0.4 V6 + 0.4 AD + 0.2 V5": 0.4 * s_v6 + 0.4 * s_ad + 0.2 * s_v5,
    "0.25 V6+0.25 V5+0.5 AD": 0.25 * s_v6 + 0.25 * s_v5 + 0.5 * s_ad,
    "0.35 V6 + 0.45 AD + 0.2 large": 0.35 * s_v6 + 0.45 * s_ad + 0.2 * s_lg,
}
res = {k: [] for k in cands}
for fit, ev in ab_splits(N, reps=6):
    for k, s in cands.items():
        res[k].append(fast_qwk(y[ev] - 1, labels_from_thresholds(s[ev], th_single(s[fit], y[fit])) - 1))

print()
print("=" * 72)
print("2. STRUCTURED MULTI-WAY BLENDS (honest holdout QWK)")
print("=" * 72)
for k in sorted(res, key=lambda k: -np.mean(res[k])):
    v = np.array(res[k]) * 100
    print(f"  {k:34s} {v.mean():.3f} +- {v.std():.3f}")
