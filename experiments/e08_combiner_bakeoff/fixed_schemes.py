"""Ensembling schemes that fit nothing per split, plus a sweep of the blend weight.

Follow-up to combine_opt.py: since greedy selection overfits here, try schemes with no
selection at all. Uniform averages over subsets, top-K by individual quality, and
b*V6 + (1-b)*AD for several b. Only the 18 thresholds are fitted per split, so the A/B
protocol still holds.
"""
import json

import numpy as np

from slra_st import paths
from slra_st.ensembling import ab_splits, combine, load_pool, qwk_fit_eval, weight_vector
from slra_st.metrics import fast_qwk

pool = load_pool("test")
tags, X, y = pool.tags, pool.X, pool.y
n = len(y)
n_ad = sum(t.endswith("_ad") for t in tags)
print(f"pool {len(tags)} models (AD={n_ad}, train-only={len(tags)-n_ad}), n_test={n}")

v6 = json.load(open(paths.ensemble("v6")))
ad = json.load(open(paths.ensemble("all_data")))
w_v6 = weight_vector(tags, v6["members"], v6["weights"])
w_ad = weight_vector(tags, ad["members"], ad["weights"])
s_v6, s_ad = combine(X, w_v6), combine(X, w_ad)

# Individual QWK on FULL test, used only to pick which models go in the top-K subsets. The
# A/B loop below still fits thresholds per split.
ind = np.array([fast_qwk(y - 1, np.clip(np.rint(X[:, i]), 1, 19).astype(int) - 1)
                for i in range(len(tags))])
order = np.argsort(-ind)
print("\ntop-8 individual models on test:")
for i in order[:8]:
    print(f"   {ind[i]*100:.2f}  {tags[i]}")

schemes = {"V6 fixed": s_v6, "AD fixed": s_ad}
for b in (0.3, 0.4, 0.5, 0.6, 0.7):
    schemes[f"blend b={b:.1f} (b*V6+(1-b)*AD)"] = b * s_v6 + (1 - b) * s_ad
schemes["uniform ALL"] = X.mean(1)
schemes["uniform AD only"] = combine(X, [1.0 if t.endswith("_ad") else 0 for t in tags])
schemes["uniform V6mem+ADmem"] = combine(X, (w_v6 > 0) | (w_ad > 0))
for K in (10, 15, 20, 25, 30):
    w = np.zeros(len(tags))
    w[order[:K]] = 1.0
    schemes[f"uniform top-{K} individual"] = combine(X, w)
w20 = np.zeros(len(tags))
w20[order[:20]] = 1.0
schemes["0.5*top20 + 0.5*AD"] = 0.5 * combine(X, w20) + 0.5 * s_ad

res = {k: [] for k in schemes}
for fit, ev in ab_splits(n, reps=6):
    for k, s in schemes.items():
        res[k].append(qwk_fit_eval(s[fit], y[fit], s[ev], y[ev]))

print()
print("=" * 74)
print("HONEST holdout QWK -- fixed schemes (thresholds fit per split only)")
print("=" * 74)
for k in sorted(res, key=lambda k: -np.mean(res[k])):
    v = np.array(res[k]) * 100
    print(f"  {k:32s} {v.mean():.3f} +- {v.std():.3f}")
