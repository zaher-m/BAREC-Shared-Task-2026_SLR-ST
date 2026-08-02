"""Which way of combining the train-only and all-data families generalizes best?

TEST is the only split clean for both families, so everything happens there with an A/B
split: fit on one half, score the other half, both directions, several permutations. What is
being measured is the procedure, not one fitted ensemble.

Schemes: V6 fixed, AD fixed, 50/50 blend of the two, greedy over the whole pool, greedy over
the AD models only.

Result: the fixed 50/50 blend scores highest (86.94). Greedy re-selection over 49 models overfits the
selection split (86.60) and a uniform average underfits (86.53). With this many correlated
candidates, fitting which members to use costs more than it gains.
"""
import json
import os

import numpy as np

from slra_st import paths
from slra_st.ensembling import ab_splits, combine, load_pool, qwk_fit_eval, weight_vector
from slra_st.metrics import fast_qwk

N_SPLITS = int(os.environ.get("N_SPLITS", "4"))


def greedy(X_fit, y_fit, pool_idx, rounds=25):
    """Caruana greedy forward selection with replacement. Returns a weight vector.

    Uses naive rounding instead of fitting thresholds inside the search: at 25 rounds x pool
    size the threshold fit would take all the runtime, and only the ranking matters here.
    """
    counts = np.zeros(X_fit.shape[1])
    cur = np.zeros(len(y_fit))
    best_hist, best_counts = -1, None
    for _ in range(rounds):
        best_q, best_j = -1, None
        for j in pool_idx:
            cand = (cur * counts.sum() + X_fit[:, j]) / (counts.sum() + 1)
            q = fast_qwk(y_fit - 1, np.clip(np.rint(cand), 1, 19).astype(int) - 1)
            if q > best_q:
                best_q, best_j = q, j
        counts[best_j] += 1
        cur = combine(X_fit, counts)
        if best_q > best_hist:
            best_hist, best_counts = best_q, counts.copy()
    return best_counts


pool = load_pool("test")
tags, X, y = pool.tags, pool.X, pool.y
print(f"pool: {len(tags)} models on clean test (n={len(y)})")
ad_idx = [i for i, t in enumerate(tags) if t.endswith("_ad")]
print(f"  all-data members: {len(ad_idx)}   train-only: {len(tags) - len(ad_idx)}")

v6 = json.load(open(paths.ensemble("v6")))
ad = json.load(open(paths.ensemble("all_data")))
s_v6 = combine(X, weight_vector(tags, v6["members"], v6["weights"]))
s_ad = combine(X, weight_vector(tags, ad["members"], ad["weights"]))
s_bl = 0.5 * s_v6 + 0.5 * s_ad

results = {k: [] for k in ("V6 fixed", "AD fixed", "V6+AD blend", "greedy(all)", "greedy(ad-only)")}
sel_all, sel_ad = [], []
for i, (fit, ev) in enumerate(ab_splits(len(y), N_SPLITS)):
    results["V6 fixed"].append(qwk_fit_eval(s_v6[fit], y[fit], s_v6[ev], y[ev]))
    results["AD fixed"].append(qwk_fit_eval(s_ad[fit], y[fit], s_ad[ev], y[ev]))
    results["V6+AD blend"].append(qwk_fit_eval(s_bl[fit], y[fit], s_bl[ev], y[ev]))
    wg = greedy(X[fit], y[fit], range(len(tags)))
    sg = combine(X, wg)
    results["greedy(all)"].append(qwk_fit_eval(sg[fit], y[fit], sg[ev], y[ev]))
    sel_all.append(wg)
    wa = greedy(X[fit], y[fit], ad_idx)
    sa = combine(X, wa)
    results["greedy(ad-only)"].append(qwk_fit_eval(sa[fit], y[fit], sa[ev], y[ev]))
    sel_ad.append(wa)
    print(f"  half {i} done", flush=True)

print()
print("=" * 74)
print("HONEST holdout QWK (select+calibrate on half of test, score untouched half)")
print("=" * 74)
for k in sorted(results, key=lambda k: -np.mean(results[k])):
    v = np.array(results[k]) * 100
    print(f"  {k:18s} {v.mean():.3f} +- {v.std():.3f}")

# What greedy keeps picking across splits tells more than any single run's pick.
agg = np.sum(sel_all, axis=0)
print("\ngreedy(all) most-picked models across splits:")
for i in np.argsort(-agg)[:14]:
    if agg[i] > 0:
        fam = "AD" if tags[i].endswith("_ad") else "tr"
        print(f"   {agg[i]:5.0f}  [{fam}] {tags[i]}")

# The committed artifacts/diagnostics/combine_diag.json is the Aug 2 record, over the 49 members
# that existed then. Greedy selection is pool-dependent, so a re-run today would overwrite it with
# a different, later result. Write to the scratch directory instead and let the reader diff.
paths.REGENERATED.mkdir(parents=True, exist_ok=True)
out = paths.REGENERATED / "combine_diag.json"
json.dump({"tags": tags, "agg_all": agg.tolist(), "agg_ad": np.sum(sel_ad, axis=0).tolist()},
          open(out, "w"), indent=2)
print(f"\nwrote {out}  (committed record: {paths.DIAGNOSTICS / 'combine_diag.json'}, 49 members)")
