"""Can fitted ensemble weights beat the fixed 50/50 blend?

Between greedy (overfits, 86.60) and uniform (underfits, 86.53): fit non-negative weights on
the simplex with a ridge pull toward a prior, keeping every member but shrinking the weights.
Priors tried: uniform, and the 50/50 blend itself.

Result: no, at any lambda. The member scores are too correlated, the least-squares problem is
ill-conditioned, and the fit spends its freedom on noise.
"""
import json

import numpy as np

from slra_st import paths
from slra_st.calibration import QWKThresholdOptimizer, labels_from_thresholds
from slra_st.ensembling import ab_splits, combine, load_pool, weight_vector
from slra_st.metrics import fast_qwk


def nnls_simplex(X, y, lam, w0=None, iters=400):
    """Projected-gradient fit of ||X w - y||^2 + lam*||w - w0||^2 over the simplex."""
    M = X.shape[1]
    w0 = np.full(M, 1.0 / M) if w0 is None else w0
    w = w0.copy()
    A = X.T @ X / X.shape[0]
    b = X.T @ y / X.shape[0]
    L = np.linalg.eigvalsh(A).max() + lam
    for _ in range(iters):
        w = w - (A @ w - b + lam * (w - w0)) / L
        w = np.clip(w, 0, None)
        s = w.sum()
        w = w / s if s > 0 else np.full(M, 1.0 / M)
    return w


pool = load_pool("test")
tags, X, y = pool.tags, pool.X, pool.y
N, M = X.shape
v6 = json.load(open(paths.ensemble("v6")))
ad = json.load(open(paths.ensemble("all_data")))
w_v6 = weight_vector(tags, v6["members"], v6["weights"])
w_ad = weight_vector(tags, ad["members"], ad["weights"])
w_v6 /= w_v6.sum()
w_ad /= w_ad.sum()
w_blend = 0.5 * w_v6 + 0.5 * w_ad
s_blend = combine(X, w_blend)
print(f"pool M={M}, n_test={N}")

lams = [0.001, 0.01, 0.05, 0.2, 1.0]
res = {"blend 0.5/0.5 (current)": [], "ridge lam=0.05 (blend prior)": [],
       **{f"ridge-simplex lam={l}": [] for l in lams}}
for i, (fit, ev) in enumerate(ab_splits(N, reps=6)):
    th = QWKThresholdOptimizer().fit(s_blend[fit], y[fit]).thresholds_
    res["blend 0.5/0.5 (current)"].append(
        fast_qwk(y[ev] - 1, labels_from_thresholds(s_blend[ev], th) - 1))
    for l in lams:
        w = nnls_simplex(X[fit], y[fit].astype(float), l)
        s = combine(X, w)
        th = QWKThresholdOptimizer().fit(s[fit], y[fit]).thresholds_
        res[f"ridge-simplex lam={l}"].append(
            fast_qwk(y[ev] - 1, labels_from_thresholds(s[ev], th) - 1))
    w = nnls_simplex(X[fit], y[fit].astype(float), 0.05, w0=w_blend)
    s = combine(X, w)
    th = QWKThresholdOptimizer().fit(s[fit], y[fit]).thresholds_
    res["ridge lam=0.05 (blend prior)"].append(
        fast_qwk(y[ev] - 1, labels_from_thresholds(s[ev], th) - 1))
    print(f"  half {i} done", flush=True)

print()
print("=" * 70)
print("HONEST holdout QWK -- regularized weight estimators")
print("=" * 70)
for k in sorted(res, key=lambda k: -np.mean(res[k])):
    v = np.array(res[k]) * 100
    mark = "   <-- current" if k.startswith("blend") else ""
    print(f"  {k:32s} {v.mean():.3f} +- {v.std():.3f}{mark}")
