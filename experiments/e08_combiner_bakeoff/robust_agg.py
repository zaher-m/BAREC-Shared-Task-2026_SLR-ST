"""Does a median or trimmed mean beat the weighted mean?

Idea: this system scored higher on Acc, Acc+-1, Acc5 and Acc3 but lower on QWK, so the loss should be
concentrated in a few very wrong predictions. A mean gets dragged by one member misreading a
sentence; a median does not. Also tried shrinking high-disagreement items toward the mean.

Members are replicated by their integer weights first so median/trim respect the weighting.

Result: no. Every robust variant is at or below the weighted mean. The tail diagnostic at the
bottom says why: the big errors are all members wrong together, not one rogue member, so
there is no outlier for a median to throw away.
"""
import json

import numpy as np

from slra_st import paths
from slra_st.calibration import QWKThresholdOptimizer, labels_from_thresholds
from slra_st.ensembling import ab_splits, combine, load_members
from slra_st.metrics import fast_qwk

ens = json.load(open(paths.ensemble("blend_v7")))
weights = np.asarray(ens["weights"], float)
sm = load_members(ens["members"], "test")
X, y = sm.X, sm.y
N, M = X.shape
print(f"blend members M={M}, n_test={N}")

rep = np.repeat(np.arange(M), weights.astype(int))
Xr = X[:, rep]
print(f"weight-replicated members: {Xr.shape[1]}")

base = combine(X, weights)
schemes = {"weighted mean (current)": base, "median (w-replicated)": np.median(Xr, axis=1)}
for tr in (0.10, 0.20, 0.30):
    k = max(1, int(round(tr * Xr.shape[1] / 2)))
    schemes[f"trimmed mean {int(tr*100)}%"] = np.sort(Xr, axis=1)[:, k:Xr.shape[1] - k].mean(axis=1)

mu_all = base.mean()
disp = Xr.std(axis=1)
dz = (disp - disp.mean()) / (disp.std() + 1e-9)
for lam in (0.05, 0.10, 0.20):
    f = np.clip(dz, 0, None)  # only shrink items where members disagree more than average
    f = f / (f.max() + 1e-9)
    schemes[f"disagreement shrink lam={lam:.2f}"] = base + lam * f * (mu_all - base)
schemes["0.5*mean + 0.5*median"] = 0.5 * base + 0.5 * np.median(Xr, axis=1)

res = {k: [] for k in schemes}
for fit, ev in ab_splits(N, reps=6):
    for k, s in schemes.items():
        th = QWKThresholdOptimizer().fit(s[fit], y[fit]).thresholds_
        res[k].append(fast_qwk(y[ev] - 1, labels_from_thresholds(s[ev], th) - 1))

print()
print("=" * 74)
print("HONEST holdout QWK -- robust aggregation of the blend")
print("=" * 74)
for k in sorted(res, key=lambda k: -np.mean(res[k])):
    v = np.array(res[k]) * 100
    mark = "   <-- current" if k.startswith("weighted mean") else ""
    print(f"  {k:34s} {v.mean():.3f} +- {v.std():.3f}{mark}")

print("\ntail diagnostic on full test (thresholds fit on full test):")
for k in ("weighted mean (current)", "median (w-replicated)", "0.5*mean + 0.5*median"):
    s = schemes[k]
    th = QWKThresholdOptimizer().fit(s, y).thresholds_
    p = labels_from_thresholds(s, th)
    e = np.abs(p - y)
    print(f"  {k:34s} |err|>=3: {(e>=3).mean()*100:5.2f}%   "
          f"|err|>=5: {(e>=5).mean()*100:4.2f}%   MSE={np.mean((p-y)**2):.4f}")
