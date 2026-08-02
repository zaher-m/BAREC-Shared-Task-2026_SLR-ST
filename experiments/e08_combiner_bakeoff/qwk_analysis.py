"""The QWK identity, and whether the predictions are under-dispersed.

With quadratic weights:

    QWK = 1 - MSE / (Var_true + Var_pred + (mu_true - mu_pred)^2)

so QWK likes low MSE and large predicted variance. Ensemble averaging shrinks predictions
toward the mean, which suggests we could be losing QWK to under-dispersion. For a linear
rescale p' = mu_t + a(p - mu_p) the best a is sqrt(Var_t/Var_p), where QWK = corr(t, p).

Three parts: A/B comparison of the two family ensembles, the variance decomposition of each,
and a sweep over dispersion factors.

Result for the lever: no gain, the measured a* is about 1.00, so the system already sits at
the dispersion optimum and what is left is correlation. The identity itself stayed useful: it
is what later explained the test-to-blind drop as denominator shrinkage.
"""
import numpy as np

from slra_st.calibration import QWKThresholdOptimizer, labels_from_thresholds
from slra_st.ensembling import ab_splits, ensemble_scores, honest_ab
from slra_st.metrics import fast_qwk


def decompose(y, pred):
    t, p = y.astype(float), pred.astype(float)
    mse = np.mean((t - p) ** 2)
    den = t.var() + p.var() + (t.mean() - p.mean()) ** 2
    return mse, t.var(), p.var(), t.mean() - p.mean(), 1 - mse / den


print("=" * 72)
print("FAIR COMPARISON (thresholds fit on half of test, scored on other half)")
print("=" * 72)
res = {}
for name, config in (("V6 (train-only)", "v6"), ("ALL-DATA (_ad)", "all_data")):
    s, y = ensemble_scores(config, "test")
    m, sd = honest_ab(s, y)
    res[name] = (s, y)
    print(f"  {name:18s} holdout QWK = {m:.3f} +- {sd:.3f}")

print()
print("=" * 72)
print("QWK STRUCTURE: is the prediction UNDER-DISPERSED? (Var_pred vs Var_true)")
print("=" * 72)
for name, (s, y) in res.items():
    th = QWKThresholdOptimizer().fit(s, y).thresholds_
    pred = labels_from_thresholds(s, th)
    mse, vt, vp, dmu, q = decompose(y, pred)
    print(f"  {name}")
    print(f"     MSE={mse:.4f}  Var_true={vt:.3f}  Var_pred={vp:.3f}  "
          f"ratio Vp/Vt={vp/vt:.3f}  d_mu={dmu:+.3f}  QWK={q*100:.3f}")
    print(f"     optimal linear a* = sqrt(Vt/Vp) = {np.sqrt(vt/vp):.3f}   "
          f"(a* > 1 would mean predictions are too compressed)")

print()
print("=" * 72)
print("DISPERSION-EXPANDED CALIBRATION: scale scores about their mean, then fit thresholds")
print("=" * 72)
print("  (alpha and thresholds both fit on half of test; scored on the held-out half)")
for name, (s, y) in res.items():
    by_alpha = {}
    for fit, ev in ab_splits(len(y), reps=6, seed_offset=100):
        mu = s[fit].mean()
        for a in (1.0, 1.05, 1.10, 1.15, 1.20, 1.30, 1.40):
            sf, se = mu + a * (s[fit] - mu), mu + a * (s[ev] - mu)
            th = QWKThresholdOptimizer().fit(sf, y[fit]).thresholds_
            by_alpha.setdefault(a, []).append(
                fast_qwk(y[ev] - 1, labels_from_thresholds(se, th) - 1))
    print(f"  {name}")
    for a, v in sorted(by_alpha.items()):
        mark = "  <-- baseline" if a == 1.0 else ""
        print(f"     alpha={a:.2f}  holdout QWK = {np.mean(v)*100:.3f}{mark}")
