"""Check prior-adaptive calibration by simulation before using it on blind.

The blend scores better than test on every coarse metric (Acc3/5/7) but loses 2.2 QWK. Via
QWK = 1 - MSE/(Var_true + Var_pred + dmu^2) that is what a more concentrated label
distribution looks like: smaller Var_true, smaller denominator. If that is right, thresholds
tuned on test are mis-dispersed for blind.

Blind labels are not available, so test stands in for it:
  - split test into A (calibrate) / B (evaluate)
  - resample B to a shifted prior
  - estimate that prior from predictions only, with BBSE (A's confusion matrix + B's
    predicted distribution)
  - compare A's plain thresholds against two adaptations:
      (A) alpha-rescale so Var_pred matches the estimated Var_true (1 parameter)
      (B) reweight A to the estimated prior and refit thresholds (19-dim, stronger but noisier)

Only an adaptation that helps at the concentrated (k<0) shifts is worth using on blind. (B)
was applied and produced submission 10. The est_var column also shows BBSE tracks the true
variance.
"""
import numpy as np

from slra_st.calibration import QWKThresholdOptimizer, labels_from_thresholds
from slra_st.ensembling import ensemble_scores
from slra_st.metrics import fast_qwk
from slra_st.shift import bbse, prior_mean_var, prior_of, resample_indices, tilt_prior

S, y = ensemble_scores("blend_v7", "test")
N = len(y)
print(f"blend on test: n={N}, Var_true={y.var():.3f}")

p_test = prior_of(y)
mean_test, var_test = prior_mean_var(p_test)
print(f"test prior: mean={mean_test:.2f} var={var_test:.2f}")
print("\nsimulated target priors (k<0 = concentrated, blind-like):")
for k in (-1.5, -0.75, 0.0, 0.75):
    m, v = prior_mean_var(tilt_prior(p_test, k))
    print(f"   k={k:+.2f}: var={v:.2f} mean={m:.2f}")

rows = []
for rep in range(5):
    rng = np.random.RandomState(rep)
    perm = rng.permutation(N)
    A, B = perm[: N // 2], perm[N // 2:]
    th_A = QWKThresholdOptimizer().fit(S[A], y[A]).thresholds_
    pred_A = labels_from_thresholds(S[A], th_A)
    for k in (-1.5, -0.75, 0.0, 0.75):
        tgt_p = tilt_prior(p_test, k)
        idx = resample_indices(y[B], tgt_p, rng, len(B))
        Sb, yb = S[B][idx], y[B][idx]
        q_naive = fast_qwk(yb - 1, labels_from_thresholds(Sb, th_A) - 1)

        # Target prior comes from PREDICTIONS ONLY, yb is never touched here.
        q_pred = prior_of(labels_from_thresholds(Sb, th_A))
        p_hat = bbse(y[A], pred_A, q_pred)
        _, v_hat = prior_mean_var(p_hat)
        _, v_true = prior_mean_var(prior_of(yb))

        base = labels_from_thresholds(Sb, th_A).astype(float)
        a = np.sqrt(max(v_hat, 1e-6) / max(base.var(), 1e-6))
        mu = Sb.mean()
        q_alpha = fast_qwk(yb - 1, labels_from_thresholds(mu + a * (Sb - mu), th_A) - 1)

        ridx = resample_indices(y[A], p_hat, rng, len(A))
        th_rw = QWKThresholdOptimizer().fit(S[A][ridx], y[A][ridx]).thresholds_
        q_rw = fast_qwk(yb - 1, labels_from_thresholds(Sb, th_rw) - 1)
        rows.append((k, q_naive, q_alpha, q_rw, v_true, v_hat, a))

rows = np.array(rows)
print()
print("=" * 88)
print("SIMULATED LABEL SHIFT on test  (QWK x100; est_var vs true_var checks BBSE quality)")
print("=" * 88)
print(f"{'k':>6} {'naive':>8} {'alpha-adapt':>12} {'reweight-th':>12} {'true_var':>9} {'est_var':>8} {'alpha':>6}")
for k in (-1.5, -0.75, 0.0, 0.75):
    m = rows[rows[:, 0] == k]
    print(f"{k:+6.2f} {m[:,1].mean()*100:8.3f} {m[:,2].mean()*100:12.3f} {m[:,3].mean()*100:12.3f}"
          f" {m[:,4].mean():9.2f} {m[:,5].mean():8.2f} {m[:,6].mean():6.3f}")
print()
print("An adaptation is only worth applying to blind if it beats 'naive' at the concentrated")
print("(k<0) shifts, which is what blind looks like.")
