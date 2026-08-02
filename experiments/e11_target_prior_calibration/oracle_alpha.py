"""Is there any headroom left in shift adaptation, and which way should blind be rescaled?

Q1 ORACLE. Under a simulated concentrated shift, how far is 'naive' (thresholds from the
   unshifted calibration split) from an oracle that fits thresholds on the shifted labels? If
   the oracle is close, dispersion adaptation is dead. If it is far, my adaptations were just
   bad estimators and it is worth another try.

Q2 BLIND ALPHA. Estimate blind's label variance with BBSE (prior_adapt.py showed the estimate
   tracks the truth), using the test confusion matrix and the submitted blind prediction
   distribution, then compare it to the variance of the blind predictions. QWK is maximized at
   Var_pred == Var_true, so alpha* = sqrt(Vt_hat/Vp). If alpha* is far from 1 the saved integer
   predictions can just be rescaled, with no GPU and no re-inference.

Measured alpha* is about 1.00, so we are already at the dispersion optimum and what is left is
correlation, not calibration.
"""
import numpy as np
import pandas as pd

from slra_st import paths
from slra_st.calibration import QWKThresholdOptimizer, labels_from_thresholds
from slra_st.ensembling import ensemble_scores
from slra_st.metrics import fast_qwk
from slra_st.shift import bbse, prior_mean_var, prior_of, resample_indices, tilt_prior

S, y = ensemble_scores("blend_v7", "test")
N = len(y)
p_test = prior_of(y)

print("=" * 84)
print("Q1  ORACLE HEADROOM under simulated concentrated shift (QWK x100)")
print("=" * 84)
print(f"{'k':>6} {'naive':>8} {'ORACLE':>8} {'gap':>7} {'alpha-int':>10} {'true_var':>9}")
for k in (-1.5, -1.0, -0.5, 0.0):
    nv, orc, ai, tv = [], [], [], []
    for rep in range(5):
        rng = np.random.RandomState(rep)
        perm = rng.permutation(N)
        A, B = perm[: N // 2], perm[N // 2:]
        th_A = QWKThresholdOptimizer().fit(S[A], y[A]).thresholds_
        idx = resample_indices(y[B], tilt_prior(p_test, k), rng, len(B))
        Sb, yb = S[B][idx], y[B][idx]
        pred_naive = labels_from_thresholds(Sb, th_A)
        nv.append(fast_qwk(yb - 1, pred_naive - 1))
        th_o = QWKThresholdOptimizer().fit(Sb, yb).thresholds_  # oracle: sees the shifted labels
        orc.append(fast_qwk(yb - 1, labels_from_thresholds(Sb, th_o) - 1))
        # Cheapest possible adaptation: rescale the integer predictions about their mean.
        _, vt = prior_mean_var(prior_of(yb))
        mu = pred_naive.mean()
        a = np.sqrt(max(vt, 1e-6) / max(pred_naive.var(), 1e-6))
        pa = np.clip(np.rint(mu + a * (pred_naive - mu)), 1, 19).astype(int)
        ai.append(fast_qwk(yb - 1, pa - 1))
        tv.append(vt)
    print(f"{k:+6.2f} {np.mean(nv)*100:8.3f} {np.mean(orc)*100:8.3f} "
          f"{(np.mean(orc)-np.mean(nv))*100:+7.3f} {np.mean(ai)*100:10.3f} {np.mean(tv):9.2f}")

print()
print("=" * 84)
print("Q2  BLIND: estimated true variance vs the predicted variance")
print("=" * 84)
th_full = QWKThresholdOptimizer().fit(S, y).thresholds_
pred_test = labels_from_thresholds(S, th_full)
_, vt_test = prior_mean_var(prior_of(y))
_, vp_test = prior_mean_var(prior_of(pred_test))
print(f"  TEST : Var_true={vt_test:.3f}  Var_pred={vp_test:.3f}  ratio={vp_test/vt_test:.3f}")

pb = pd.read_csv(paths.SUBMITTED / "prediction_blend")["Prediction"].astype(int).values
p_hat = bbse(y, pred_test, prior_of(pb))
mt_hat, vt_hat = prior_mean_var(p_hat)
vp_blind, mu_blind = float(pb.var()), float(pb.mean())
print(f"  BLIND: Var_pred={vp_blind:.3f} (mean {mu_blind:.2f})")
print(f"         BBSE-estimated Var_true={vt_hat:.3f} (mean {mt_hat:.2f})")
alpha = np.sqrt(max(vt_hat, 1e-6) / max(vp_blind, 1e-6))
print(f"         => alpha* = sqrt(Vt_hat/Vp) = {alpha:.4f}   "
      f"({'COMPRESS' if alpha < 1 else 'EXPAND'} by {abs(1-alpha)*100:.1f}%)")

# With MSE fixed, dQWK/QWK ~ dD/D where D = Vt + Vp + dmu^2. A ratio below 1 means the
# denominator shrinks at alpha*, so compressing only pays if it cuts MSE by more than D.
# That is what the alpha-int column in Q1 measures.
print("\n  implied QWK sensitivity:")
D_now, D_opt = vt_hat + vp_blind, 2 * vt_hat
print(f"    D_now={D_now:.2f}  D_at_alpha*={D_opt:.2f}  ratio={D_opt/D_now:.4f}")
