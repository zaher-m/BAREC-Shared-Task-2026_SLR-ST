"""Push accuracy under the estimated blind prior instead of the raw test prior.

What blind told us over three submissions: QWK stayed flat at 85.1-85.3 while Acc went
35.9 -> 37.9 -> 40.4. The test-fitted frontier predicted a QWK cost that never happened. The
reason is the shift: blind labels are more concentrated (BBSE Var_true 9.09 vs test 10.63),
so thresholds that put mass on the frequent levels actually suit blind.

So stop optimizing against raw test. Reweight test to the estimated blind prior first, then
maximize Acc subject to a QWK floor under that prior. This is the step that led into the
target-prior calibration in e11.

At the time a competitor sat at Acc 41.7 / QWK 84.2, so the target was Acc > 42 at QWK 85.3.
"""
import json

import numpy as np
import pandas as pd

from slra_st import paths, submission
from slra_st.calibration import QWKThresholdOptimizer, labels_from_thresholds, optimize_thresholds
from slra_st.ensembling import ensemble_scores, load_blind
from slra_st.metrics import fast_qwk
from slra_st.shift import bbse, prior_mean_var, prior_of, resample_indices

# Regenerated candidates go to a scratch directory so re-running never overwrites the
# committed record in submissions/.
OUT_DIR = paths.REGENERATED
OUT_DIR.mkdir(parents=True, exist_ok=True)

S, y = ensemble_scores("final_v8", "test")
N = len(y)
bids, _, _, bs = load_blind("prediction_final")

fr = json.load(open(paths.THRESHOLDS / "acc_frontier.json"))
th_040 = np.asarray(fr["0.004"], float)  # the eps=0.004 frontier point already submitted (85.3 / 40.4)

# Both sides of BBSE need the SAME thresholds, otherwise the confusion matrix and the blind
# predicted distribution come from two different systems.
pred_test_040 = labels_from_thresholds(S, th_040)
q_blind_040 = prior_of(labels_from_thresholds(bs, th_040))
p_hat = bbse(y, pred_test_040, q_blind_040)
m_hat, v_hat = prior_mean_var(p_hat)
print(f"BBSE blind prior: mean={m_hat:.2f} var={v_hat:.2f}   (test var={y.var():.2f})")

rng = np.random.RandomState(0)
idx = resample_indices(y, p_hat, rng, 20000)
Sb, yb = S[idx], y[idx]
print(f"reweighted test: n={len(yb)} var_true={yb.var():.2f}")


def acc_of(th, s, yy):
    return float((labels_from_thresholds(s, th) == yy).mean())


def qwk_of(th, s, yy):
    return fast_qwk(yy - 1, labels_from_thresholds(s, th) - 1)


base_th = QWKThresholdOptimizer().fit(Sb, yb).thresholds_
q_ref = qwk_of(base_th, Sb, yb)
print(f"\nunder blind-like prior: QWK-optimal QWK={q_ref*100:.3f} Acc={acc_of(base_th,Sb,yb)*100:.2f}")
print(f"submitted eps0.40 there: QWK={qwk_of(th_040,Sb,yb)*100:.3f} Acc={acc_of(th_040,Sb,yb)*100:.2f}")

print()
print("=" * 86)
print("Acc-max under BLIND-LIKE prior, at several QWK floors (blind Acc/QWK should track these)")
print("=" * 86)
print(f"{'floor':>8} {'QWK':>9} {'dQWK':>7} {'Acc':>8} {'blind var_pred':>15}")
out = {}
for eps in (0.002, 0.004, 0.008, 0.015, 0.025):
    th = optimize_thresholds(Sb, yb, "acc", init=base_th, qwk_floor=q_ref - eps)
    q, a = qwk_of(th, Sb, yb), acc_of(th, Sb, yb)
    pb = labels_from_thresholds(bs, th)
    out[eps] = (th, q, a, pb)
    print(f"{eps*100:8.2f} {q*100:9.3f} {(q-q_ref)*100:+7.3f} {a*100:8.2f} {pb.var():15.2f}")

cur = pd.read_csv(paths.SUBMITTED / "prediction_final")["Prediction"].values
for eps, name in ((0.008, "P1_acc"), (0.015, "P2_acc")):
    th, q, a, pb = out[eps]
    dst = submission.write(OUT_DIR / f"pred_{name}", bids, pb)
    print(f"\n{name}: reweighted Acc={a*100:.2f} QWK={q*100:.3f} | blind mean={pb.mean():.2f} "
          f"var={pb.var():.2f} differs from submitted on {(pb!=cur).mean()*100:.1f}% -> {dst}.zip")

th_out = paths.THRESHOLDS / "push_acc.json"
json.dump({str(k): v[0].tolist() for k, v in out.items()}, open(th_out, "w"), indent=2)
print(f"wrote {th_out}")
