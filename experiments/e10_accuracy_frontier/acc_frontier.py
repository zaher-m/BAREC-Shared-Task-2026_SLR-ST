"""How much exact accuracy can be bought without losing QWK?

Situation: blind QWK 85.3 at Acc 35.9, while another system reached the same QWK at Acc 37.7
(35.9 vs 37.7), with another team at Acc 41.7 / QWK 84.2.

QWK-optimal thresholds spread the predictions so Var_pred matches Var_true. Exact accuracy
wants the opposite, mass on the frequent levels (L10/L12/L14 are ~50% of BAREC). So there is
a real frontier. The useful part: QWK is flat near its optimum and accuracy is not, and blind
QWK is reported to one decimal, so a loss under ~0.05 does not even show up.

Method: maximize Acc subject to QWK >= QWK_opt - eps, sweep eps, then check on held-out
halves whether the accuracy gain survives. Threshold sets are saved per eps.
"""
import json

import numpy as np

from slra_st import paths
from slra_st.calibration import QWKThresholdOptimizer, labels_from_thresholds, optimize_thresholds
from slra_st.ensembling import ab_splits, ensemble_scores
from slra_st.metrics import fast_qwk

S, y = ensemble_scores("final_v8", "test")
N = len(y)


def acc_of(th, s, yy):
    return float((labels_from_thresholds(s, th) == yy).mean())


def qwk_of(th, s, yy):
    return fast_qwk(yy - 1, labels_from_thresholds(s, th) - 1)


base_th = QWKThresholdOptimizer().fit(S, y).thresholds_
q0, a0 = qwk_of(base_th, S, y), acc_of(base_th, S, y)
print(f"baseline (QWK-optimal) on full test: QWK={q0*100:.3f}  Acc={a0*100:.2f}")
print(f"blind for the same config:           QWK=85.3    Acc=35.9   (another system: 85.3 / 37.7)")
print()
print("=" * 78)
print("PARETO FRONTIER on full test: max Acc subject to QWK >= QWK_opt - eps")
print("=" * 78)
print(f"{'eps':>6} {'QWK':>9} {'dQWK':>7} {'Acc':>8} {'dAcc':>7}")
front = {}
for eps in (0.0, 0.0005, 0.001, 0.002, 0.004, 0.008):
    th = optimize_thresholds(S, y, "acc", init=base_th, qwk_floor=q0 - eps)
    q, a = qwk_of(th, S, y), acc_of(th, S, y)
    front[eps] = th
    print(f"{eps*100:6.2f} {q*100:9.3f} {(q-q0)*100:+7.3f} {a*100:8.2f} {(a-a0)*100:+7.2f}")

print()
print("=" * 78)
print("HONEST A/B: does the Acc gain GENERALISE? (optimise on half, score other half)")
print("=" * 78)
print(f"{'eps':>6} {'heldout QWK':>12} {'dQWK':>7} {'heldout Acc':>12} {'dAcc':>7}")
for eps in (0.0, 0.001, 0.002, 0.004, 0.008):
    qs, accs, q0s, a0s = [], [], [], []
    for fit, ev in ab_splits(N, reps=4):
        bt = QWKThresholdOptimizer().fit(S[fit], y[fit]).thresholds_
        th = optimize_thresholds(S[fit], y[fit], "acc", init=bt,
                                 qwk_floor=qwk_of(bt, S[fit], y[fit]) - eps)
        qs.append(qwk_of(th, S[ev], y[ev]))
        accs.append(acc_of(th, S[ev], y[ev]))
        q0s.append(qwk_of(bt, S[ev], y[ev]))
        a0s.append(acc_of(bt, S[ev], y[ev]))
    print(f"{eps*100:6.2f} {np.mean(qs)*100:12.3f} {(np.mean(qs)-np.mean(q0s))*100:+7.3f}"
          f" {np.mean(accs)*100:12.2f} {(np.mean(accs)-np.mean(a0s))*100:+7.2f}")

out = paths.THRESHOLDS / "acc_frontier.json"
json.dump({str(k): v.tolist() for k, v in front.items()}, open(out, "w"), indent=2)
print(f"\nwrote {out}")
