"""Exact importance-weighted threshold fitting against the estimated blind prior.

Anchor: calibrating to the BBSE blind prior scored blind 85.4 (vs 85.3 for raw-test
calibration) and the proxy predicted it almost exactly (85.444 -> 85.4), so the proxy is
worth optimizing against directly.

Two changes over the submitted run:
  1. Exact weights w_i = p_hat[y_i]/p_src[y_i] inside a weighted QWK, instead of drawing a
     resample. Deterministic, and it uses all 7286 points (the earlier run had to pick among
     six noisy resample fits).
  2. A refined prior: re-run BBSE with the new best submission as the anchor, i.e. one
     EM-style iteration.

Change 1 is fine. Change 2 is not, and that is the point of keeping this file: the refined
prior's variance drifted to 9.31 (from 8.99) and its submission regressed blind 85.4 -> 85.2.
Iterating BBSE on the system's own output feeds its bias back in. One pass only. The two
Acc-max variants written here were never submitted.
"""
import json

import numpy as np
import pandas as pd

from slra_st import paths, submission
from slra_st.calibration import QWKThresholdOptimizer, labels_from_thresholds, optimize_thresholds
from slra_st.ensembling import ensemble_scores, load_blind
from slra_st.metrics import weighted_accuracy, weighted_qwk
from slra_st.shift import bbse, importance_weights, prior_mean_var, prior_of, resample_indices

# Regenerated candidates go to a scratch directory so re-running never overwrites the
# committed record in submissions/.
OUT_DIR = paths.REGENERATED
OUT_DIR.mkdir(parents=True, exist_ok=True)

S, y = ensemble_scores("final_v8", "test")
N = len(y)
bids, _, _, bs = load_blind("prediction_final")

best_pred = pd.read_csv(paths.SUBMITTED / "pred_priorQWK_acc")["Prediction"].astype(int).values

# Rebuild submission 10's thresholds first, so the test confusion matrix matches
# the blind distribution we feed to BBSE.
src0 = prior_of(y)
fr = json.load(open(paths.THRESHOLDS / "acc_frontier.json"))
th040 = np.asarray(fr["0.004"], float)
p0 = bbse(y, labels_from_thresholds(S, th040), prior_of(labels_from_thresholds(bs, th040)))
idx0 = resample_indices(y, p0, np.random.RandomState(0), 20000)
th_best = QWKThresholdOptimizer().fit(S[idx0], y[idx0]).thresholds_

p_hat = bbse(y, labels_from_thresholds(S, th_best), prior_of(best_pred))
m1, v1 = prior_mean_var(p_hat)
print(f"refined blind prior: mean={m1:.2f} var={v1:.2f}   (previous est var 8.99, test 10.63)")

sw = importance_weights(y, p_hat, src0)
sw = sw / sw.mean()
ess = sw.sum() ** 2 / (sw ** 2).sum()
print(f"importance weights: min={sw.min():.3f} max={sw.max():.3f} ESS={ess:.0f}/{N}")

print()
print("=" * 80)
print("HONEST weighted A/B (fit on half of test, score untouched half, same weights)")
print("=" * 80)
raw_th_full = QWKThresholdOptimizer().fit(S, y).thresholds_
names = ("raw-test (V8 baseline)", "weighted-QWK max", "weighted-Acc @floor0.2", "weighted-Acc @floor0.5")
cands = {n: [] for n in names}
# Only one direction per permutation here: each split costs four weighted threshold fits,
# which is most of the runtime.
for i in range(4):
    perm = np.random.RandomState(i).permutation(N)
    A, B = perm[: N // 2], perm[N // 2:]
    tA = QWKThresholdOptimizer().fit(S[A], y[A]).thresholds_
    sets = {"raw-test (V8 baseline)": tA,
            "weighted-QWK max": optimize_thresholds(S[A], y[A], "qwk", init=tA, weights=sw[A],
                                                    rounds=8, grid=0.03, tol=1e-10)}
    qmax = weighted_qwk(y[A], labels_from_thresholds(S[A], sets["weighted-QWK max"]), sw[A])
    for label, floor in (("weighted-Acc @floor0.2", 0.002), ("weighted-Acc @floor0.5", 0.005)):
        sets[label] = optimize_thresholds(S[A], y[A], "acc", init=sets["weighted-QWK max"],
                                          qwk_floor=qmax - floor, weights=sw[A],
                                          rounds=8, grid=0.03, tol=1e-10)
    for k, t in sets.items():
        p = labels_from_thresholds(S[B], t)
        cands[k].append((weighted_qwk(y[B], p, sw[B]), weighted_accuracy(y[B], p, sw[B])))
    print(f"  split {i} done", flush=True)

print()
print(f"{'scheme':>26} {'wQWK':>9} {'wAcc':>8}")
for k, v in cands.items():
    v = np.array(v)
    print(f"{k:>26} {v[:,0].mean()*100:9.3f} {v[:,1].mean()*100:8.2f}")
print("\nANCHOR: blind-prior calibration scored blind 85.4 / 38.7 (proxy predicted 85.444)")

print()
print("=" * 80)
print(f"FINAL fits on all {N} test points (deterministic, exact weights)")
print("=" * 80)
th_q = optimize_thresholds(S, y, "qwk", init=raw_th_full, weights=sw, rounds=8, grid=0.03, tol=1e-10)
qmax = weighted_qwk(y, labels_from_thresholds(S, th_q), sw)
th_a2 = optimize_thresholds(S, y, "acc", init=th_q, qwk_floor=qmax - 0.002, weights=sw,
                            rounds=8, grid=0.03, tol=1e-10)
th_a5 = optimize_thresholds(S, y, "acc", init=th_q, qwk_floor=qmax - 0.005, weights=sw,
                            rounds=8, grid=0.03, tol=1e-10)
for th, name in ((th_q, "W1_wqwk"), (th_a2, "W2_wacc_tight"), (th_a5, "W3_wacc_loose")):
    p = labels_from_thresholds(S, th)
    q, a = weighted_qwk(y, p, sw) * 100, weighted_accuracy(y, p, sw) * 100
    pb = labels_from_thresholds(bs, th).astype(int)
    dst = submission.write(OUT_DIR / f"pred_{name}", bids, pb)
    ok = submission.validate(dst, reference_ids=bids, previous=paths.SUBMITTED / "pred_priorQWK_acc")
    print(f"  {name}: wQWK={q:.3f} wAcc={a:.2f} | blind var={pb.var():.2f} "
          f"diff-vs-submitted={ok.pop('changed')*100:.1f}%  VALID={all(ok.values())}")
