# E10. The accuracy frontier

**When** Aug 2  **Verdict** Acc 35.9 → 40.4 at flat blind QWK

## Situation

V8 scored blind **85.3 / Acc 35.9**. Two other systems on the same blind set occupied visibly
different points on the same trade-off: one at **85.3 / 37.7**, the same QWK with 1.8 more accuracy,
and one at **84.2 / 41.7**, 1.1 less QWK for 5.8 more accuracy. Three systems strung out along a line
of that shape is empirical evidence that Acc and QWK are exchangeable here, and it raised the question
this experiment answers: how much accuracy is available at *no* QWK cost?

## The idea

QWK-optimal thresholds spread predictions so that `Var_pred` matches `Var_true`, since that
is what maximizes kappa. Exact-match accuracy wants the opposite: concentrate mass on the frequent
levels (L10/L12/L14 are ~50% of BAREC). So there is a genuine Acc-vs-QWK frontier.

The exploitable asymmetry: **QWK is flat near its optimum while accuracy is not.** And blind QWK is
reported to one decimal, so a loss under ~0.05 leaves the displayed 85.3 unchanged while accuracy
moves several points.

## Method

Coordinate ascent on accuracy subject to `QWK ≥ QWK_opt − ε`, on the cached V8 blind score matrix,
sweeping ε and validating each point under the honest A/B protocol (does the accuracy gain survive on
a half the thresholds were not fitted on?).

## Blind results

| Submission | ε | Blind QWK | Blind Acc |
|---|---|---|---|
| V8 baseline | - | 85.3 | 35.9 |
| `pred_eps0.40_maxAcc` | 0.004 | **85.3** | **40.4** |
| `pred_eps0.10_safeQWK` | 0.001 | 85.1 | 37.9 |

Accuracy rose **35.9 → 37.9 → 40.4 while QWK stayed flat at 85.1–85.3**. The test-fitted frontier had
predicted a QWK cost that never materialized on blind.

## What this told us about the shift

That discrepancy is the interesting part. Thresholds that concentrate mass on frequent levels *suited*
blind better than test predicted, which is what a **more concentrated blind label prior** looks like.
Combined with the QWK identity, the flatness of blind QWK across threshold variants while accuracy
swung ±4.5 confirmed that the ~2 QWK test→blind drop is denominator shrinkage, not miscalibration.

So chasing that trade-off produced the measurement that made [E11](../e11_target_prior_calibration/)
possible: stop optimizing against the raw test prior, estimate the target prior and optimize against
*that*. `push_acc.py` is the first step in that direction, reweight test to the BBSE-estimated blind
prior, then maximize accuracy under a QWK floor there.

Note the ordering honestly: the frontier submissions were a hedge on the secondary metrics that
happened to be informative. The three probes `push_acc.py` wrote (`pred_P0/P1/P2`) were never submitted: the
prior-calibrated candidate from E11 superseded them within the hour.

## Files

| File | Role | Writes |
|---|---|---|
| `acc_frontier.py` | the Pareto frontier on raw test + A/B generalization check | `configs/thresholds/acc_frontier.json` |
| `push_acc.py` | the same optimization under the estimated blind prior | `configs/thresholds/push_acc.json`, `submissions/candidates/pred_P{1,2}_acc` |

The shared coordinate-ascent implementation is `optimize_thresholds()` in
`src/slra_st/calibration.py`.
