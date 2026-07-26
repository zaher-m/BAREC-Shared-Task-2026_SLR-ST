"""Distribution matching vs QWK-fitted thresholds under simulated label shift.

Distribution matching needs no posteriors: put the 18 cut points at the cumulative quantiles
of a target marginal T and the predicted marginal equals T by construction. The problem is
which T to use when the target prior is unknown, so T is swept as

    T = (1 - alpha) * p_dev + alpha * p_train

and each alpha is judged by its worst-case QWK across simulated shifts of dev.

Result: matching a marginal is only as good as the marginal. Without the true target prior it
loses badly to thresholds fitted for QWK. That failure is what pushed the work toward
estimating the target prior properly (bbse in prior_adapt.py) instead of guessing it.
"""
import numpy as np

from slra_st.calibration import QWKThresholdOptimizer, labels_from_thresholds
from slra_st.ensembling import ensemble_scores
from slra_st.metrics import fast_qwk
from slra_st.shift import prior_of, train_label_prior

rng = np.random.RandomState(0)

dev_s, dev_y = ensemble_scores("v6", "dev")
p_train = train_label_prior()
p_dev = prior_of(dev_y)


def distmatch_thresholds(scores, T):
    return np.quantile(scores, np.clip(np.cumsum(T)[:-1], 0, 1))


def qwk_of(scores, y, th):
    return fast_qwk(y - 1, labels_from_thresholds(scores, th) - 1)


def tilt_mean(p, k):
    """Linear tilt: k>0 pushes mass toward the harder levels, k<0 toward the easier ones."""
    lv = np.arange(1, 20)
    q = p * np.exp(k * (lv - 10) / 9.0)
    return q / q.sum()


def resample_to(target, n=6000):
    picks = []
    for level in range(1, 20):
        pool = np.where(dev_y == level)[0]
        count = int(target[level - 1] * n)
        if len(pool) == 0 or count == 0:
            continue
        picks.append(rng.choice(pool, count, replace=True))
    idx = np.concatenate(picks)
    return dev_s[idx], dev_y[idx]


th_qwk = QWKThresholdOptimizer().fit(dev_s, dev_y).thresholds_
print(f"unshifted dev: QWK-thresh={qwk_of(dev_s,dev_y,th_qwk)*100:.3f}  "
      f"distmatch(p_dev)={qwk_of(dev_s,dev_y,distmatch_thresholds(dev_s,p_dev))*100:.3f}  "
      f"distmatch(p_train)={qwk_of(dev_s,dev_y,distmatch_thresholds(dev_s,p_train))*100:.3f}")

shifts = {f"tilt{k:+.1f}": tilt_mean(p_dev, k) for k in (-1.5, -0.75, 0, 0.75, 1.5)}

print("\nmethod                         mean_QWK  worstcase_QWK  (across 5 simulated shifts)")
fixed = []
for target in shifts.values():
    rs, ry = resample_to(target)
    fixed.append(qwk_of(rs, ry, th_qwk))
print(f"dev-QWK-thresholds (fixed)     {np.mean(fixed)*100:7.3f}   {np.min(fixed)*100:7.3f}")

for alpha in (0, 0.25, 0.5, 0.75, 1.0):
    T = (1 - alpha) * p_dev + alpha * p_train
    scores = []
    for target in shifts.values():
        rs, ry = resample_to(target)
        # Thresholds come from the shifted sample's own score quantiles, which is what makes
        # this usable without target labels.
        scores.append(qwk_of(rs, ry, distmatch_thresholds(rs, T)))
    print(f"distmatch T=(1-a)dev+a*train a={alpha:.2f} {np.mean(scores)*100:7.3f}   {np.min(scores)*100:7.3f}")
