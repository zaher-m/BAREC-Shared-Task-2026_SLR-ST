"""Label-shift estimation (BBSE) and importance weighting.

The blind prior is more concentrated than test's (estimated Var_true 9.09 vs 10.63). Since
QWK = 1 - MSE/(Var_true + Var_pred + dmu^2), a smaller Var_true shrinks the denominator, so
the ~2 QWK test-to-blind drop comes from the metric, not from the models. Estimating the
prior and refitting the thresholds against it gave blind 85.3 -> 85.4 and +2.8 Acc, with no
retraining.

Run BBSE ONCE. Re-running it anchored on its own new output made the estimate worse
(prior var 8.99 -> 9.31) and blind went 85.4 -> 85.2.
"""
import json

import numpy as np

from . import paths
from .metrics import N_CLASSES

LEVELS = np.arange(1, N_CLASSES + 1).astype(float)


def prior_of(labels):
    p = np.bincount(np.asarray(labels, int), minlength=N_CLASSES + 1)[1:].astype(float)
    return p / p.sum()


def train_label_prior():
    """Label prior of the training split, read from artifacts/metadata/train_label_counts.json.

    Committed as aggregate counts (no corpus text) so the analysis scripts run without the
    corpus. Same numbers as the table in docs/data.md.
    """
    counts = json.load(open(paths.METADATA / "train_label_counts.json"))["counts"]
    p = np.asarray(counts, float)
    return p / p.sum()


def prior_mean_var(p):
    p = np.asarray(p, float)
    mean = float((p * LEVELS).sum())
    return mean, float((p * (LEVELS - mean) ** 2).sum())


def bbse(y_cal, pred_cal, target_pred_prior, lam=0.05):
    """Estimate the target label prior from predictions only (no target labels).

    Solves min ||C^T p - q||^2 + lam ||p - p_src||^2, where C[i,j] = P(pred=j | true=i) on
    the calibration split and q is the target's predicted distribution. The ridge term stops
    the 19-dim solve from chasing noise in the rare levels. lam=0.05 everywhere in this work.
    """
    C = np.zeros((N_CLASSES, N_CLASSES))
    for t, pr in zip(np.asarray(y_cal, int), np.asarray(pred_cal, int)):
        C[t - 1, pr - 1] += 1
    C = C / np.clip(C.sum(1, keepdims=True), 1, None)
    p_src = prior_of(y_cal)
    A = np.vstack([C.T, np.sqrt(lam) * np.eye(N_CLASSES)])
    b = np.concatenate([np.asarray(target_pred_prior, float), np.sqrt(lam) * p_src])
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    p = np.clip(p, 1e-6, None)
    return p / p.sum()


def importance_weights(y, target_prior, source_prior=None):
    """Raw ratios p_target[y]/p_source[y]. Normalize in the caller (/sum for sampling,
    /mean for weighting)."""
    y = np.asarray(y, int)
    src = prior_of(y) if source_prior is None else np.asarray(source_prior, float)
    return np.asarray(target_prior, float)[y - 1] / np.clip(src[y - 1], 1e-9, None)


def resample_indices(y, target_prior, rng, n, source_prior=None):
    """Draw indices from `y` reweighted to `target_prior`.

    This is the noisy version of exact importance weighting; kept because the earlier
    experiments used it.
    """
    w = importance_weights(y, target_prior, source_prior)
    w = w / w.sum()
    return rng.choice(len(y), n, replace=True, p=w)


def tilt_prior(p, k):
    """Tilt a prior: k<0 concentrates toward the mean, k>0 spreads out.

    Used to fake blind-like shifts on a labeled split so an adaptation can be checked
    before it is used on the real blind set.
    """
    p = np.asarray(p, float)
    mean = (p * LEVELS).sum()
    q = p * np.exp(k * ((LEVELS - mean) / 9.0) ** 2)
    return q / q.sum()
