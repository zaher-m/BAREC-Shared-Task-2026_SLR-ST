"""Continuous score -> one of the 19 levels, via 18 cut points.

Fitting the cut points for QWK instead of rounding is worth +1 to +2 QWK per model here
(the highest-scoring 2025 system reported +6.3), so it is the main post-processing step.
"""
import numpy as np

from .metrics import N_CLASSES, fast_qwk, weighted_accuracy, weighted_qwk

# Uncalibrated fallback: level midpoints, top two cuts pulled in because levels 18-19 are
# almost never predicted. Only the post-deadline research inferences use this.
DEFAULT_THRESHOLDS = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5,
                      11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.2, 18.6]


def labels_from_thresholds(scores, thresholds):
    """Map continuous scores to levels 1..19 via 18 monotonic cut points."""
    scores = np.asarray(scores, dtype=float)
    th = np.asarray(thresholds, dtype=float)
    return 1 + np.sum(scores[:, None] > th[None, :], axis=1)


class QWKThresholdOptimizer:
    """Coordinate ascent over the 18 cut points, maximizing QWK.

    Each cut point only moves between its two neighbours, so monotonicity holds by
    construction. Use rounds=3, grid=0.1 inside slow loops (greedy selection); the
    defaults are for final numbers.
    """

    def __init__(self, n_classes=N_CLASSES, rounds=8, grid=0.02):
        self.n = n_classes
        self.rounds = rounds
        self.grid = grid
        self.thresholds_ = None

    def fit(self, scores, y_true):
        scores = np.asarray(scores, float)
        y0 = np.asarray([int(x) for x in y_true]) - 1
        lo, hi = float(scores.min()) - 1.0, float(scores.max()) + 1.0
        th = np.array([min(max(k + 0.5, lo), hi) for k in range(1, self.n)], float)
        best = fast_qwk(y0, labels_from_thresholds(scores, th) - 1)
        candidates = np.arange(lo, hi + self.grid, self.grid)
        for _ in range(self.rounds):
            improved = False
            for i in range(self.n - 1):
                cur = th[i]
                left = th[i - 1] if i > 0 else lo
                right = th[i + 1] if i < self.n - 2 else hi
                local = candidates[(candidates > left) & (candidates < right)]
                for c in local:
                    th[i] = c
                    score = fast_qwk(y0, labels_from_thresholds(scores, th) - 1)
                    if score > best + 1e-9:
                        best, cur, improved = score, c, True
                th[i] = cur
            if not improved:
                break
        self.thresholds_ = th
        self.best_qwk_ = best
        return self

    def predict(self, scores):
        return labels_from_thresholds(scores, self.thresholds_)


def fit_thresholds(scores, y_true, rounds=8, grid=0.02):
    """Shortcut that returns just the cut points."""
    return QWKThresholdOptimizer(rounds=rounds, grid=grid).fit(scores, y_true).thresholds_


def optimize_thresholds(scores, y_true, objective="acc", init=None, qwk_floor=None,
                        weights=None, rounds=6, grid=0.04, tol=1e-9):
    """Coordinate ascent on QWK or on exact accuracy, with an optional QWK floor.

    QWK is flat near its optimum but accuracy is not, so maximizing Acc subject to
    QWK >= floor buys accuracy for very little QWK. Passing `weights` evaluates both the
    objective and the floor under importance weights (i.e. against an estimated target
    prior instead of the calibration set's own prior).
    """
    scores = np.asarray(scores, float)
    y_true = np.asarray(y_true, int)
    sw = None if weights is None else np.asarray(weights, float)

    def score_qwk(th):
        pred = labels_from_thresholds(scores, th)
        return weighted_qwk(y_true, pred, sw) if sw is not None else fast_qwk(y_true - 1, pred - 1)

    def score_acc(th):
        pred = labels_from_thresholds(scores, th)
        return weighted_accuracy(y_true, pred, sw) if sw is not None else float((pred == y_true).mean())

    objective_fn = score_qwk if objective == "qwk" else score_acc
    th = np.array(init if init is not None else fit_thresholds(scores, y_true), float)
    best = objective_fn(th)
    lo, hi = scores.min() - 0.5, scores.max() + 0.5
    for _ in range(rounds):
        moved = False
        for i in range(len(th)):
            cur = th[i]
            lb = th[i - 1] if i > 0 else lo
            ub = th[i + 1] if i < len(th) - 1 else hi
            if ub <= lb:
                continue
            for cand in np.arange(lb, ub + 1e-9, grid):
                th[i] = cand
                if not np.all(np.diff(th) >= 0):
                    continue
                v = objective_fn(th)
                if v > best + tol and (qwk_floor is None or score_qwk(th) >= qwk_floor):
                    best, cur, moved = v, cand, True
            th[i] = cur
        if not moved:
            break
    return th
