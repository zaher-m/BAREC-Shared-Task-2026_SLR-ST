"""Loading cached member scores, aligning them by sentence id, and combining them.

Training saved every model's continuous dev/test scores and every blind inference saved its
member matrix, so nothing in experiments/ needs a GPU. Score matrices here are always
[n_sentences, n_members].
"""
import glob
import json
import os
from typing import NamedTuple

import numpy as np

from . import paths
from .calibration import QWKThresholdOptimizer, labels_from_thresholds
from .metrics import fast_qwk


class ScoreMatrix(NamedTuple):
    """Member scores aligned on one sentence order."""
    ids: list      # sentence ids, length N
    tags: list     # member tags, length M
    X: np.ndarray  # [N, M] continuous scores
    y: np.ndarray  # [N] gold levels 1..19


def load_config(config):
    """Accept an ensemble dict, a config name ('v6', 'research/psfam') or a path."""
    if isinstance(config, dict):
        return config
    return json.load(open(paths.ensemble(str(config))))


def save_config(path, members, weights, thresholds, weighting):
    payload = {"members": list(members), "weights": list(weights),
               "weighting": weighting, "thresholds": list(thresholds)}
    json.dump(payload, open(path, "w"), indent=2)
    return payload


def load_meta(tag):
    return json.load(open(paths.meta(tag)))


def load_members(tags, split="test"):
    """Aligned scores for the given member tags.

    Careful: for the _ad/_ps/_kf regimes the cached dev and test blocks are the same
    holdout, so `split` only means something for train-only models.
    """
    tags = list(tags)
    ref, labels, cols = None, None, []
    for tag in tags:
        d = np.load(paths.member_scores(tag), allow_pickle=True)
        ids = [str(x) for x in d[f"{split}_ids"]]
        if ref is None:
            ref, labels = ids, d[f"{split}_labels"].astype(int)
        pos = {i: k for k, i in enumerate(ids)}
        s = d[f"{split}_scores"]
        cols.append(np.array([s[pos[i]] for i in ref]))
    return ScoreMatrix(ref, tags, np.vstack(cols).T, labels)


def load_pool(split="test", exclude=None):
    """Every cached member that shares the reference id set."""
    tags, cols, ref, labels = [], [], None, None
    for f in sorted(glob.glob(str(paths.MEMBER_SCORES / "scores_*.npz"))):
        tag = os.path.basename(f)[len("scores_"):-len(".npz")]
        if exclude and any(s in tag for s in exclude):
            continue
        d = np.load(f, allow_pickle=True)
        if f"{split}_ids" not in d:
            continue
        ids = [str(x) for x in d[f"{split}_ids"]]
        if ref is None:
            ref, labels = ids, d[f"{split}_labels"].astype(int)
        elif set(ids) != set(ref):
            continue
        pos = {i: k for k, i in enumerate(ids)}
        s = d[f"{split}_scores"]
        tags.append(tag)
        cols.append(np.array([s[pos[i]] for i in ref]))
    return ScoreMatrix(ref, tags, np.vstack(cols).T, labels)


def combine(X, weights):
    """Weighted average over members, ignoring zero-weight columns.

    Normalize the weights before the dot product, not after. Both are correct, but this order
    is what produced the committed configs, and doing it the other way shifts the fitted
    thresholds in the 16th decimal.
    """
    w = np.asarray(weights, float)
    nz = np.nonzero(w)[0]
    return X[:, nz] @ (w[nz] / w[nz].sum())


def weight_vector(tags, members, weights):
    """Spread an ensemble's member/weight lists over a pool's tag order."""
    index = {t: i for i, t in enumerate(tags)}
    w = np.zeros(len(tags))
    for tag, weight in zip(members, weights):
        w[index[tag]] = weight
    return w


def ensemble_scores(config, split="test"):
    """Combined score of an ensemble config on a labeled split: (scores, y)."""
    cfg = load_config(config)
    sm = load_members(cfg["members"], split)
    return combine(sm.X, cfg["weights"]), sm.y


def load_blind(name):
    """Cached blind member matrix written by infer.py: (ids, X[N, M], weights, ens_score)."""
    d = np.load(paths.BLIND_SCORES / f"{name}_scores.npz", allow_pickle=True)
    return ([str(x) for x in d["ids"]], d["member_scores"],
            np.asarray(d["weights"], float), d["ens_score"])


def ab_splits(n, reps=6, seed_offset=0):
    """A/B splits: fit on one half, score the other half, both directions.

    Anything fitted (thresholds, selection, weights) has to be fitted on the `fit` half
    only. Fitting 18 thresholds on ~3.6k points gives a +-0.1-0.2 QWK noise floor, so an
    improvement smaller than that means nothing.
    """
    for r in range(reps):
        perm = np.random.RandomState(seed_offset + r).permutation(n)
        A, B = perm[: n // 2], perm[n // 2:]
        yield A, B
        yield B, A


def qwk_fit_eval(scores_fit, y_fit, scores_eval, y_eval):
    th = QWKThresholdOptimizer().fit(scores_fit, y_fit).thresholds_
    return fast_qwk(y_eval - 1, labels_from_thresholds(scores_eval, th) - 1)


def honest_ab(scores, y, reps=6):
    """Mean +- sd holdout QWK (in percent) of a single scoring scheme."""
    out = [qwk_fit_eval(scores[f], y[f], scores[e], y[e]) for f, e in ab_splits(len(y), reps)]
    return float(np.mean(out)) * 100, float(np.std(out)) * 100
