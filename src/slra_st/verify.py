"""Recompute the reported numbers from the tracked caches and check them.

Run this first. It needs no GPU, no corpus and no trained weights, only what is committed in
artifacts/scores/, artifacts/metadata/ and configs/. Every number it prints is recomputed here
and compared against the value written in docs/, so a failure means the docs and the artifacts
disagree.

    python -m slra_st.verify            # everything (~2 min)
    python -m slra_st.verify --quick    # skip the A/B resampling checks (~10 s)

What it does NOT cover: numbers that were produced by scripts that were not preserved. Those
are listed in docs/reproducibility.md and are marked in the docs where they appear.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from . import paths
from .calibration import QWKThresholdOptimizer, fit_thresholds, labels_from_thresholds
from .ensembling import ensemble_scores, honest_ab, load_blind
from .metrics import fast_qwk, full_report
from .shift import bbse, prior_mean_var, prior_of, resample_indices, train_label_prior

CONFIGS = ["v4", "v5", "v6", "all_data", "blend_v7", "final_v8",
           "research/new3", "research/psfam", "research/allam", "research/kfemd"]


class Report:
    def __init__(self):
        self.rows = []

    def check(self, name, got, expected, tol=None, note=""):
        if tol is None:
            ok = got == expected
            shown = f"{got} vs {expected}"
        else:
            ok = abs(float(got) - float(expected)) <= tol
            shown = f"{float(got):.4f} vs {float(expected):.4f}"
        self.rows.append((ok, name, shown, note))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52} {shown}  {note}")
        return ok

    def summary(self):
        failed = [r for r in self.rows if not r[0]]
        print()
        print("=" * 78)
        print(f"{len(self.rows) - len(failed)}/{len(self.rows)} checks passed")
        for _, name, shown, _ in failed:
            print(f"  FAILED: {name}  ({shown})")
        print("=" * 78)
        return not failed


def check_configs(r):
    print("\n-- ensemble configs: structure --")
    for name in CONFIGS:
        cfg = json.load(open(paths.ensemble(name)))
        th = np.asarray(cfg["thresholds"], float)
        r.check(f"{name}: 18 monotone thresholds", bool(len(th) == 18 and np.all(np.diff(th) >= 0)), True)
        missing = [t for t in cfg["members"] if not paths.member_scores(t).exists()]
        r.check(f"{name}: all {len(cfg['members'])} members cached", missing, [])


def check_model_records(r):
    print("\n-- per-model records: metadata vs cached scores --")
    files = sorted(glob.glob(str(paths.MEMBER_SCORES / "scores_*.npz")))
    bad = []
    for f in files:
        tag = os.path.basename(f)[len("scores_"):-len(".npz")]
        d = np.load(f, allow_pickle=True)
        meta = json.load(open(paths.meta(tag)))
        q = fast_qwk(d["dev_labels"].astype(int) - 1,
                    labels_from_thresholds(d["dev_scores"], meta["thresholds"]) - 1) * 100
        if abs(q - meta["dev_qwk_cal"]) > 0.01:
            bad.append(tag)
    r.check(f"all {len(files)} models: stored cal QWK recomputes", bad, [],
            note="thresholds in meta.json applied to cached scores")
    r.check("model count", len(files), 61)
    r.check("training label counts sum to the split size",
            json.load(open(paths.METADATA / "train_label_counts.json"))["n_sentences"], 54845)
    r.check("training label prior mean", (train_label_prior() * np.arange(1, 20)).sum(), 10.5749, tol=0.001)


def check_ensembles(r, quick):
    print("\n-- ensemble scores on the labeled splits --")
    sd, yd = ensemble_scores("v6", "dev")
    st, yt = ensemble_scores("v6", "test")
    thd = fit_thresholds(sd, yd)
    r.check("V6 dev QWK (dev-fitted thresholds)",
            fast_qwk(yd - 1, labels_from_thresholds(sd, thd) - 1) * 100, 85.894, tol=0.005)
    r.check("V6 test QWK (dev-fitted thresholds)",
            fast_qwk(yt - 1, labels_from_thresholds(st, thd) - 1) * 100, 86.398, tol=0.005)

    s8, y8 = ensemble_scores("final_v8", "test")
    rep = full_report(y8, labels_from_thresholds(s8, fit_thresholds(s8, y8)))
    for key, exp in (("QWK", 87.2653), ("Acc19", 36.21), ("Acc7", 59.83), ("Acc5", 66.96),
                     ("Acc3", 74.46), ("Adj+-1", 74.94), ("MAE", 1.0852)):
        r.check(f"V8 full-test fit: {key}", rep[key], exp, tol=0.005,
                note="run_endgame.log" if key == "QWK" else "")

    print("\n-- committed thresholds == a fresh fit on the same split --")
    for name, split in (("v4", "dev"), ("v5", "dev"), ("v6", "dev"),
                        ("all_data", "test"), ("blend_v7", "test"), ("final_v8", "test")):
        cfg = json.load(open(paths.ensemble(name)))
        s, y = ensemble_scores(name, split)
        r.check(f"{name}: thresholds reproduce ({split} fit)",
                bool(np.allclose(cfg["thresholds"], fit_thresholds(s, y))), True)

    if quick:
        print("\n-- A/B holdout checks skipped (--quick) --")
        return
    print("\n-- A/B holdout QWK (fit on half, score the other half, 6 permutations) --")
    for name, exp_m, exp_sd in (("v6", 86.628, 0.327), ("all_data", 86.834, 0.265),
                                ("blend_v7", 86.936, 0.345), ("final_v8", 86.946, 0.369)):
        s, y = ensemble_scores(name, "test")
        m, sd = honest_ab(s, y)
        r.check(f"{name}: A/B mean", m, exp_m, tol=0.005)
        r.check(f"{name}: A/B sd", sd, exp_sd, tol=0.005)


def check_shift(r):
    print("\n-- blind label-shift estimate (BBSE, one pass, submission 8's anchor) --")
    S, y = ensemble_scores("final_v8", "test")
    bs = load_blind("prediction_final")[3]
    th040 = np.asarray(json.load(open(paths.THRESHOLDS / "acc_frontier.json"))["0.004"], float)
    p_hat = bbse(y, labels_from_thresholds(S, th040), prior_of(labels_from_thresholds(bs, th040)))
    mean, var = prior_mean_var(p_hat)
    r.check("estimated blind prior mean", mean, 10.47, tol=0.01)
    r.check("estimated blind prior variance", var, 8.99, tol=0.01, note="test variance is 10.63")
    r.check("test label variance", float(y.var()), 10.628, tol=0.005)


def check_submissions(r):
    print("\n-- submitted files re-derived from configs + the cached blind matrix --")
    bids, X, w, bs = load_blind("prediction_final")
    r.check("blind cache shape", list(X.shape), [8077, 21])
    fr = json.load(open(paths.THRESHOLDS / "acc_frontier.json"))
    cfg8 = json.load(open(paths.ensemble("final_v8")))
    pc = json.load(open(paths.THRESHOLDS / "prior_calibrated.json"))["thresholds"]
    cases = [
        ("prediction_final", cfg8["thresholds"], "V8, submission 7 (blind 85.3)"),
        ("pred_eps0.40_maxAcc", fr["0.004"], "submission 8 (blind 85.3 / 40.4)"),
        ("pred_eps0.10_safeQWK", fr["0.001"], "submission 9 (blind 85.1 / 37.9)"),
        ("pred_priorQWK_acc", pc, "submission 10, prior-calibrated (blind 85.4 / 38.7)"),
    ]
    for fname, th, note in cases:
        got = labels_from_thresholds(bs, np.asarray(th, float)).astype(int)
        want = pd.read_csv(paths.SUBMITTED / fname)["Prediction"].values
        r.check(f"{fname} reproduces exactly", int((got != want).sum()), 0, note=note)
    r.check("acc_frontier eps=0.0 == V8 thresholds",
            bool(np.allclose(fr["0.0"], cfg8["thresholds"])), True)
    r.check("prior-calibrated thresholds monotone",
            bool(np.all(np.diff(np.asarray(pc, float)) >= 0)), True)
    v8_labels = labels_from_thresholds(bs, np.asarray(cfg8["thresholds"], float)).astype(int)
    pc_labels = labels_from_thresholds(bs, np.asarray(pc, float)).astype(int)
    r.check("prior-calibrated vs V8: rows the re-thresholding moved",
            int((v8_labels != pc_labels).sum()), 922,
            note="11.4% of rows, mean shift -0.58 levels")
    ids_ok = bids == pd.read_csv(paths.SUBMITTED / "prediction_final")["Sentence ID"].astype(str).tolist()
    r.check("blind id order matches the submission", ids_ok, True)


def check_prior_calibration_chain(r):
    """Re-run submission 10's whole calibration chain, from the caches up.

    The script that emitted the file on deadline night was not kept, but its procedure was
    transcribed into the rebuild block of e11/weighted_opt.py minutes afterwards, and that block
    is deterministic. This recomputes it rather than reading a stored threshold vector, so a
    failure here means the documented method no longer produces the file that was uploaded.
    """
    print("\n-- submission 10, recomputed end to end from the cached scores --")
    S, y = ensemble_scores("final_v8", "test")
    bs = load_blind("prediction_final")[3]
    th040 = np.asarray(json.load(open(paths.THRESHOLDS / "acc_frontier.json"))["0.004"], float)
    p0 = bbse(y, labels_from_thresholds(S, th040), prior_of(labels_from_thresholds(bs, th040)))
    idx = resample_indices(y, p0, np.random.RandomState(0), 20000)
    th = QWKThresholdOptimizer().fit(S[idx], y[idx]).thresholds_

    stored = np.asarray(json.load(open(paths.THRESHOLDS / "prior_calibrated.json"))["thresholds"], float)
    r.check("prior-calibrated thresholds: recomputed == committed", bool(np.allclose(th, stored)), True,
            note="BBSE -> resample(seed 0, n=20000) -> QWK fit")
    got = labels_from_thresholds(bs, th).astype(int)
    want = pd.read_csv(paths.SUBMITTED / "pred_priorQWK_acc")["Prediction"].values
    r.check("submission 10: recomputed chain reproduces the upload", int((got != want).sum()), 0,
            note="0 of 8077 rows, no stored thresholds used")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quick", action="store_true", help="skip the A/B resampling checks")
    args = ap.parse_args()

    print("=" * 78)
    print("Verifying the reported numbers against the committed artifacts")
    print("=" * 78)
    r = Report()
    check_configs(r)
    check_model_records(r)
    check_ensembles(r, args.quick)
    check_shift(r)
    check_submissions(r)
    check_prior_calibration_chain(r)
    raise SystemExit(0 if r.summary() else 1)


if __name__ == "__main__":
    main()
