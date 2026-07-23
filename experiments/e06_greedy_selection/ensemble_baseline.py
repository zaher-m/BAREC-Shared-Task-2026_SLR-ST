"""First try at ensembling: uniform vs dev-QWK-weighted average over all cached members.

Superseded by ensemble_select.py, which beat both by picking a weighted subset instead of
using everything. Kept as the baseline greedy had to beat.
"""
import argparse

import numpy as np

from slra_st import paths, submission
from slra_st.calibration import QWKThresholdOptimizer
from slra_st.ensembling import combine, load_meta, load_pool, save_config
from slra_st.metrics import print_report, qwk


def evaluate(weights, dev, test):
    dev_s = combine(dev.X, weights)
    opt = QWKThresholdOptimizer().fit(dev_s, dev.y)
    dev_q = qwk(dev.y, opt.predict(dev_s))
    test_q = qwk(test.y, opt.predict(combine(test.X, weights)))
    return dev_q, test_q, opt


def main():
    ap = argparse.ArgumentParser()
    # Both outputs default to the gitignored scratch directory: this script globs the whole pool,
    # so its result depends on how many members are on disk and must not land in the committed tree.
    ap.add_argument("--out", default=str(paths.REGENERATED / "prediction_test"))
    ap.add_argument("--config-out", default=str(paths.REGENERATED / "baseline.json"))
    args = ap.parse_args()
    paths.REGENERATED.mkdir(parents=True, exist_ok=True)

    dev = load_pool("dev")
    test = load_pool("test")
    metas = {t: load_meta(t) for t in dev.tags}
    print("Members:")
    for t in dev.tags:
        m = metas[t]
        print(f"  {t:22s} dev_qwk_cal={m.get('dev_qwk_cal'):.3f} "
              f"({m['model']} / {m['objective']} / {m['variant']})")

    uniform = np.ones(len(dev.tags))
    # Shift the QWK scale instead of using it raw, so the weakest member keeps a small
    # weight instead of dropping to zero.
    qwk_w = np.array([metas[t].get("dev_qwk_cal", 1.0) for t in dev.tags])
    qwk_w = np.clip(qwk_w - qwk_w.min() + 1.0, 1e-3, None)

    best = None
    for name, w in (("uniform", uniform), ("qwk-weighted", qwk_w)):
        dev_q, test_q, opt = evaluate(w, dev, test)
        print(f"[{name}] dev_QWK={dev_q*100:.4f}  test_QWK={test_q*100:.4f}")
        if best is None or dev_q > best[0]:
            best = (dev_q, name, w, opt)

    dev_q, name, w, opt = best
    print(f"\n== selected weighting: {name} (dev_QWK={dev_q*100:.4f}) ==")
    test_pred = opt.predict(combine(test.X, w)).astype(int)
    print_report("ENSEMBLE DEV ", dev.y, opt.predict(combine(dev.X, w)).astype(int))
    print_report("ENSEMBLE TEST", test.y, test_pred)

    submission.write(args.out, test.ids, test_pred)
    print(f"wrote {args.out}(.zip)")
    save_config(args.config_out, dev.tags, w.tolist(), opt.thresholds_.tolist(), name)
    print(f"wrote {args.config_out}")


if __name__ == "__main__":
    main()
