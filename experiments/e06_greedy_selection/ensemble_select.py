"""Greedy forward selection with replacement (Caruana style) over the model zoo.

Start from the best single member, then keep adding whichever member improves dev QWK most.
Repeats are allowed, so how often a member is picked becomes its weight. Each candidate is
re-calibrated before scoring: coarse during the search, fine for the final pick.

This produced V4 -> V5 -> V6. V6 ended up with four large models, five objective variants of
AraBERTv2 and only three other backbones, which is the objective-diversity and capacity
results showing up in an automatic selection.

Caveat found later: on the 49-model pool this same procedure overfits the selection split and
loses to a fixed 50/50 blend (see e08).
"""
import argparse
from collections import Counter

from slra_st import paths, submission
from slra_st.calibration import QWKThresholdOptimizer
from slra_st.ensembling import load_pool, save_config
from slra_st.metrics import print_report, qwk


def dev_qwk_of(score, y, coarse=True):
    opt = (QWKThresholdOptimizer(rounds=3, grid=0.1) if coarse else QWKThresholdOptimizer()).fit(score, y)
    return qwk(y, opt.predict(score)), opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--exclude", nargs="*", default=None, help="substrings of tags to exclude")
    ap.add_argument("--write", action="store_true", help="write the chosen ensemble and a test prediction")
    # Scratch by default: greedy selection is pool-dependent, so the committed V4/V5/V6 configs
    # must only be replaced by an explicit --config-out.
    ap.add_argument("--config-out", default=str(paths.REGENERATED / "greedy.json"))
    args = ap.parse_args()

    dev = load_pool("dev", args.exclude)
    test = load_pool("test", args.exclude)
    tags = dev.tags
    dev_cols = {t: dev.X[:, i] for i, t in enumerate(tags)}
    test_cols = {t: test.X[:, i] for i, t in enumerate(test.tags)}

    print(f"pool ({len(tags)}):")
    for t in tags:
        q, _ = dev_qwk_of(dev_cols[t], dev.y)
        print(f"  {t:22s} dev_QWK={q*100:.3f}")

    singles = sorted(((dev_qwk_of(dev_cols[t], dev.y)[0], t) for t in tags), reverse=True)
    chosen = [singles[0][1]]
    cur_sum = dev_cols[chosen[0]].copy()
    best_dev, _ = dev_qwk_of(cur_sum, dev.y)
    best_chosen = list(chosen)
    print(f"\ninit: {chosen[0]}  dev={best_dev*100:.3f}")

    for r in range(args.rounds):
        cand_best = (-1, None)
        for t in tags:
            avg = (cur_sum + dev_cols[t]) / (len(chosen) + 1)
            q, _ = dev_qwk_of(avg, dev.y)
            if q > cand_best[0]:
                cand_best = (q, t)
        q, t = cand_best
        chosen.append(t)
        cur_sum = cur_sum + dev_cols[t]
        if q > best_dev + 1e-9:
            best_dev, best_chosen = q, list(chosen)
        print(f"round {r+1}: +{t:22s} dev={q*100:.3f}  (best={best_dev*100:.3f})")

    counts = Counter(best_chosen)
    print(f"\n== selected multiset ({len(best_chosen)} picks) ==")
    for t, c in counts.most_common():
        print(f"  {c}x {t}")

    total = float(sum(counts.values()))
    dev_score = sum(counts[t] * dev_cols[t] for t in tags) / total
    test_score = sum(counts[t] * test_cols[t] for t in tags) / total
    opt = QWKThresholdOptimizer().fit(dev_score, dev.y)
    print_report("GREEDY DEV ", dev.y, opt.predict(dev_score))
    print_report("GREEDY TEST", test.y, opt.predict(test_score))

    if args.write:
        pred = opt.predict(test_score).astype(int)
        out = paths.PREDICTIONS / "prediction_test"
        submission.write(out, test.ids, pred)
        used = [t for t in tags if counts[t] > 0]
        save_config(args.config_out, used, [counts[t] for t in used],
                    opt.thresholds_.tolist(), "greedy-multiset")
        print(f"wrote {out}(.zip) and {args.config_out}")


if __name__ == "__main__":
    main()
