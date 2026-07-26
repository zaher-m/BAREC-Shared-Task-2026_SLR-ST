"""Where does the ensemble lose QWK, and does blind look shifted?

Four things: the most expensive confusion cells, per-level recall and signed bias, the dev
marginal vs the true marginal, and the blind predicted distribution vs the training prior.

This is the script that found the level-12 under-prediction tail. It is a bias, not noise,
which is why seed replicas did nothing for it and more capacity did. The last section was
also the first hint that the blind prior is more concentrated than test's.
"""
import numpy as np
import pandas as pd

from slra_st import paths
from slra_st.calibration import QWKThresholdOptimizer
from slra_st.ensembling import ensemble_scores
from slra_st.metrics import QUAD_W, qwk
from slra_st.shift import LEVELS, train_label_prior

dev_s, dev_y = ensemble_scores("v6", "dev")
test_s, test_y = ensemble_scores("v6", "test")
opt = QWKThresholdOptimizer().fit(dev_s, dev_y)
dev_p, test_p = opt.predict(dev_s), opt.predict(test_s)
print(f"ensemble dev QWK={qwk(dev_y, dev_p)*100:.3f}  test QWK={qwk(test_y, test_p)*100:.3f}\n")

O = np.zeros((19, 19))
for t, p in zip(dev_y - 1, dev_p - 1):
    O[t, p] += 1
cost = QUAD_W * O
total = cost.sum()
print("=== Top-12 costliest confusion cells on DEV (true->pred: count, dist, cost-share%) ===")
cells = [(i + 1, j + 1, int(O[i, j]), abs(i - j), cost[i, j] / total * 100)
         for i in range(19) for j in range(19) if i != j and O[i, j] > 0]
for t, p, c, dist, share in sorted(cells, key=lambda x: -x[4])[:12]:
    print(f"  true {t:2d} -> pred {p:2d} : n={c:4d}  |dist|={dist}  cost={share:4.1f}%")

print("\n=== Per-true-level (DEV): n, recall, mean signed error (pred-true) ===")
for t in range(1, 20):
    mask = dev_y == t
    if mask.sum() == 0:
        continue
    print(f"  L{t:2d}: n={mask.sum():4d} recall={(dev_p[mask]==t).mean()*100:5.1f}%  "
          f"mean(pred-true)={(dev_p[mask]-t).mean():+.2f}")

print("\n=== DEV marginal: true vs predicted share per level ===")
true_share = np.bincount(dev_y, minlength=20)[1:] / len(dev_y) * 100
pred_share = np.bincount(dev_p, minlength=20)[1:] / len(dev_p) * 100
print("  L  true%  pred%")
for l in range(19):
    print(f"  {l+1:2d} {true_share[l]:5.1f} {pred_share[l]:6.1f}")

# The blind set ships only ID + Sentence, so there is no metadata to compare against.
# The available proxy is this system's own predicted blind distribution vs the training prior.
print("\n=== BLIND predicted distribution vs training prior (shift probe) ===")
p_train = train_label_prior()
train_share = p_train * 100
bp = pd.read_csv(paths.SUBMITTED / "prediction")
blind_share = np.bincount(bp["Prediction"].astype(int), minlength=20)[1:] / len(bp) * 100
print("  L  train%  devTrue%  BLINDpred%   (blind-train)")
for l in range(19):
    print(f"  {l+1:2d} {train_share[l]:6.1f} {true_share[l]:8.1f} {blind_share[l]:10.1f}"
          f"   {blind_share[l]-train_share[l]:+.1f}")
print(f"\n  mean predicted blind level: {bp['Prediction'].mean():.2f}")
print(f"  mean train level:           {(p_train * LEVELS).sum():.2f}")
print(f"  mean dev level:             {dev_y.mean():.2f}")
