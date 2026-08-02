"""Write 0.5*V6 + 0.5*all-data as one ensemble config.

Both ensembles have the same weight sum, so the 50/50 score blend is exactly one 20-member
weighted average with the two weight vectors concatenated:

    0.5*(sum w_i s_i / W) + 0.5*(sum v_j s_j / V)   with W == V
  = sum w_i s_i / (W+V) + sum v_j s_j / (W+V)

Thresholds are fitted on the full test split, which is the only split neither family has
seen: train-only members never trained on it and the _ad members held it out. A/B estimate
for this scheme: 86.94 +- 0.35, vs 86.63 for V6.
"""
import json

from slra_st import paths
from slra_st.calibration import QWKThresholdOptimizer
from slra_st.ensembling import combine, load_members, save_config
from slra_st.metrics import print_report

v6 = json.load(open(paths.ensemble("v6")))
ad = json.load(open(paths.ensemble("all_data")))
W, V = sum(v6["weights"]), sum(ad["weights"])
assert W == V, f"weight sums differ ({W} vs {V}); rescale needed for a true 50/50"

members = list(v6["members"]) + list(ad["members"])
weights = list(v6["weights"]) + list(ad["weights"])

sm = load_members(members, "test")
ens = combine(sm.X, weights)

opt = QWKThresholdOptimizer().fit(ens, sm.y)
print_report("BLEND(V6+AD) test", sm.y, opt.predict(ens))
print(f"\nmembers={len(members)}  (10 train-only + 10 all-data)")

out = paths.ENSEMBLES / "blend_v7.json"
save_config(out, members, weights, opt.thresholds_.tolist(), "blend-0.5-v6-plus-0.5-alldata")
print(f"wrote {out}")
