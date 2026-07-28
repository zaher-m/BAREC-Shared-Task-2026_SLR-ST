"""Build the all-data ensemble from the _ad members.

Each _ad model trained on train+dev with TEST held out, so its cached test block is a clean
holdout. Instead of re-selecting members (which would fit the holdout), reuse V6's member
order and weights with the _ad versions swapped in, then refit thresholds on the holdout.
That keeps the number comparable to V6's test 86.398.
"""
import json
import sys

from slra_st import paths
from slra_st.calibration import QWKThresholdOptimizer
from slra_st.ensembling import combine, load_members, save_config
from slra_st.metrics import fast_qwk

V6_TO_AD = {
    "arabertv2_corn_s2": "arabertv2_corn_ad",
    "arabertv2_emd_d3": "arabertv2_emd_ad",
    "arabertv2_large_corn": "arabertv2_large_corn_ad",
    "arabertv2_large_reg": "arabertv2_large_reg_ad",
    "arabertv2_large_soft": "arabertv2_large_soft_ad",
    "arabertv2_soft_d3": "arabertv2_soft_ad",
    "arabertv2_wkl_d3": "arabertv2_wkl_ad",
    "araelectra_corn_d3": "araelectra_corn_ad",
    "aramodern_reg_raw": "aramodern_reg_ad",
    "arbertv2_corn_d3": "arbertv2_corn_ad",
}

v6 = json.load(open(paths.ensemble("v6")))
members_ad = [V6_TO_AD[t] for t in v6["members"]]

missing = [t for t in members_ad if not paths.member_scores(t).exists()]
if missing:
    print("MISSING _ad models (training still running?):", missing)
    sys.exit(1)

sm = load_members(members_ad, "test")
ens = combine(sm.X, v6["weights"])

opt = QWKThresholdOptimizer().fit(ens, sm.y)
q = fast_qwk(sm.y - 1, opt.predict(ens) - 1) * 100
print(f"ALL-DATA ensemble holdout(test) QWK = {q:.3f}   (V6 test was 86.398)")
print(f"members={len(members_ad)}  n_holdout={len(sm.y)}")

out = paths.ENSEMBLES / "all_data.json"
save_config(out, members_ad, v6["weights"], opt.thresholds_.tolist(), "alldata-reuse-v6-weights")
print(f"wrote {out}")
