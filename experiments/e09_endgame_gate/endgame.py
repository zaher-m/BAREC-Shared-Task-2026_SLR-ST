"""Enlarge the blend with whatever _ad models finished, and only ship it if it improves.

AD half = the 10 mapped members at V6 weights, plus any new _ad model at weight 1. Uniform
for the new ones because greedy selection had just been shown to overfit at this pool size.
Blend stays 0.5*V6 + 0.5*AD, the equal-regime split that beat every fitted weighting.

The gate is an A/B on clean test against the currently shipped blend. If the candidate does
not clear the gate, nothing is written and the previous submission stands.

By default the AD half is whatever _ad models are on disk, which is what the original run did.
That makes the output depend on WHEN you run it: with the full zoo present the candidate no
longer clears the gate (86.934 vs 86.936) and nothing is written. Pass --ad-members to pin the AD half and
reproduce a specific configuration; the 11 tags behind the shipped V8 are listed in README.md.
"""
import argparse
import glob
import json
import os

import numpy as np

from slra_st import paths
from slra_st.calibration import QWKThresholdOptimizer
from slra_st.ensembling import combine, honest_ab, load_members, save_config
from slra_st.metrics import print_report

V6_TO_AD = {
    "arabertv2_corn_s2": "arabertv2_corn_ad", "arabertv2_emd_d3": "arabertv2_emd_ad",
    "arabertv2_large_corn": "arabertv2_large_corn_ad", "arabertv2_large_reg": "arabertv2_large_reg_ad",
    "arabertv2_large_soft": "arabertv2_large_soft_ad", "arabertv2_soft_d3": "arabertv2_soft_ad",
    "arabertv2_wkl_d3": "arabertv2_wkl_ad", "araelectra_corn_d3": "araelectra_corn_ad",
    "aramodern_reg_raw": "aramodern_reg_ad", "arbertv2_corn_d3": "arbertv2_corn_ad",
}


def scores_of(tags, weights, split="test"):
    sm = load_members(tags, split)
    return combine(sm.X, weights), sm.y


ap = argparse.ArgumentParser()
ap.add_argument("--ad-members", nargs="*", default=None,
                help="pin the AD half instead of globbing whatever is on disk")
ap.add_argument("--config-out", default=str(paths.ENSEMBLES / "final_v8.json"))
args = ap.parse_args()

have = {os.path.basename(f)[len("scores_"):-len(".npz")]
        for f in glob.glob(str(paths.MEMBER_SCORES / "scores_*.npz"))}
if args.ad_members:
    have = set(args.ad_members)

v6 = json.load(open(paths.ensemble("v6")))
cur = json.load(open(paths.ensemble("blend_v7")))
s_cur, y = scores_of(cur["members"], cur["weights"])
m_cur, sd_cur = honest_ab(s_cur, y)
print(f"CURRENT blend ({len(cur['members'])} members):  holdout {m_cur:.3f} +- {sd_cur:.3f}")

ad_tags, ad_w = [], []
for t, wt in zip(v6["members"], v6["weights"]):
    a = V6_TO_AD[t]
    if a in have:
        ad_tags.append(a)
        ad_w.append(wt)
extra = sorted(t for t in have if t.endswith("_ad") and t not in ad_tags)
ad_tags += extra
ad_w += [1] * len(extra)
print(f"AD half: {len(ad_tags)} members ({len(extra)} new: {extra})")

v6_tags, v6_w = list(v6["members"]), list(v6["weights"])
s_v6, _ = scores_of(v6_tags, v6_w)
s_ad, _ = scores_of(ad_tags, ad_w)
m_ad, sd_ad = honest_ab(s_ad, y)
s_new = 0.5 * s_v6 + 0.5 * s_ad
m_new, sd_new = honest_ab(s_new, y)
print(f"AD half alone:                 holdout {m_ad:.3f} +- {sd_ad:.3f}")
print(f"CANDIDATE blend:               holdout {m_new:.3f} +- {sd_new:.3f}")

if m_new > m_cur:
    # Wording frozen: artifacts/logs/run_endgame.log (Aug 2) contains this line verbatim, and
    # docs/reproducibility.md claims the script reproduces that log line for line. Rephrasing it
    # would break the claim, so it stays as it was written on the night.
    print(f"\n=> CANDIDATE WINS by {m_new - m_cur:+.3f}; writing {args.config_out}")
    # Rescale the AD weights so both halves contribute equally even though the sums differ.
    scale = sum(v6_w) / sum(ad_w)
    members = v6_tags + ad_tags
    weights = [float(x) for x in v6_w] + [float(x) * scale for x in ad_w]
    s_chk, _ = scores_of(members, weights)
    assert np.allclose(s_chk, s_new, atol=1e-9), "flattened weights must reproduce the blend"
    opt = QWKThresholdOptimizer().fit(s_chk, y)
    print_report("FINAL blend test", y, opt.predict(s_chk))
    out = args.config_out
    save_config(out, members, weights, opt.thresholds_.tolist(),
                "final-0.5-v6-plus-0.5-alldata-enlarged")
    print(f"wrote {out}")
else:
    print(f"\n=> candidate does NOT beat current ({m_new - m_cur:+.3f}); keep the shipped blend")
