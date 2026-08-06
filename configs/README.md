# Configs

An ensemble config is the complete specification of a system: which members, at which weights, with
which 18 threshold cut points. Given the cached member scores, a config plus
`python -m slra_st.infer` reproduces a submission exactly.

```json
{
  "members":    ["arabertv2_corn_s2", "..."],
  "weights":    [1, 1, 2, "..."],
  "weighting":  "greedy-multiset",
  "thresholds": [2.5987, 2.6387, "... 18 values ..."]
}
```

## ensembles/

| File | Members | Weighting | Blind | Built by |
|---|---|---|---|---|
| `v4.json` | 7 | greedy multiset | 84.4 | [e06](../experiments/e06_greedy_selection/) |
| `v5.json` | 11 | greedy multiset | - | e06 |
| `v6.json` | 10 | greedy multiset | 85.0 | e06 |
| `all_data.json` | 10 | V6's map and weights on `_ad` substitutes | - | [e07](../experiments/e07_all_data_retraining/) |
| `blend_v7.json` | 20 | 0.5·V6 + 0.5·all-data, flattened | 85.1 | [e08](../experiments/e08_combiner_bakeoff/) |
| **`final_v8.json`** | **21** | **as above, enlarged, rescaled to an exact 50/50 regime split** | **85.3 → 85.4** | [e09](../experiments/e09_endgame_gate/) |
| `research/{new3,psfam,allam,kfemd}.json` | 3 / 2 / 1 / 5 | uniform | - | [e12](../experiments/e12_post_deadline_ablations/) |


A few things worth knowing when reading these files:

- **`v6.json` was `ensemble.json`.** Renamed for clarity; the file is unchanged, including its fitted
  thresholds.
- **The AD weights in `final_v8.json` are fractional** (0.9285714285714286 = 13/14, doubled members
  1.857…). That is not a fitted result, it is the rescaling that makes the two data regimes
  contribute exactly equal mass despite different weight sums. `endgame.py` asserts the flattened
  vector reproduces the 50/50 score blend to 1e-9.
- **The four `research/` configs carry the uncalibrated default grid**, so their predictions are
   naive. They were made for their cached score matrices.
- **A diverged member is part of the shipped system.** `arabertv2_large_reg_ad` (QWK 0.0000, constant
  output) is a weighted member of `blend_v7.json` and `final_v8.json`. Joint calibration absorbs it and
  removing it was measured neutral, but it must be present to reproduce the submission bit-exactly.

## thresholds/

Fitted threshold sets kept separately because they are *systems in their own right* on top of a fixed
score matrix, the whole endgame consisted of re-thresholding one 21-member ensemble.

| File | Contents |
|---|---|
| `acc_frontier.json` | six sets keyed by ε ∈ {0, 0.0005, 0.001, 0.002, 0.004, 0.008}: max accuracy subject to QWK ≥ optimum − ε. The ε=0 set is identical to V8's |
| `push_acc.json` | the same optimization performed under the BBSE-estimated blind prior ([e10](../experiments/e10_accuracy_frontier/)) |

Per-model thresholds (61 × 18) are in `artifacts/metadata/meta_<tag>.json`. They are reporting
artifacts only, ensembles recalibrate jointly on the combined score and never use them.
