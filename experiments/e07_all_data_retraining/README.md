# E7. All-data retraining

**When** Jul 27–28 (plus the divergence fix on Aug 3) · **Cost** ~18 runs · **Verdict** +0.31 honest

## Hypothesis

Two effects at once, from one change. Folding dev into training gives **+13% data** on already-proven
members, and a family trained on *different data* should have decorrelated errors from the
train-only family, which is a second, independent gain. Test becomes the early-stop and calibration
holdout; blind is never touched.

## Results (calibrated QWK on the test holdout)

| Model | Test holdout |
|---|---|
| arabertv2_wkl_ad | 85.52 |
| arabertv2_large_soft_ad | 85.51 |
| arabertv2_soft_ad | 85.43 |
| arabertv2_large_corn_ad | 85.12 |
| arabertv2_corn_ad | 85.07 |
| arabertv2_emd_ad | 85.06 |
| arabertv2_reg_ad | 85.04 |
| arabertv2_large_reg_ad2 (lr 8e-6 refit) | 84.97 |
| camelbert_corn_ad | 84.65 |
| arbertv2_corn_ad | 84.17 |
| araelectra_corn_ad | 84.16 |
| marbert_corn_ad | 83.76 |
| aramodern_reg_ad | 82.30 |
| **arabertv2_large_reg_ad** | **0.0000, diverged** |

**These numbers are not comparable to the train-only family's**: for `_ad` models the calibration
split *is* the test holdout, which shifts the measurement. The honest estimate of the retraining
itself, under A/B, is **+0.31**, real but modest. `build_alldata.py` mirrors V6's member map and
weights onto `_ad` substitutes rather than re-selecting, exactly so the comparison is not
contaminated by a second selection pass.

## The diverged member

`arabertv2_large_reg_ad` collapsed to constant output: loss plateau ≈ 6.5 (11.57 → 6.81 → 6.68 →
6.53), dev QWK exactly 0.000 from epoch 0, MAE 3.62, degenerate thresholds. Diagnosis:
**bert-large with a scalar regression head diverges at lr 1.5e-5.** Retrained at **8e-6** as
`arabertv2_large_reg_ad2`, it converged cleanly to 84.97, the strongest all-data single model of the
base+large encoder set. `arabertv2_large_soft_ad` at the same learning rate showed a transient dip
(epoch 1 naive QWK 61.09) before recovering, corroborating that 1.5e-5 was marginal for this
backbone.

Two things about how it was discovered matter more than the bug:

1. **The dead member shipped.** It is a weighted member of both `blend_v7.json` and `final_v8.json`, i.e. inside submission 10. Joint threshold calibration absorbs a constant member, so it
   was harmless, and removing it was later measured **neutral**. But nothing in the pipeline noticed
   a member with QWK 0.000, which is why a member health check is the recommended guard.
2. It must be **reproduced as-is** for submission 10 to be bit-exact
   ([../../docs/reproduce.md](../../docs/reproduce.md)).

## Interruptions

A host reboot killed the first `_ad2` fix run at epoch 1 (healthy: 83.554);
`run_fix2.sh` relaunched with epochs cut 6 → 4 to bank the result sooner and cut exposure to another
reboot. `run_ad3.sh` relaxed the global GPU lock to share the device roughly 50/50 with
a sibling open-track project, its guard waits only on its own tag. The two projects shared the device
and nothing else: every model here is trained by `slra_st.train` on the BAREC parquet files listed in
[data/README.md](../../data/README.md), and no model, prediction or external corpus crossed over from
the open-track work ([../../docs/compliance.md](../../docs/compliance.md)). `araelectra_soft_ad`, planned
twice, was dropped and never trained.

The defense throughout: idempotent, restart-safe runners (skip if the score cache exists) and
detached execution.

## Files

| File | Role | Writes |
|---|---|---|
| `run_alldata.sh` | the 10 V6 members retrained on train+dev | - |
| `run_ad2.sh` / `run_ad3.sh` | pool expansion; `ad3` shares the GPU | - |
| `run_fix.sh` / `run_fix2.sh` | the lr 8e-6 divergence fix, relaunched after a reboot | - |
| `build_alldata.py` | mirror V6's map onto `_ad` members, calibrate on the holdout | `configs/ensembles/all_data.json` |

Logs: `artifacts/logs/run_alldata.log`, `run_ad2.log`, `run_ad3.log`, `run_fix*.log`.
