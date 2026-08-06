# Artifacts

Everything the pipeline produced.

```
scores/members/     61 x scores_<tag>.npz   per-model continuous dev/test scores   TRACKED
scores/blind/        5 x <name>_scores.npz  per-member blind score matrices        TRACKED
metadata/           61 x meta_<tag>.json    backbone, objective, regime, thresholds TRACKED
logs/               98 log files            training, campaign, inference logs      TRACKED
predictions/        research + bootstrap prediction files                          TRACKED
diagnostics/        combine_diag.json       greedy selection-count diagnostics      TRACKED
models/             61 model directories, 41 GB                                    NOT tracked
```

## scores/

Training saved every model's continuous, pre-threshold scores for its dev and test blocks, and every blind inference saved its full member × sentence
matrix. Two consequences:

- All ensembling, calibration, minimality and ablation work in [../experiments/](../experiments/) runs
  with **no GPU and no corpus**, from these `.npz` files alone.
- Every submission after the 21-member inference was a **re-thresholding** of
  `blind/prediction_final_scores.npz`, which is why the accuracy frontier and the prior calibration
  could be explored in minutes on the night of the deadline.

Two schemas:

| Schema | Files | Contents |
|---|---|---|
| per-model | `members/scores_<tag>.npz` (61) | `dev_ids, dev_scores, dev_labels, test_ids, test_scores, test_labels` |
| ensemble blind | `blind/<name>_scores.npz` (5) | `ids` (8,077), `members`, `weights`, `member_scores` (8,077 × M), `ens_score` |

For the all-data, pseudo-label and 5-fold regimes the `dev` and `test` blocks are the **same holdout**;
for `_kf` they are that fold's out-of-fold block (13,081–15,557 rows). See
[../docs/method.md](../docs/method.md#data-regimes) before comparing numbers across regimes.

`blind/prediction_final_scores.npz` holds M=21, the shipped system. The other four
(`new3`, `blind_psfam`, `blind_allam`, `blind_kfemd`) are the post-deadline research inferences.

## metadata/

`meta_<tag>.json` is byte-identical to `models/<tag>/meta.json` and holds backbone, text variant,
objective, LoRA flag, `max_len`, tag, calibrated QWK, and the model's own 18 fitted thresholds. It is
tracked exactly because `models/` is not: the record of what each model was survives the weights.

Note what these files do **not** contain: epochs, learning rate, batch size, seed. Those live in the
launch scripts and the training logs, and are consolidated in
[../docs/results.md](../docs/results.md).

`train_label_counts.json` is here too: the label counts of the training split, so the analysis
scripts that need the training prior run without the corpus. Aggregate counts only, no text.

## logs/

- 63 `train_<tag>.log`, per-epoch loss, dev naive QWK, epoch seconds, and the final calibrated
  DEV/TEST metric lines. 61 completed runs plus 2 that were killed before saving, which is why there
  are 63 logs and 61 models.
- 20 campaign logs (`run_*.log`). `run_endgame.log` is the only one carrying ensemble-level metrics.
- 10 inference logs, plus `preprocess_d3tok.log`, `medvote.log` and two download logs.

Two inference runs have no log at all (the producer of the first scored submission, and the V6
rewrite); one submission's ensemble membership is consequently unknown. Flagged rather than papered
over, see [../submissions/README.md](../submissions/README.md).

## predictions/

| Directory | Contents |
|---|---|
| `research/` | the four post-deadline inference sets (`new3_blind`, `blind_psfam`, `blind_allam`, `blind_kfemd`). Thresholded with the **uncalibrated default grid**, they exist for their score matrices, not their labels |
| `bootstrap/` | the E0 insurance artifacts: `FALLBACK_3reg_upsampled_test84.05.*` (a public-test prediction from the first three regression models, banked hours into the project) and `mock_blind_*` (a 300-row dry run of the required `prediction`-inside-`prediction.zip` format) |

Submitted files and never-submitted candidates live in [../submissions/](../submissions/), not here.

## models/, not tracked

61 directories, 41 GB: `model.pt` (~540 MB base, ~1.48 GB large, 67 MB for the LoRA adapter+head),
`meta.json`, and tokenizer files. Regenerable from the launchers in
[../experiments/](../experiments/); see [../docs/reproduce.md](../docs/reproduce.md) for the cost.

`infer.py` reads `models/<tag>/meta.json` and `models/<tag>/model.pt`, so blind inference needs the
checkpoints present. Everything else in this repository does not.
