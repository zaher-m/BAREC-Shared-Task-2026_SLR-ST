# Experiments

One directory per experiment: the hypothesis, the code that ran it, and the verdict, including the
ones that failed. Numbering follows the chronology of the work, so gaps are meaningful: E0 and E2
produced no code of their own, and E13/E14 shared the two overnight campaigns in
[e12](e12_post_deadline_ablations/).


| ID | Dates | Experiment | Question | Verdict |
|---|---|---|---|---|
| E0 | Jul 23 | Bootstrap | Can a submission be produced at all, in the right format? | Fallback public-test prediction (84.05) and a format dry-run banked within hours → `artifacts/predictions/bootstrap/` |
| [E1](e01_backbone_objective_matrix/) | Jul 23–24 | Backbone × objective matrix | Which Arabic encoder, which text variant, upsample or not? | **AraBERTv2 + d3tok** is the anchor (~84 test QWK); AraModernBERT prefers raw; upsampling dropped |
| E2 | Jul 24 | First blind submissions | Does the pipeline transfer to blind? | 84.1 → 84.5, then a −0.1 dip at 7 members. See [../submissions/](../submissions/) |
| [E3](e03_objective_diversity/) | Jul 26 | Objective diversity | Do five loss geometries on one backbone decorrelate better than five backbones? | **Confirmed.** Best train-only single model 85.55; blind 84.5 → 84.6 |
| [E4](e04_capacity_scaling/) | Jul 26–27 | Capacity: large + XLM-R | Does capacity attack the systematic error tail? | **Confirmed, +0.4 blind** (84.6 → 85.0), the largest single lever. XLM-R not competitive |
| [E5](e05_seed_diversity/) | Jul 26 | Seed diversity | Do seed replicas add usable diversity? | **Refuted.** The tail is bias, not variance; campaign aborted mid-family and redirected |
| [E6](e06_greedy_selection/) | Jul 24–27 | Greedy ensemble selection | Which members, which weights? | V4 → V5 → **V6** (10 members, test 86.398, blind 85.0) |
| [E7](e07_all_data_retraining/) | Jul 27–28 | All-data retraining | Is train+dev genuine signal, and does it decorrelate? | **+0.31** honest A/B; one member diverged to constant output |
| [E8](e08_combiner_bakeoff/) | Aug 2 | Combiner bake-off | Greedy, learned weights, robust aggregation, or a fixed blend? | **Fixed 50/50 two-regime blend scores highest** (86.936 ± 0.345); four sibling ideas refuted |
| [E9](e09_endgame_gate/) | Aug 2 | Validated enlargement | Ship a bigger blend only if it actually improves on untouched data | +0.010 → shipped as V8 (blind 85.3 / 35.9) |
| [E10](e10_accuracy_frontier/) | Aug 2 | Accuracy frontier | How much accuracy is free at fixed QWK? | Acc 35.9 → 37.9 → 40.4 at **flat** blind QWK; proved the shift was real |
| [E11](e11_target_prior_calibration/) | Aug 2–3 | Target-prior calibration | Can the label shift be exploited ? | **The largest measured gain**: blind 85.4 / 38.7. Iterating it regresses to 85.2 |
| [E12](e12_post_deadline_ablations/) | Aug 3–4 | Distillation, ALLaM-7B, 5-fold | What else could add information? | Nothing does. 98.2% compression, +0.045 from a 7B, 0.00 from 5 folds |
| E15 | Aug 2–4 | Negative-results catalog | - | 15 refuted levers: [../docs/negative-results.md](../docs/negative-results.md) |
| E16 | Aug 4 | Minimality and information accounting | How little of the system was needed? | k=1: 82.97 · k=3: 83.61 · k=5: 84.00 · k=8: 84.32 · 21: 85.48 |

## Where the numbers live

- Per-model results: [../docs/results.md](../docs/results.md), from `artifacts/metadata/` and
  `artifacts/logs/train_*.log`.
- Ensemble compositions and fitted thresholds: [../configs/](../configs/).
- Submission record with scores: [../submissions/README.md](../submissions/README.md).
- Campaign logs, including the only run log containing ensemble metrics
  (`artifacts/logs/run_endgame.log`).

## Reproducing an experiment

```bash
export PYTHONPATH=src        # or: pip install -e .
.venv/bin/python -m slra_st.verify                                   # check the reported numbers
.venv/bin/python experiments/e08_combiner_bakeoff/qwk_analysis.py    # then any experiment
```
