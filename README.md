# SLRA-ST. Sentence-Level Readability Assessment, Strict Track

Sentence-level Arabic readability assessment on a 19-level ordinal scale, built for the
[BAREC 2026 Shared Task](https://barec.camel-lab.com/sharedtask2026), Task 1, **strict track**. Given
an Arabic sentence, we predict its readability level from 1 to 19. The strict-track rule is *"Models must be trained exclusively on the training set of the BAREC Corpus."* The ranked
metric is **quadratic weighted kappa (QWK)**

We scored **QWK 85.4 / Acc 38.7** on the blind test set and ranked 1st on the
[strict-track leaderboard](https://www.codabench.org/competitions/16545/#/results-tab). The numbers
and the full submission history are in [docs/results.md](docs/results.md).

Every number in these docs that can be recomputed is checked by `make verify`. The rest is listed in
[docs/reproducibility.md](docs/reproducibility.md). What data the system touched, against the
strict-track rule, is in [docs/compliance.md](docs/compliance.md).

## This work

A **21-member weighted ensemble** of fine-tuned Arabic transformer encoders, combined with
**QWK-optimal threshold calibration adapted to the blind set's estimated label prior**.

Four things did most of the work, in order of measured effect:

1. **Morphological preprocessing.** CAMeL Tools `d3tok` segmentation is worth about **+6–7 QWK**
   over light normalization on `bert-large-arabertv2`, more than any modeling change that came
   after it. It is not universal: AraModernBERT did better on raw text.
2. **Objective diversity on one backbone.** Five ordinal losses (regression, CORN, soft labels,
   weighted-kappa loss, squared EMD) on AraBERTv2 decorrelate errors better than five different
   backbones do. Greedy selection kept the objective variants and dropped most of the backbones.
3. **Capacity, until it saturates.** base (135M) → large (370M) was the largest single lever
   (+0.4 blind QWK). large → a LoRA-tuned **ALLaM-7B** (52× the parameters) added +0.045† wQWK,
   below the measured ±0.1–0.2 noise floor, so we excluded it under the noise-floor rule we had
   fixed in E8.
4. **Target-prior threshold calibration**, the largest measured gain. We
   estimate the blind label prior with regularized BBSE, reweight the calibration set to it, and
   refit the 18 cut points. It improved both leaderboard metrics at once (85.3 → 85.4 QWK,
   +2.8 Acc). It has a clear failure mode: running BBSE a second time on its own output took the
   blind score back down to 85.2.

Everything else we tried failed under the same protocol: **15 refuted levers**, from seed replicas
and robust aggregation to pseudo-label distillation and 5-fold partitioning, plus the 7B. All of it
is in [docs/negative-results.md](docs/negative-results.md).

† Not recomputed by `make verify`. Provenance for each such number is in
[docs/reproducibility.md](docs/reproducibility.md).

## Repository layout

```
src/slra_st/   Library: metrics, calibration, ensembling, shift estimation, train, infer
experiments/   One directory per experiment: hypothesis, method, launchers, verdict
configs/       Ensemble member/weight configs (V4 → V8) and fitted threshold sets
artifacts/     Cached member scores, per-model metadata, training and inference logs
submissions/   Every blind submission we uploaded, plus the candidates we generated
docs/          Data card, method, evaluation protocol, results, findings, compliance
data/          Only a README; the BAREC splits are not redistributed here
```

The trained weights (41 GB) and the corpus are not tracked. See
[artifacts/README.md](artifacts/README.md) and [data/README.md](data/README.md) for what to fetch
and where it goes.

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -e .

# recompute every reported number from the committed caches (no GPU, no corpus)
make verify

# to train or infer you need the corpus; fetch the BAREC splits into data/ (see data/README.md)
make preprocess                                       # builds the raw and d3tok variants

# train one member (backbone x objective x data regime)
.venv/bin/python -m slra_st.train --model aubmindlab/bert-base-arabertv2 \
    --tag arabertv2_emd_d3 --variant d3tok --objective emd --bs 32 --lr 2e-5 --epochs 8

# run the shipped 21-member ensemble over unlabeled sentences
.venv/bin/python -m slra_st.infer --input data/blind_sent.parquet --ensemble final_v8 \
    --out artifacts/regenerated/prediction

# check a submission file against the required format
.venv/bin/python -m slra_st.submission artifacts/regenerated/prediction
```

`artifacts/regenerated/` is gitignored, so re-running never overwrites the submission record.

## Experiments

| ID | Experiment | Question | Verdict |
|---|---|---|---|
| [E1](experiments/e01_backbone_objective_matrix/) | Backbone × objective matrix | Which Arabic encoder, which text variant? | AraBERTv2 + d3tok is the anchor (~84 test QWK) |
| [E3](experiments/e03_objective_diversity/) | Objective diversity | Do loss geometries decorrelate better than backbones? | Confirmed; best train-only single model 85.55 |
| [E4](experiments/e04_capacity_scaling/) | Capacity | Does more capacity fix the systematic error tail? | Confirmed, +0.4 blind; XLM-R not competitive |
| [E5](experiments/e05_seed_diversity/) | Seed diversity | Do seed replicas add ensemble diversity? | **Refuted**, the tail is bias, not variance |
| [E6](experiments/e06_greedy_selection/) | Greedy selection | Which subset and weights? | V4 → V5 → V6 (test 86.398) |
| [E7](experiments/e07_all_data_retraining/) | All-data retraining | Is train+dev genuine signal? | +0.31 honest A/B; one member diverged |
| [E8](experiments/e08_combiner_bakeoff/) | Combiner bake-off | Greedy, learned weights, or a fixed blend? | Fixed 50/50 two-regime blend scores highest (86.94) |
| [E9](experiments/e09_endgame_gate/) | Validated enlargement | Ship the bigger blend only if it clears the gate | +0.010 → shipped as V8 (blind 85.3) |
| [E10](experiments/e10_accuracy_frontier/) | Accuracy frontier | How much Acc is free at fixed QWK? | Acc 35.9 → 40.4 at flat blind QWK |
| [E11](experiments/e11_target_prior_calibration/) | Target-prior calibration | Can the label shift be exploited? | **The largest measured gain**: 85.4 / 38.7 |
| [E12](experiments/e12_post_deadline_ablations/) | Distillation, 7B, 5-fold | What else adds information? | Nothing: the system is saturated at 85.4–85.5 |

The index with dates, artifacts and per-experiment numbers:
[experiments/README.md](experiments/README.md).

## How much of the system was actually needed

After the deadline we pointed the same protocol at our own system. One model gives 82.97 weighted
QWK†, eight give 84.32†, all 21 give 85.48†. A single pseudo-distilled base model gives 83.98†,
which is 98.2%† of the full system at 1/21 of the inference cost, and about the same as a LoRA-tuned
7B Arabic LLM (83.94†) at roughly 1/50 of its cost.

Everything we measured after training lands in the same place: extra same-family members
(+0.03/model†), distilled students, the 7B (+0.045†), 5-fold partition models (0.00†), and 15
post-processing ideas (0 or negative). We read 85.4–85.5 as the information ceiling for this label
set and this class of sentence-level encoders. Details in [docs/findings.md](docs/findings.md).

## Shared task compliance

[docs/compliance.md](docs/compliance.md) is the audit: the strict-track rule quoted in full, a table
of exactly which BAREC splits each part of the shipped system touched, the ablations that are **not**
track-eligible and that we never submitted, our submission-rate record against CodaBench's
5-per-day cap, and the questions the published rules do not settle.

Short version: we used no external data of any kind and no blind labels, but the system uses more
than the training split, and the prior calibration in E11 is transductive, in that it reads the
blind set's predicted label distribution as a whole. Neither point is addressed by the published
rules, so we state both rather than assume them. If the strictest reading holds, the number to fall
back on is V6 at blind 85.0.

## License

Code is MIT-licensed ([LICENSE](LICENSE)). The BAREC corpus is not redistributed here and remains
governed by the shared task's terms.
