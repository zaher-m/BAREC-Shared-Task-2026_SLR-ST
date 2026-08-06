# E12–E14. Post-deadline ablations

**When** Aug 4-6 · **Verdict** nothing adds information; the system is saturated at 85.4–85.5

With the submission phase closed, we answered three questions under rules we had fixed beforehand. Two overnight
campaigns cover all three, which is why they share one directory: `run_campaign3.sh` launched the
pseudo-label family and the folds, `run_campaign4.sh` reprioritized to put the 7B first and completed
the emd folds.

---

## E12. Pseudo-label distillation (`_ps`)

**Question:** how much of a 21-model ensemble fits in ONE base model, if the ensemble teaches it?

**Method:** transductive pseudo-labeling. The 8,077 blind sentences, labeled by the prior-calibrated
submission, join training (train + dev + pseudo = 70,232; test stays the holdout; 5 epochs). Two
students: `arabertv2_emd_ps` and `arabertv2_soft_ps`.

**Results:** the strongest single models of the entire project, naive holdout QWK 85.2–85.3 against
83.5–84.5 for conventional training, calibrated **86.02** (emd) and **86.23** (soft). Under the
fixed-anchor proxy, a single pseudo-distilled base model reaches **83.98† wQWK ≈ 98.2%† of the full
21-model system (85.48†) at 1/21 the inference cost**.

**But distillation compresses; it cannot exceed.** β-sweeping the student back into the teacher
ensemble hurt at every β > 0. Pseudo-labels re-inject the teacher's own information, correlated
noise, not new signal.

**And it produced the evaluation warning.** A pseudo-label student scored under a *self-estimated*
prior showed a phantom **89.1†** weighted QWK, impossible because it exceeds the test-fit oracle bound
of 87.98†. The student echoes its teacher's blind distribution, so the estimated prior is
self-referential. Full write-up:
[../../docs/evaluation.md](../../docs/evaluation.md#the-mistake-to-avoid-a-self-referential-prior).

---

## E13. ALLaM-7B LoRA

**Question (the capacity endpoint we had committed to):** does a 7B Arabic LLM, 52× the large encoder's
parameters, a different pretraining corpus, beat the 370M encoder?

**Configuration:** `ALLaM-AI/ALLaM-7B-Instruct-preview`, soft objective, d3tok, all-data regime,
attention-only LoRA r=16, bs 16, lr 1e-4, 2 epochs.

**Training:** epoch 0 loss 0.7338 / dev-naive QWK 83.294 (4,179 s); epoch 1 loss 0.4629 / **84.649**
(4,206 s); total 2 h 27 m 43 s. Calibrated test-holdout QWK **85.88**, third-best single model,
behind the two distilled students.

**System-level verdict:** single-model weighted QWK **83.94†**, and the best blend into the system
(β = 0.2) adds **+0.045† wQWK, below the ±0.1–0.2 noise floor**. By the noise-floor rule we fixed in e08,
we did not include it.

The striking comparison: a pseudo-distilled **135M** student (83.98†) *matches* the LoRA-tuned **7B**
(83.94†) at roughly 1/50 the inference cost. Caveats stated plainly: 2 epochs, attention-only LoRA
r=16, one objective. A fuller exploration might move the endpoint slightly, not by the ~1.0 QWK that
would change the conclusion.

---

## E14. Document-partitioned 5-fold (`_kf`)

**Question:** the 2-partition family (train-only ⊕ all-data) was worth +0.31. Does partition diversity
keep paying at 5?

**Method:** five document-disjoint folds over all 69,441 labeled sentences, deterministic md5
document hashing, emd objective, 5 epochs each. Folding by document matters: sentences from one
document must not straddle the split.

**Per-fold out-of-fold QWK:** kf0 85.35 (13,081 sentences) · kf1 83.81 (13,949) · kf2 83.49 (15,557) ·
kf3 84.97 (13,185) · kf4 85.20 (13,669). Pooled honest per-point OOF single-model **83.80† wQWK /
40.9† wAcc**, healthy models by any measure.

**System-level verdict:** blending the pooled OOF vector into the system **hurts at every β > 0**
(85.482 → 85.421 → 85.386)†. The 2-partition blend had already captured whatever partition diversity
has to give. Planned `soft` and `large_soft` fold families were never executed, `run_campaign4.sh`
prioritized the 7B, and the emd folds had already answered the question.

---

## The convergent conclusion

Four independent information sources, one protocol, the same answer:

| Source | Gain to the system |
|---|---|
| extra same-family members | +0.03 QWK per model† |
| pseudo-distilled students | negative at every β |
| a 52×-capacity Arabic LLM | +0.045 wQWK (noise)† |
| 5 document-partitioned folds | 0.00† |
| 15 post-processing levers | 0 or negative ([../../docs/negative-results.md](../../docs/negative-results.md)) |

That is what we mean by saturated, and it is why we read 85.4–85.5 as the ceiling for this label
set and this class of models, rather than as a limit of this particular system
([../../docs/findings.md](../../docs/findings.md#11-a-convergent-ceiling-at-854855)).

## Files

| File | Role |
|---|---|
| `run_campaign3.sh` | blind d3tok + pseudo-labels (step 0), the `_ps` family, the fold families |
| `run_campaign4.sh` | ALLaM-7B LoRA first, then the emd folds |

Both scripts end each family with a blind inference whose thresholds are the **uncalibrated default
grid**, those predictions (`artifacts/predictions/research/`) exist for their cached score matrices,
not for their labels. Logs: `artifacts/logs/run_campaign3.log`, `run_campaign4.log`,
`train_allam7b_soft_ad.log`, `train_arabertv2_*_ps.log`, `train_arabertv2_emd_kf*.log`.

† These numbers come from our working notes. The per-fold and per-model training numbers above *are* in
`artifacts/logs/`. See [../../docs/reproducibility.md](../../docs/reproducibility.md#tier-3-not-verifiable-from-this-repository).
