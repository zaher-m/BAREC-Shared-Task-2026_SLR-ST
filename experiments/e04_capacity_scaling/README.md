# E4. Capacity scaling: AraBERT-large and XLM-R

**When** Jul 26–27 · **Cost** 7 large runs + 1 abandoned · **Verdict** confirmed, the largest lever

## Hypothesis

`error_analysis.py` had just shown that the residual loss is concentrated in a **systematic**
under-prediction tail at level 12, true-12-predicted-7 alone is 7.5% of the QWK numerator on dev,
with true 12 filling nine of the twelve costliest confusion cells. A systematic bias is not
something averaging can remove (which is why [E5](../e05_seed_diversity/) failed), but it is
something a higher-capacity encoder might read differently.

## Results (calibrated QWK, dev / test)

| Model | bs / lr / ep | Dev | Test |
|---|---|---|---|
| arabertv2_large_corn | 24 / 1.5e-5 / 6 | 84.15 | 84.93 |
| arabertv2_large_soft | 24 / 1.5e-5 / 6 | 83.91 | 84.53 |
| arabertv2_large_wkl | 24 / 1.5e-5 / 6 | 84.10 | 83.94 |
| arabertv2_large_emd | 24 / 1.5e-5 / 6 | 83.81 | 83.54 |
| arabertv2_large_reg | 24 / 1.5e-5 / 6 | 83.21 | 83.49 |
| arabertv2_large_corn_raw | 24 / 1.5e-5 / 6 | 78.08 | 78.56 |
| arabertv2_large_reg_raw | 24 / 1.5e-5 / 6 | 77.39 | 76.97 |

Two results, not one:

**Capacity.** Adding the large family took blind QWK **84.6 → 85.0** with accuracy 35.9 → 38.8†, the largest single lever in the project, and the level-12 tail is what it fixed. Four of the ten
members that greedy selection kept for V6 are large models.

**Preprocessing, measured properly.** The two `_raw` runs are a controlled raw-vs-d3tok pair at
large scale: **d3tok is worth ≈ +6–7 QWK** (76.97–78.56 against 83.49–84.93 test). This is the
single strongest piece of evidence for morphological preprocessing in the project, and it is larger
than every modeling lever that came after it.

## XLM-R-large: not competitive

`xlmr_large_reg` (550M, raw, bs 16, lr 1e-5) trained healthily for 5 epochs (dev naive QWK
77.33 → 80.89 at epoch 3) at ~1,245 s/epoch, then was killed mid-epoch-5 when the GPU was reclaimed
for the all-data campaign; no checkpoint was saved and the queued `xlmr_large_corn` never started.
At ≈ −2 naive QWK against *base* AraBERTv2 for 5× the epoch cost, multilingual pretraining did not
transfer at parity. (The logs record no kill rationale; that reading is inferred from the numbers.)

## How far capacity goes

This experiment establishes the slope; [E12](../e12_post_deadline_ablations/) establishes the
ceiling. 135M → 370M was +0.4 blind QWK. 370M → a LoRA-tuned 7B was +0.045† wQWK, below the noise
floor. Returns to capacity collapse past a few hundred million parameters on this task.

## Files

| File | Role |
|---|---|
| `error_analysis.py` | the diagnostic that motivated this experiment (read-only) |
| `run_large.sh` | first attempt, bs 12; also banks the checkpoint in the cache first |
| `run_large_local.sh` | four objectives at bs 24 on the local GPU |
| `run_large2.sh` | the wkl head, the raw-variant ablation pair, and XLM-R |

Logs: `artifacts/logs/run_large*.log`, `train_arabertv2_large_*.log`, `train_xlmr_large_reg.log`.

† From the author's working notes, not recomputable here; see
[../../docs/reproducibility.md](../../docs/reproducibility.md#tier-3-not-verifiable-from-this-repository).
