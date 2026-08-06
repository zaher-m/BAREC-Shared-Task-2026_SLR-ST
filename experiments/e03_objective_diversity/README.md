# E3. Objective diversity on one backbone

**When** Jul 26 · **Cost** 7 training runs + one throughput experiment · **Verdict** confirmed

## Hypothesis

Loss-geometry diversity on one strong backbone decorrelates errors better than backbone diversity
does at equal single-model quality. Concretely: add soft labels (SORD), weighted-kappa loss and
squared EMD to the existing regression and CORN heads, mostly on AraBERTv2.

The reasoning: these models all see the same text through the same encoder, but a squared-EMD head
and a scalar-regression head are wrong in different places. The first cares about the shape of the
predicted distribution, the second only about its mean.

## Results (calibrated QWK, dev / test)

| Model | Dev | Test |
|---|---|---|
| **arabertv2_emd_d3** | 84.42 | **85.55** |
| arabertv2_wkl_d3 | 84.26 | 84.85 |
| arabertv2_soft_d3 | 84.12 | 84.89 |
| araelectra_soft_d3 | 83.31 | 83.75 |
| marbert_soft_d3 | 82.93 | 83.10 |
| camelbert_soft_d3 | 82.89 | 84.09 |
| marbert_wkl_d3 | 82.56 | 83.28 |

`arabertv2_emd_d3` became the best train-only single model of the project (85.55 test calibrated).
A blind submission 14 minutes after the family finished scored **84.6**, up from the 84.5 baseline.

The decisive evidence came later, from [E6](../e06_greedy_selection/): given a pool containing both
kinds of diversity, greedy selection kept **five objective variants of AraBERTv2** and only three
members from other backbones.

## Side experiment: parallel vs sequential training

`run_par.sh` trained 3–4 models concurrently on the one GPU, on the theory that free VRAM was being
wasted. It was **abandoned within ~43 minutes**: an epoch that overlapped three co-tenants took
**1,273 s against a 298 s solo baseline, 4.2× slower per job**, for zero net throughput. The device
is compute-bound; free VRAM is a red herring.

`run_seq.sh` is the sequential replacement, with dynamic per-batch padding in the trainer instead, which *did* pay: 300 → 213 s/epoch on soft, 265 → 209 s on regression, and exact under masked mean
pooling. `run_seq2.sh` then pushed batch size 32 → 64 (lr 2e-5 → 2.5e-5), since at these sequence
lengths bigger batches mean bigger, more efficient matmuls and half the step count.

The failed script is kept as the record of an engineering idea that looked free and was not.

## Files

| File | Role |
|---|---|
| `run_obj.sh` | the objective-diversity family |
| `run_par.sh` | concurrent training, **refuted**, kept as the record |
| `run_seq.sh` | sequential replacement with dynamic padding |
| `run_seq2.sh` | continuation at bs 64 |

Logs: `artifacts/logs/run_obj.log`, `run_par.log`, `run_seq*.log`.
