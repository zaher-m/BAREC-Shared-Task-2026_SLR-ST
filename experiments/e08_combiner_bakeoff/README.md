# E8. Combiner bake-off

**When** Aug 2, entirely GPU-free · **Verdict** a fixed 50/50 two-regime blend beats everything fitted

## Setting

After a five-day pause, research resumed on the cached score matrices alone, no training, no
inference. Test (7,286 points) is the only split clean for both model families, so it hosts every
comparison, under the honest A/B protocol
([../../docs/evaluation.md](../../docs/evaluation.md#the-honest-ab-protocol)): fit everything that is
fitted on one half, score the untouched half, both directions, several permutations.

What is being estimated is how well a **procedure** generalizes, not how well one fitted ensemble
scores. That is the whole experiment: greedy selection looked best on the data it selected on.

## The bake-off

| Scheme | Honest holdout QWK |
|---|---|
| **fixed 0.5·V6 + 0.5·all-data** | **86.936 ± 0.345** |
| all-data half alone | ~86.79 |
| V6 fixed | ~86.60 |
| greedy re-selection over all 49 | ~86.60 (overfits the selection split) |
| uniform average over all 49 | ~86.53 (underfits) |

Selection-free schemes and a blend-weight sweep (`fixed_schemes.py`) confirm it: b = 0.5 is at or
near the top of the b·V6 + (1−b)·AD sweep, and top-K-by-individual-quality subsets do not catch it.

Why the fixed choice scores highest: with 49 correlated candidates, anything that *fits* which members to use
spends its freedom on the noise in the selection split. Giving equal mass to the two data regimes
fits nothing at all. It just encodes the one thing we already knew from
[E7](../e07_all_data_retraining/), that the two families make decorrelated errors, and that
generalizes.

`build_blend.py` then flattens the blend into a single 20-member weight vector, which is exact
because both source ensembles have equal weight sums.

## Four related ideas

| Lever | Result |
|---|---|
| **Robust aggregation** (median, trimmed 10/20/30%, disagreement shrinkage, mean-median hedge) | all at or below the weighted mean. The tail diagnostic says why: the median's \|err\|≥3 rate is 11.13% against the mean's 10.58%, the large errors are **all-members-wrong**, not one-member-rogue, so there is no outlier to reject |
| **Ridge / simplex weight fitting** (NNLS on the simplex, λ ∈ {0.001…1.0}, uniform or blend prior) | 86.665–86.677 against the blend's 86.936, at every λ: ill-conditioned on correlated members |
| **Bagged thresholds** (8 bootstrap resamples averaged) | no gain, the QWK optimum in threshold space is sharp, and averaging smooths across it |
| **3-way structured blends** (add a large-only or V5 third group) | every variant below the 2-way blend: the third group re-correlates with an existing half |

## The QWK identity

`qwk_analysis.py` works through `QWK = 1 − MSE / (Var_true + Var_pred + Δμ²)` and sweeps dispersion
expansion as a lever. The lever does not work: the measured optimum is `a*` ≈ 1.00, so the system is
already at the dispersion optimum. The identity was still the most useful thing to come out of this
directory. It is what later explained the test→blind drop as denominator shrinkage instead of model
failure, and it is why the label-shift work in [E11](../e11_target_prior_calibration/) was
predictable rather than a guess.

## Reproducing this

`combine_opt.py`, `fixed_schemes.py`, `ridge_weights.py` and the `large` group in
`bagging_and_3way_blends.py` build their pool by globbing every cached member. There were 49 at the
time and 56 usable today, so the pool-wide rows move: `combine_opt.py` now reports 86.947 for the
blend and 86.845 for greedy(all), against 86.94 and 86.60 then. The ranking, and so the conclusion,
is unchanged. The config-driven rows (V6, AD, the 50/50 blend) are stable and are checked by
`make verify`.

One thing to know about that larger pool: the seven members added since Aug 2 include the two
pseudo-label students (`arabertv2_emd_ps`, `arabertv2_soft_ps`), which trained on the blind sentences
and so would not be eligible for the strict track
([../../docs/compliance.md](../../docs/compliance.md)). They were not on disk when this experiment
ran, and nothing they touch was ever submitted, but a re-run today is selecting over a pool that
includes them. Pass `exclude=("_ps",)` to `load_pool` to reproduce the in-track pool. The 5-fold
models need no exclusion: their cached block is their out-of-fold partition, not the test split, so
`load_pool("test")` drops them automatically.

## Files

| File | Role | Writes |
|---|---|---|
| `combine_opt.py` | the five-scheme bake-off, greedy included | `artifacts/diagnostics/combine_diag.json` |
| `fixed_schemes.py` | selection-free schemes + blend-weight sweep | - |
| `build_blend.py` | flatten 0.5·V6 + 0.5·AD into one config | `configs/ensembles/blend_v7.json` |
| `robust_agg.py` | median / trimmed / disagreement shrinkage, refuted | - |
| `ridge_weights.py` | ridge and simplex weight fitting, refuted | - |
| `bagging_and_3way_blends.py` | bagged thresholds and 3-way blends, both refuted | - |
| `qwk_analysis.py` | the QWK identity, variance decomposition, dispersion sweep | - |

Logs: `artifacts/logs/run_obj.log`, `run_endgame.log` (the only run log carrying ensemble metrics).
