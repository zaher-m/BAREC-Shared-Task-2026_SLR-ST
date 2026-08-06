# E6. Greedy ensemble selection

**When** Jul 24 – Jul 27 · **Verdict** V6, 10 members, test QWK 86.398, blind 85.0

## Question

Given a growing pool of members, which subset and which weights? Exhaustive subset search is
hopeless and overfits; uniform averaging ignores member quality.

## Method

Caruana-style greedy forward selection **with replacement** on calibrated dev QWK: start from the
best single member, then repeatedly add whichever member most improves the ensemble, allowing repeats
so that multiplicity becomes an integer weight. 25 rounds. Threshold fitting is coarse during the
search (3 rounds, grid 0.1) and fine for the final pick (8 rounds, grid 0.02), the search needs
rankings, not precision.

`ensemble_baseline.py` is the baseline it had to beat: uniform and dev-QWK-weighted averages over the
whole pool.

## Results

| Config | Members | Weight sum | Test QWK | Blind |
|---|---|---|---|---|
| [V4](../../configs/ensembles/v4.json) | 7 | 12 | - | 84.4 |
| [V5](../../configs/ensembles/v5.json) | 11 | 25 | - | not submitted |
| [**V6**](../../configs/ensembles/v6.json) | 10 | 13 | **86.398** | **85.0** |

V6: `arabertv2_corn_s2`:1 · `arabertv2_emd_d3`:1 · `arabertv2_large_corn`:1 ·
`arabertv2_large_reg`:1 · `arabertv2_large_soft`:2 · `arabertv2_soft_d3`:1 · `arabertv2_wkl_d3`:1 ·
`araelectra_corn_d3`:2 · `aramodern_reg_raw`:2 · `arbertv2_corn_d3`:1

## What V6's composition says

V6 is the verdict on the two experiments before it, reached by an automatic procedure rather than by
argument:

- **four large members**, capacity ([E4](../e04_capacity_scaling/)) earns its GPU hours;
- **five objective variants of AraBERTv2**, objective diversity ([E3](../e03_objective_diversity/))
  beats backbone diversity;
- **only three members from other backbones** survive at all.

V4 is a useful negative data point too: it had more members than the 6-member set
submitted before it and scored **0.1 lower** on blind (84.4 vs 84.5). More members is not monotone
improvement, which is the first sign of what E8 later measured properly.

## The important caveat, added later

On the enlarged 49-model pool, **this same procedure overfits the selection split**: greedy
re-selection scored 86.60 honest holdout against 86.94 for a fixed 50/50 two-regime blend
([E8](../e08_combiner_bakeoff/)). Greedy selection was the right tool at 24 members and the wrong
tool at 49. The switch was not a matter of taste, it was measured.

## Files

| File | Role | Writes |
|---|---|---|
| `ensemble_select.py` | greedy forward selection with replacement | ensemble config + test prediction with `--write` |
| `ensemble_baseline.py` | uniform / QWK-weighted baseline (superseded) | `configs/ensembles/baseline.json` |

Logs: `artifacts/logs/infer_blind_v3.log`, `infer_blend*.log`, `infer_new3.log`.
