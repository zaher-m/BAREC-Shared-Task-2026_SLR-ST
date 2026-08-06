# E1. Backbone × objective matrix

**When** Jul 23–24 · **Cost** 15 training runs · **Verdict** AraBERTv2 + d3tok is the anchor

## Question

Which Arabic encoder should carry this task, with which text variant, and do two design choices
that look free actually help, class upsampling, and morphological segmentation for every backbone?

## Method

Six base backbones × {regression, CORN}, one at a time on one GPU. `run_matrix.sh` used √-damped
inverse-frequency upsampling; `run_matrix2.sh` dropped it and assigned AraModernBERT to raw text;
`run_matrix3.sh` added AraELECTRA and ARBERTv2 and filled in the missing CORN heads.

bs 48 (matrix, matrix2) or 32 (matrix3), lr 2e-5, 7 epochs, patience 2, seed 42.

## Results (calibrated QWK, dev / test)

| Model | Dev | Test |
|---|---|---|
| **arabertv2_corn_d3** | **83.77** | **84.09** |
| **arabertv2_reg_d3** | **83.73** | **84.10** |
| arbertv2_reg_d3 | 82.77 | 82.79 |
| camelbert_corn_d3 | 82.68 | 83.29 |
| araelectra_corn_d3 | 82.59 | 83.31 |
| camelbert_reg_d3n (no upsample) | 82.55 | 83.09 |
| marbert_corn_d3 | 82.11 | 82.60 |
| camelbert_reg_d3 (upsampled) | 81.76 | 82.60 |
| marbert_reg_d3 (max_len 192) | 81.76 | 81.75 |
| marbert_reg_d3n (no upsample) | 81.21 | 81.85 |
| aramodern_corn_raw | 80.92 | 81.53 |
| araelectra_reg_d3 | 80.80 | 81.71 |
| aramodern_reg_raw | 80.38 | 81.24 |
| aramodern_reg_d3 (upsampled) | 80.20 | 79.39 |

`aramodern_corn_d3` was queued in `run_matrix.sh` and never started, the campaign was superseded.

## Conclusions

1. **AraBERTv2-base + d3tok is the strongest base configuration** (~84 test QWK) and became the
   workhorse backbone for everything after this.
2. **Upsampling: mixed, so dropped.** camelbert improved without it (82.55 vs 81.76 dev), marbert
   slightly regressed (81.21 vs 81.76). This is a weak negative, dropped for simplicity, not
   proven harmful.
3. **AraModernBERT prefers raw text** (81.24 vs 79.39 test on regression): its own tokenizer suits
   unsegmented input. All later `aramodern` models used raw. Preprocessing is a per-backbone choice,
   not a global one.
4. **CORN ≥ regression on most backbones**, which is the first hint that the loss geometry matters
   independently of the encoder, the thread picked up in [E3](../e03_objective_diversity/).

## Files

| File | Role |
|---|---|
| `run_matrix.sh` | first pass, with `--upsample` |
| `run_matrix2.sh` | no upsampling, per-backbone variant assignment |
| `run_matrix3.sh` | AraELECTRA + ARBERTv2, remaining CORN heads |

Logs: `artifacts/logs/run_matrix*.log`, `artifacts/logs/train_<tag>.log`.
