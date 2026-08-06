# E5. Seed diversity

**When** Jul 26 · **Cost** 3 saved runs + 1 killed, campaign aborted · **Verdict** REFUTED

## Hypothesis

Seed replicas of the proven core add ensemble diversity for free: no new backbone to download, no
new objective to implement, just the same recipe at seed 101 and 202. This is standard practice in
Kaggle-style ensembling, so it was phase 1 of a ~24-model campaign.

## Results (calibrated QWK, dev / test)

| Model | Dev | Test |
|---|---|---|
| arabertv2_soft_s2 | 84.05 | 84.65 |
| arabertv2_reg_s2 | 84.03 | 84.19 |
| arabertv2_corn_s2 | 83.71 | 84.46 |
| arabertv2_emd_s2 | - | killed at epoch 4, no checkpoint |

Individually healthy, within noise of their seed-42 twins, exactly as expected. The problem is what
they contribute to the ensemble: **~0**.

## Why it fails here

The residual error is a **systematic** under-prediction tail at level 12
([E4](../e04_capacity_scaling/)), not variance around a correct mean. Averaging N runs of the same
recipe cancels variance; it cannot cancel bias, because every replica has the same bias. The
campaign's own header records the pivot: *"seeds proved neutral b/c the L12 tail is a SYSTEMATIC
error, not random."*

This diagnosis is what made the next two bets the right ones, capacity (which changes what the
model can represent) and objective diversity (which changes where it is wrong), and it is the same
reason robust aggregation failed later ([E8](../e08_combiner_bakeoff/)): there is no rogue member to
reject when all members are wrong together.

## What happened to the budget

Phases 2–5 of `run_campaign.sh` (a seed-202 family, more cross-backbone seeds, more large models)
were abandoned. `run_campaign2.sh`, launched the same evening, spent the same GPU hours on
decorrelated members instead: `marbert_emd_d3`, `camelbert_emd_d3`, `camelbert_wkl_d3`,
`araelectra_emd_d3`, `araelectra_wkl_d3`, `arbertv2_corn_d3`, `arbertv2_soft_d3`,
`aramodern_soft_raw`, then the four AraBERT-large models.

One seed replica did survive into the shipped system: `arabertv2_corn_s2` is a V6 member. Greedy
selection is free to pick a replica when it happens to help, the refuted claim is that replicas are
a *reliable* source of diversity, not that any individual one is useless.

## Files

| File | Role |
|---|---|
| `run_campaign.sh` | the 5-phase campaign; phase 1 ran, phases 2–5 abandoned |
| `run_campaign2.sh` | the pivot: decorrelated members instead of replicas |

Logs: `artifacts/logs/run_campaign.log`, `run_campaign2.log`.
