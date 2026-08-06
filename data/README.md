# Data

**Nothing in this directory is tracked.** The BAREC corpus is not redistributed here: it is
distributed by CAMeL Lab and governed by the shared task's terms. This file documents what to place
here to make the pipeline run. The full data card, sizes, label distribution, preprocessing, caveats, is [../docs/data.md](../docs/data.md).

## What to obtain

From the official release (HuggingFace collection `CAMeL-Lab/barec-shared-task-2026`):

```
data/barec_sent_train.parquet          54,845 sentences, 1,518 documents
data/barec_sent_validation.parquet      7,310 sentences,   194 documents
data/barec_sent_test.parquet            7,286 sentences,   210 documents
data/blind_sent.parquet                 8,077 sentences, unlabeled (registered participants only)
```

The blind set was released to registered participants during the testing phase; it carries only
`Sentence ID` and `Sentence`.

## What the pipeline builds

```bash
make preprocess      # or: python -m slra_st.preprocess --variant {raw,d3tok}
```

```
data/proc_raw_{train,validation,test}.parquet      light normalization only
data/proc_d3tok_{train,validation,test}.parquet    + CAMeL d3tok morphological segmentation
```

Each has exactly `[ID, text, label19]`. d3tok takes ~3 minutes for all 69,441 sentences and needs the
CAMeL Tools `calima-msa-r13` model (`camel_data -i disambig-mle-calima-msa-r13`).

There is  **no `proc_raw_blind.parquet`**: the raw path applies light normalization on the
fly during inference, and only the expensive d3tok variant was materialized.

## Careful with `proc_d3tok_blind.parquet`

That file's `label19` column holds **pseudo-labels**. It was built for one post-deadline ablation
([../experiments/e12](../experiments/e12_post_deadline_ablations/)) by the `step 0` block of
`run_campaign3.sh`, not by `preprocess.py`.

## Strict-track constraint

Training data is **BAREC only**: no external data. Which BAREC
splits each model regime uses is a separate question, and the rule text names only the training set,
so the per-component breakdown is in [../docs/compliance.md](../docs/compliance.md). The blind
sentences under the system's own pseudo-labels, in the ablation above, **are post-deadline and not part
of the submitted system**.
