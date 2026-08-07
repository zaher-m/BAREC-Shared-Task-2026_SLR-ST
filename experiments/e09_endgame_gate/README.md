# E9. Validated enlargement: the endgame gate

**When** Aug 2  **Verdict** candidate cleared the gate by +0.010 → shipped as V8 (blind 85.3)

## The problem

Members were still finishing training as the deadline approached. The tempting move, ship the
biggest ensemble available, is exactly the move that produced the −0.1 dip at
[V4](../e06_greedy_selection/). More members is not monotone improvement.

So the decision was automated and gated: enlarge the all-data half with every `_ad` model that
finished, and ship the result **only if it beats the incumbent on untouched data**.

## Method

- **AD half** = the 10 mapped members at V6 weights, plus any newly trained `_ad` model at weight 1.
  Uniform for the newcomers, because [E8](../e08_combiner_bakeoff/) had just shown greedy selection
  overfits at this pool size.
- **Blend** = 0.5·V6 + 0.5·AD, the structural choice that beat every data-driven weighting.
- **Gate** = honest A/B on clean test against the currently shipped blend. No improvement, no write.

`run_endgame.sh` wraps it as an unattended sequence: wait for the last `_ad` model (hard cutoff at
22:15), stop this project's training so inference gets the GPU, matching kills on this project's own
tags, leaving a sibling project's run alone, run the gate, then infer and machine-validate the
submission file.

## Recorded outcome

Quoted verbatim from `artifacts/logs/run_endgame.log` (2026-08-02 22:03), which is why the wording
here is the script's original wording rather than the vocabulary used elsewhere in these docs:

```
CURRENT blend (20 members):  holdout 86.936 +- 0.345
AD half alone:               holdout 86.793 +- 0.294
CANDIDATE blend:             holdout 86.946 +- 0.369
=> CANDIDATE WINS by +0.010; writing ensemble_final.json
[FINAL blend test] QWK=87.2653 Acc19=36.21 Acc7=59.83 Acc5=66.96 Acc3=74.46 Adj+-1=74.94 MAE=1.0852
differs from V7 on 557/8077 (6.9%)  mean pred 10.49
```

(`ensemble_final.json` is the file's name at the time; it is now
[`configs/ensembles/final_v8.json`](../../configs/ensembles/final_v8.json).)

**+0.010 is inside the noise floor**, and worth being honest about: the gate did not establish that
21 members beat 20, only that the enlargement was not a regression. It shipped because it was free
and not harmful, not because it was an improvement. The +0.03†-per-model saturation estimate in
[E12](../e12_post_deadline_ablations/) is the same finding stated cleanly.

## Why the score cache mattered

V8's 21×8,077 member-score matrix was cached during that inference run
(`artifacts/scores/blind/prediction_final_scores.npz`). Every submission after this one, the two
accuracy-frontier points, submission 10, and the later regression, is that matrix **re-thresholded**.
No GPU, seconds per candidate, on a night when the GPU was busy and the deadline was hours away.

## Reproducing this exactly

`endgame.py` globs whatever `_ad` models are on disk, so its result depends on when you run it.
With the full zoo present (14 `_ad` models) the candidate scores 86.934 against the incumbent's
86.936 and the gate correctly writes nothing. To replay the original decision, pin the AD half to
the 11 models that existed at 22:03:

```bash
python experiments/e09_endgame_gate/endgame.py --ad-members \
  arabertv2_corn_ad arabertv2_emd_ad arabertv2_large_corn_ad arabertv2_large_reg_ad \
  arabertv2_large_soft_ad arabertv2_soft_ad arabertv2_wkl_ad araelectra_corn_ad \
  aramodern_reg_ad arbertv2_corn_ad arabertv2_reg_ad \
  --config-out /tmp/final_v8_check.json
```

That reproduces the log above line for line, including `+0.010` and the seven-metric FINAL line,
and writes a config byte-identical to `configs/ensembles/final_v8.json`.

## Files

| File | Role | Writes |
|---|---|---|
| `endgame.py` | build the candidate, A/B it against the incumbent, write only if it clears the gate | `configs/ensembles/final_v8.json` |
| `run_endgame.sh` | the unattended deadline sequence: wait, stop training, gate, infer, validate | submission + blind score cache |

Log: `artifacts/logs/run_endgame.log`, `infer_final.log`.

† From our working notes, see
[../../docs/reproducibility.md](../../docs/reproducibility.md#tier-3-not-verifiable-from-this-repository).
