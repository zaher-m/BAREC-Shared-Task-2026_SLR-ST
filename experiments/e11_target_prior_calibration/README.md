# E11. Target-prior threshold calibration

**When** Aug 2 – Aug 3 · **Verdict** submission 10: blind 85.4 / 38.7

The headline method, and it costs **zero training**. Three steps on top of an ensemble that already
existed:

1. **Estimate the blind label prior** by regularized BBSE: build the row-normalized confusion matrix
   `C[i,j] = P(pred=j | true=i)` on held-out test, take the system's own blind *predicted*
   distribution `q`, and solve `min ‖Cᵀp − q‖² + λ‖p − p_src‖²` with **λ = 0.05**, then clip and
   renormalize. Estimated blind `Var_true` = **9.09** against test's 10.63, the blind prior is more
   concentrated.
2. **Reweight the calibration set** to that prior, with importance weights `p̂[y]/p_src[y]`.
3. **Refit the 18 QWK-optimal cut points** on the reweighted set and apply them to the cached blind
   scores.

Result: **blind QWK 85.3 → 85.4 and accuracy +2.8 → 38.7**, improving both leaderboard metrics at
once. The fixed-anchor proxy predicted it almost exactly beforehand, proxy 85.444 against the
baseline configuration's 85.357, which had corresponded to blind 85.3.

## How it was checked before submitting

The procedure was validated by simulation *before* it touched blind, in `prior_adapt.py`: split test
in half, resample one half to a shifted prior, estimate that prior **without its labels** via BBSE,
and check whether the adaptation beats naive thresholds at the concentrated shifts, the ones that
look like blind. Two adaptations were compared, a one-parameter dispersion rescale and the 19-dim
reweight-and-refit; the second is the one that earned the right to be applied.

The simulation also validates BBSE itself: estimated variance tracks true variance across the
simulated shifts (7.78/8.31, 8.85/9.20, 10.57/10.69, 12.80/12.81).

`oracle_alpha.py` answered the complementary question, is there headroom left at all? An oracle that
fits thresholds on the shifted labels beats naive by only ~0.7–0.8 QWK, and the blind dispersion
optimum is `a* = 1.0025`, i.e. **already optimal**. Whatever deficit remains is correlation, not
calibration.

## The failure mode, found the same night

A second BBSE pass, anchored on the system's own *updated* output, the natural EM-style iteration, drifted the estimated prior variance from 8.99 to **9.31** and its submission regressed blind
**85.4 → 85.2**. That submission (`pred_W1_wqwk`, generated 00:08, submitted the next morning) was
the project's last. **One pass only.**

## Provenance note

The ad-hoc script that emitted the submitted file (`pred_Q0`, `pred_Q1`,
`pred_priorQWK_acc`, between 23:59 and 00:01) was **never saved**. The method is documented
above and reconstructable from `src/slra_st/shift.py` plus `weighted_opt.py`, and the exact label
vector survives as `submissions/submitted/pred_priorQWK_acc`. `weighted_opt.py` is the
post-hoc, exact re-derivation: deterministic importance weights over all 7,286 points instead of a
finite resample.

## Files

| File | Role | Writes |
|---|---|---|
| `prior_adapt.py` | validate the procedure under simulated shift before trusting it | - |
| `oracle_alpha.py` | oracle headroom, and the blind dispersion optimum `a*` | - |
| `weighted_opt.py` | exact importance-weighted threshold fitting; the iterated-BBSE regression | `submissions/candidates/pred_W{1,2,3}_*` |
| `distribution_matching.py` | quantile-matching calibration, **refuted** without a true prior | - |

BBSE, priors and importance weights live in `src/slra_st/shift.py`; the weighted metrics in
`src/slra_st/metrics.py`.
