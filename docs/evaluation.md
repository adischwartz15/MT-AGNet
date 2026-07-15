# Evaluation Metric Definitions

## Age (regression)

- **MAE**: mean absolute error, q50 vs. true age.
- **RMSE**: root mean squared error, q50 vs. true age.
- **R2**: coefficient of determination, q50 vs. true age.
- **Interval coverage**: fraction of samples where `q_low <= true_age <= q_high`.
- **Mean / median interval width**: `q_high - q_low`, averaged (mean or
  median) across samples.
- **Calibration error**: `|empirical coverage - target coverage|` for the
  q10-q90 interval (`expected_calibration_error_intervals`, default
  `target_coverage=0.80`).
- **Age error percentiles**: median / p90 / p95 of absolute age error --
  complements MAE/RMSE (both mean-based) with distributional detail.
- **Tail-error rates**: fraction of samples with absolute age error
  exceeding 5 / 10 / 15 / 20 years.
- **Per-age-bucket metrics** (`age_uncertainty_by_bucket`): MAE, coverage,
  and interval width computed within fixed age ranges. 

**Raw vs. calibrated intervals.** The q10-q90 interval is a nominal 80%
interval (`calibration.alpha: 0.10` in `configs/training.yaml`) as output
directly by the trained quantile head - its *empirical* coverage on held
out data is not guaranteed to match 80% (or any other rate) until a
conformal offset has been fit and applied (see `docs/calibration.md`).
Always state explicitly whether a reported interval/coverage number is
raw or calibrated; never call a raw q10-q90 interval "a 90% interval."

## Gender label (classification with abstention)

The gender-label head abstains ("Not sure") whenever its top softmax
probability falls below `confidence_threshold` (default 0.80,
`configs/model.yaml: model.gender_head.confidence_threshold`). 
- **Selective accuracy** (`gender_accuracy`): accuracy over **non-abstained**     predictions only (denominator excludes abstentions).
- **Coverage** (`gender_coverage`): fraction of samples actually
  answered, `1 - abstention_rate`.
- **Abstention rate** (`abstention_rate`): fraction returned as "Not
  sure."
- **Effective accuracy** (`gender_effective_accuracy`): correct-and-accepted
  predictions divided by **all** samples (denominator includes
  abstentions) - "how often a user actually gets a correct answer out of
  everything asked."

A model can have excellent selective accuracy while abstaining on every
difficult case, which would look poor on effective accuracy - reporting
selective accuracy alone would hide that trade-off. Whenever this
repository or its documentation reports a bare "gender accuracy" number,
it means **selective accuracy** unless explicitly labeled otherwise;
always check whether coverage/effective accuracy are reported alongside
it before treating the number as a full picture.

## Selective prediction / AURC (both tasks)

`src/evaluation/selective.py`, used identically for gender (confidence =
max class probability, per-sample loss = 0/1 error) and age (confidence =
`-(q90 - q10)`, narrower interval = more confident, per-sample loss = MAE
or RMSE):
- **Risk-coverage curve**: sweeps coverage from low to full, keeping the
  most-confident fraction of samples at each step, and reports the mean
  loss ("risk") over that fraction.
- **AURC** (area under the risk-coverage curve): trapezoidal integration
  of risk over coverage; lower is better.
- **Paired bootstrap CI at a fixed coverage level**: resamples the same
  indices for both models each iteration (valid only when both were
  evaluated on the identical, index-aligned test set), giving a CI for
  the risk difference **at one specific coverage level**.
- **Paired bootstrap CI on the AURC statistic itself**: a separate,
  stricter test -- a CI at one fixed coverage level is not sufficient
  evidence for a claim about AURC as a whole; only a CI computed directly
  on the AURC difference is treated as such (see
  `docs/backbone_comparison.md` and `docs/final_evaluation_protocol.md`).

