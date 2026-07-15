# Results 

## Architecture parameter comparison (Experiments A-D)

| Experiment | Backbone params | Adapter params | Total params |
|---|---|---|---|
| A -- separate backbones | 22,353,024 | 0 | 22,484,997 |
| B -- shared, no adapters | 11,176,512 | 0 | 11,308,485 |
| C -- shared + adapters | 11,176,512 | 263,424 | 11,571,909 |
| D -- shared + adapters + learned balancing | 11,176,512 | 263,424 | 11,571,911 |

Sharing the backbone (B/C/D) roughly halves parameter count versus
independent backbones (A); adapters add back only ~2.4% of the shared
backbone's parameters per task. *Per-experiment accuracy/MAE comparison
(does sharing + adapters actually help, not just cost fewer parameters)
requires re-running `scripts/evaluate.py` against each experiment's
checkpoint and isn't included here yet -- the sections below reflect one
specific (shared-backbone + adapters) checkpoint, not a cross-experiment
comparison.*

## Parametric Model vs. k-NN Baseline (Experiment E)

Same trained checkpoint (Experiment D), compared two ways: its own
prediction heads ("Parametric") vs. a k-NN search (k=15) over that same
checkpoint's learned embeddings ("k-NN"). This shows how much of the
performance comes from the trained heads, not just the learned features.

| Metric | Parametric | k-NN (k=15) | Edge |
|---|---:|---:|---|
| Age MAE (lower is better) | **5.71** | 5.79 | Parametric, by 1.4% |
| Age RMSE (lower is better) | **8.32** | 8.53 | Parametric, by 2.5% |
| q10-q90 interval coverage (raw, uncalibrated) | 0.79 | **0.91** | k-NN, by 12 pts |
| Mean interval width (lower is tighter) | **16.79** | 26.88 | Parametric, ~38% narrower |
| Gender-label selective accuracy | **0.970** | 0.966 | Parametric, by 0.4 pts |
| Abstention rate (lower is more decisive) | 0.192 | **0.179** | k-NN, by 1.3 pts |
| Latency per image | **1.8 ms** | 2.0 ms | Parametric, ~10% faster |

**Bottom line:** the two methods score about the same on gender-label
accuracy. For age, they trade off in opposite directions: the parametric
model gives much tighter intervals (16.79 vs. 26.88 average width) but
hits its coverage target less often; k-NN's intervals are wider but land
closer to the target. Wider-but-more-accurate coverage is what a more
cautious method looks like -- it doesn't mean k-NN actually "wins."

Two things to keep in mind:
- **Neither row is calibrated.** The q10-q90 range is defined to cover
  80% of cases; 0.79 and 0.91 above are the raw, uncalibrated coverage --
  not a guarantee.
- **"Accuracy" here means selective accuracy** -- only counted on cases
  the gender head actually answered (see `docs/evaluation.md` for how
  this differs from coverage and effective accuracy).

## Gradient interference and representation similarity

Measured on the shared-backbone + adapters model (30 sampled batches; see
`docs/architecture_analysis.md`, sections 4-5, for full methodology):

- Mean task-gradient cosine similarity: **+0.08** (std 0.33) -- weakly
  positive, i.e. the age and gender-label gradients are not strongly in
  conflict on this dataset/split, with meaningful batch-to-batch variance.
- Linear CKA: shared-vs-age-adapter **0.79**, shared-vs-gender-adapter
  **0.90**, age-vs-gender-adapter **0.59** -- the gender adapter moves the
  shared representation less than the age adapter does, and the two
  adapters diverge from each other more than either diverges from the
  shared embedding.

## Robustness (deterministic corruptions, severity 1 of 3)

See `docs/robustness.md` for how to run this evaluation and
`docs/architecture_analysis.md` (section 7) / `docs/final_evaluation_protocol.md`
for the full corruption/severity definitions.

| Condition | Age MAE | Gender-label selective accuracy |
|---|---|---|
| Clean (no corruption) | 5.52 | 0.975 |
| Gaussian blur | 5.72 | 0.953 |
| Low resolution | 5.82 | 0.934 |
| Low brightness | 6.21 | 0.962 |
| JPEG compression | 6.60 | 0.960 |
| High brightness | 6.80 | 0.947 |
| Partial crop | 8.50 | 0.868 |
| **Partial occlusion** | **13.35** | 0.765 |
| **Gaussian noise** | **14.82** | 0.960 |

Gaussian noise and partial occlusion are, by a wide margin, the most
damaging conditions for age estimation in this run; gender-label selective
accuracy degrades more gracefully except under occlusion. This table is
severity 1 of 3 only -- see `docs/robustness.md` for the full
severity range and `docs/final_evaluation_protocol.md` for the exact
per-severity corruption parameters.

## Results depend on your data

Every number on this page -- age MAE, gender-label accuracy, interval
coverage, robustness curves, gradient interference, CKA -- is a property
of **the specific dataset, labels, split, and evaluation design used for
this run**, not a universal statement about the underlying task.
Different datasets have different demographic coverage, label quality,
and image conditions; do not extrapolate these results to populations,
cameras, or use cases outside the evaluation data. See
`docs/data_card.md` and `docs/model_card.md`.
