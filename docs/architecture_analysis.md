# Architecture Analysis (Methodology)

## 0. Model components

```
Input face image
      |
      v
Custom ResNet-18 backbone (manually implemented, block layout [2,2,2,2])
      |
      v
Shared feature vector z (512-d)
      |
      +----------------------+----------------------+
      |                                             |
      v                                             v
Age Adapter (residual bottleneck)          Gender Adapter (residual bottleneck)
      |                                             |
      v                                             v
Age Quantile Head                          Gender Classification Head
  -> q10, q50, q90                            -> probabilities
```

- **Backbone**: `src/models/custom_resnet.py` -- hand-written `BasicBlock`
  residual blocks, manual downsampling (strided 1x1 conv + BN shortcuts),
  stem conv + BN + ReLU + max-pool, adaptive average pooling, 512-d
  embedding. 
  The only way to initialize non-random weights is a checkpoint produced
  by this repository (supervised training or the optional SimCLR-style
  pretraining) or a compatible local file you explicitly point at.

- **Adapters**: 
`src/models/adapters.py` -- `adapter_output = z + up(dropout(gelu(down(z))))`,
  configurable bottleneck dimension (default 256), near-identity at
  initialization (zero-initialized up-projection).

- **Heads**: `src/models/heads.py` -- age quantile head (safe
  `q50, q50 - softplus(.), q50 + softplus(.)` parameterization guaranteeing
  `q10 <= q50 <= q90`) and a softmax gender classification head.

- **Loss balancing**: `src/losses/multitask_loss.py` -- fixed weights or
  learned homoscedastic-uncertainty weighting, with masked losses so a
  task with no labels in a batch contributes nothing.

This document describes *how* this repository analyzes the shared-backbone
/ adapter / loss-balancing architecture questions, and how to read the
generated numbers. It is a fixed methods reference; the actual numbers for
your run live in `docs/architecture_analysis_generated.md` (produced by
`make architecture-report`, backed by `src/evaluation/reports.py`) and in
`outputs/architecture_analysis/`. See `docs/experiment_plan.md` for the
per-experiment hypotheses this analysis is meant to test.

## 1. Parameter counts

`MultiTaskFaceModel.parameter_breakdown()` (`src/models/multitask_model.py`)
splits parameters into `backbone`, `adapters`, `heads`, and (if learned
loss balancing is enabled) `log_variance`. For Experiment A (separate
backbones) `backbone` is the sum of both independent backbones' parameters.
This lets you directly compare, e.g., Experiment A's 2x backbone cost
against Experiment C/D's single backbone + two small adapters.

## 2. Training time / inference latency / GPU memory

`src/training/trainer.py` records wall-clock epoch time per stage
(`outputs/metrics/<experiment>_timing.json`). `scripts/evaluate.py` and
`scripts/build_knn_index.py` record per-image inference latency for both
the parametric model and the k-NN baseline. GPU memory, when running on
CUDA, can be read from `torch.cuda.max_memory_allocated()` around a
training or inference call if you need it for your own reporting - it is
not persisted by default since this repository's default target hardware
is not assumed to have a GPU.

## 3. Task performance

Standard metrics (`src/evaluation/metrics.py`): age MAE/RMSE/R2, q10-q90
interval coverage and width (before/after conformal calibration, see
`docs/reproducibility.md` and `src/evaluation/calibration.py`), gender-label
accuracy (computed only over non-abstained predictions), abstention rate,
and confidence statistics.

## 4. Gradient interference (task-gradient cosine similarity)

For shared-backbone architectures (Experiments B/C/D; not defined for
Experiment A's independent backbones), `src/evaluation/architecture_analysis.py:compute_gradient_cosine_similarity`
does, per sampled batch:

1. Forward pass once.
2. Backward the age (pinball) loss with `retain_graph=True`; snapshot the
   gradient of every shared-backbone parameter.
3. Zero gradients; backward the gender (cross-entropy) loss; snapshot
   again.
4. Cosine similarity between the two flattened gradient vectors.

**Interpretation:**
- **Positive** mean cosine similarity: the two tasks pull shared weights
  in aligned directions -- evidence the shared representation is not (on
  average) fighting itself.
- **Negative**: the tasks pull in conflicting directions for at least part
  of training -- a mechanistic signal for "negative transfer", motivating
  adapters or learned loss balancing.
- **Near zero**: a weak/inconsistent relationship; not strong evidence
  either way.

This is measured for shared-backbone runs with and without adapters so the
report can state whether adapters change the *effective* conflict at the
backbone (adapters change what reaches the backbone via backprop, since the
adapter sits between the shared feature and the loss).

## 5. Representation similarity (linear CKA)

`src/evaluation/architecture_analysis.py:linear_cka` implements linear
Centered Kernel Alignment (Kornblith et al., 2019) between the shared
embedding `z` and each adapter's output. CKA is invariant to orthogonal
transformation and isotropic scaling, so it measures representational
similarity independent of arbitrary rotations a network might learn.

**Interpretation:**
- CKA close to 1: the adapter barely changes the representation for that
  task (little specialization, or the task doesn't need much).
- Lower CKA: the adapter meaningfully transforms the representation
  (more specialization). This is descriptive -- it does not by itself
  indicate whether that specialization is *helpful*; read it alongside the
  task performance tables.

## 6. Representation visualization (PCA / t-SNE)

`reduce_embeddings` projects shared embeddings to 2D via PCA or t-SNE for
visualization only, colored by age bucket (where age labels exist) and,
separately, by dataset gender label (where labels exist). These plots are
descriptive/exploratory - proximity or separation in a 2D projection does
not establish a causal claim about what the network is "using" to make a
prediction.

## 7. Robustness

See `configs/robustness.yaml` and `src/evaluation/robustness.py`: eleven
deterministic corruption types (Gaussian blur, Gaussian noise,
low-resolution/resize degradation, JPEG compression, low/high brightness,
low/high contrast, grayscale, partial occlusion, partial crop) at multiple
severities, evaluated with a fixed seed (the same corrupted images are
shown to every model compared). 
Reported per corruption/severity: age MAE, interval coverage/width, gender accuracy, abstention rate, and the same metrics for the non-parametric baseline.
`compute_degradation()` / `build_robustness_diff_table()` add delta/percent
degradation and a direct model-vs-model difference table for multi-model
comparisons (see section 9).

## 8. Grad-CAM ("model attention visualization")

Manually implemented (`src/evaluation/gradcam.py`) - no external Grad-CAM
library. Separate heatmaps for the age decision (backprop from q50) and
the gender-label decision (backprop from the selected class logit) at
the last residual stage (`layer4` by default). 
**This is a gradient-weighted activation visualization. It is not proof of causality, and it does not explain the model's "reasoning" in any human sense** - treat it as a
diagnostic aid, not an explanation.

## 9. Backbone comparison suite (selective prediction, tail errors, honest interpretation)

`src/evaluation/backbone_comparison.py` (via `scripts/compare_backbones.py`)
adds analyses the sections above don't cover, across two or more
checkpoints at once:

Clean-test summary: Beyond the usual average error (MAE/RMSE), this reports how big the errors are at the median and near the worst cases (90th/95th percentile), plus how often the model is off by more than 5, 10, 15, or 20 years. For gender, it reports accuracy on the predictions the model actually made, how often it made a prediction at all, how often it said "not sure," and overall accuracy counting the "not sure" cases as wrong (see docs/model_card.md, "Abstention behavior," for why this last number differs from plain accuracy).

Selective-prediction analysis (src/evaluation/selective.py): Sorts predictions from most to least confident (for gender: how sure the model was about its top guess; for age: how narrow its predicted range was), then asks "if the model only answered its most confident X%, how accurate would it be?" It does this across the full range of X, and summarizes it with one number (AURC) — lower is better. Two models are only compared while answering the same percentage of cases, since any model can look better just by refusing more often.

Paired bootstrap confidence intervals: To check if one model is really better (not just luck), it repeatedly resamples the same test examples for both models and measures the gap between them each time. If that gap consistently stays above (or below) zero across resamples, the difference is treated as real — never based on a single number.

Tail-error analysis: Looks specifically at the model's worst mistakes — the full distribution of age errors, and the average error broken down by age group (0-12, 13-19, 20-34, 35-49, 50-64, 65+) — to see if a model avoids big mistakes even if its average error looks the same as another model's.

Final interpretation: A built-in, honest verdict (build_final_interpretation) on whether the more complex residual architecture is actually worth it. It only credits the residual model if the bootstrap test shows a real, statistically solid advantage — otherwise it says plainly that the simpler model is the better choice. It's built to be able to conclude either way, not to favor the more complex model by default.
