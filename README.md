# Face Multi-Task Research

An uncertainty-aware multi-task vision system for age quantile estimation
and dataset gender-label classification, built around a manually
implemented ResNet-18, shared representations, task-specific adapters,
conformal calibration, and selective prediction.

> **Research and demonstration only.** Predictions may be inaccurate,
> biased, or unreliable. Gender-related output reflects labels in the
> training dataset and is **not** a determination of identity. This
> project must not be used for employment, policing, surveillance,
> identity verification, medical diagnosis, admissions, insurance, or any
> other high-impact decision. See [Ethical limitations](#ethical-limitations).

## Highlights

- **Manually implemented ResNet-18** backbone -- no `torchvision.models`,
  `timm` backbones, or externally pretrained checkpoints anywhere in the
  ablation suite.
- **Shared backbone + task-specific residual bottleneck adapters** for
  age and dataset gender-label prediction, with an optional learned
  homoscedastic-uncertainty loss balancer instead of fixed weights.
- **Quantile-based age estimation** (q10/q50/q90) with **split-conformal
  calibration** for a marginal coverage guarantee, not just a point
  estimate.
- **Confidence-based abstention** ("Not sure") for the gender-label head,
  with selective/effective accuracy always reported together.
- **Parametric vs. k-NN comparison** in the model's own learned embedding
  space, deterministic **robustness testing** under 11 corruption types,
  and manually implemented **Grad-CAM** ("model attention visualization").
- **Controlled architecture ablations** (shared vs. separate backbones,
  adapters vs. none, fixed vs. learned loss balancing, and a true
  residual-connections ablation) with gradient-interference and
  representation-similarity (CKA) analysis.
- Reproducible, leakage-checked data splitting and fully isolated
  per-experiment/seed artifacts -- see [Reproducibility and scope](#reproducibility-and-scope).

## Research question

Does a **shared** visual backbone learn useful common features for both
age and dataset gender-label prediction, and do **task-specific
adapters** plus **learned loss balancing** reduce negative transfer
relative to independent backbones and fixed loss weights? Separately: is
the added complexity of a residual (skip-connection) architecture
actually justified, once measured against a depth/width-matched
non-residual baseline rather than an unrelated compact CNN?

These questions are answered by a config-driven ablation suite (Experiments
0/0b/0c, A-F) against one fixed, reused data split, following a protocol
pre-registered before results are observed -- see
[docs/experiment_plan.md](docs/experiment_plan.md) and
[docs/final_evaluation_protocol.md](docs/final_evaluation_protocol.md).
Results below are not a claim that every question was answered
conclusively; see [docs/results.md](docs/results.md) for what one real
run actually found.

## Architecture

```
Input face image
    |
    v
Custom ResNet-18 backbone (manually implemented)
    |
    v
Shared 512-dimensional embedding
    |
    +-- Age Adapter -------- Age Quantile Head -------- q10, q50, q90
    |
    +-- Gender Adapter ----- Classification Head ------- probabilities / "Not sure"
```

A single hand-written ResNet-18 backbone feeds two residual bottleneck
adapters, one per task, which in turn feed the age quantile head and the
gender classification head. Two controlled baseline backbones
(`simple_cnn`, `plain_deep18_no_skip`) also exist purely to isolate what
the residual design contributes -- neither is used by the deployed model.
See [docs/architecture_analysis.md](docs/architecture_analysis.md) for
the full module-by-module design and analysis methodology.

## Headline results

From one real training run on UTKFace (checkpoint
`exp_d_shared_adapters_learned_balance`, one seed, one dataset split --
see [docs/results.md](docs/results.md) for the full numbers, robustness
table, and gradient-interference/CKA analysis).

| Metric | Parametric | k-NN (k=15) |
|---|---|---|
| Age MAE | 5.71 | 5.79 |
| q10-q90 interval coverage (raw, uncalibrated) | 0.79 | 0.91 |
| Gender-label selective accuracy | 0.970 | 0.966 |
| Abstention rate (confidence threshold 0.80) | 0.192 | 0.179 |
| Latency per image (ms) | 1.8 | 2.0 |

| Experiment | Backbone params | Adapter params | Total params |
|---|---|---|---|
| A -- separate backbones | 22,353,024 | 0 | 22,484,997 |
| D -- shared + adapters + learned balancing | 11,176,512 | 263,424 | 11,571,911 |

"Gender-label selective accuracy" is computed only over non-abstained
predictions (see [docs/evaluation.md](docs/evaluation.md) for the
distinction from coverage and effective accuracy). The q10-q90 interval
is a nominal 80% interval before conformal calibration; the row above is
raw, not calibrated. These numbers describe one checkpoint on one
dataset split -- see [Reproducibility and scope](#reproducibility-and-scope).

## Quick start

```bash
git clone https://github.com/adischwartz15/AgeGender.git
cd AgeGender
make install
cp .env.example .env              # fill in Kaggle credentials (see docs/data_card.md)
make download-data
make prepare-data
make train
make calibrate CHECKPOINT=checkpoints/multitask_best_balanced_score.pt
make evaluate CHECKPOINT=checkpoints/multitask_best_balanced_score.pt
```

Requirements: Python 3.11+ (3.10+ also works).

## Main workflow

```bash
make prepare-data                          # validate + split raw metadata
make train                                  # single default configuration
make experiments                            # full ablation suite (0/0b/0c, A-F)
make calibrate CHECKPOINT=<checkpoint>.pt   # split-conformal age intervals
make build-knn CHECKPOINT=<checkpoint>.pt   # k-NN baseline index
make evaluate CHECKPOINT=<checkpoint>.pt    # test-set metrics + k-NN comparison
make robustness CHECKPOINT=<checkpoint>.pt  # corruption robustness sweep
make gradcam CHECKPOINT=<checkpoint>.pt     # Grad-CAM heatmaps
```

`prepare-data`, `pretrain`, `train`, and `experiments` accept
`--set key.path=value` config overrides via `ARGS` (e.g.
`make train ARGS="--set model.architecture=shared_no_adapters"`) instead
of editing YAML in place. The evaluation-side commands (`calibrate`,
`build-knn`, `evaluate`, `robustness`, `gradcam`, `compare-backbones`,
`run-seeds`) take explicit flags instead (`CHECKPOINT=`, `EXPERIMENT=`,
`SEEDS=`, etc. -- see each script's `--help`), not `--set`. See
[Documentation](#documentation) below for the guide covering each stage.

## Repository structure

```
configs/     YAML configuration (data, model, training, experiments, robustness)
src/         Library code (data, models, losses, training, evaluation, inference, utils)
scripts/     CLI entry points, one per pipeline stage
tests/       Pytest suite, including a synthetic-data smoke training test
notebooks/   Self-contained Colab and Kaggle notebooks running the full pipeline
docs/        Architecture, experiments, data/model cards, reproducibility
```

`data/`, `checkpoints/`, `experiments/`, and `outputs/` hold generated
artifacts and are never committed -- see
[docs/reproducibility.md](docs/reproducibility.md) for the full layout.

## Documentation

- [Architecture and model design](docs/architecture_analysis.md)
- [Experiment plan (Experiments 0/0b/0c, A-F)](docs/experiment_plan.md)
- [Final evaluation protocol (pre-registered)](docs/final_evaluation_protocol.md)
- [Headline results (full numbers)](docs/results.md)
- [Backbone comparison suite](docs/backbone_comparison.md)
- [Conformal calibration](docs/calibration.md)
- [Robustness evaluation](docs/robustness.md)
- [Evaluation metric definitions](docs/evaluation.md)
- [Non-parametric baselines (raw/PCA and frozen-backbone)](docs/nonparametric_baselines.md)
- [Colab and Kaggle notebooks, execution modes and flags](docs/execution_modes.md)
- [Reproducibility](docs/reproducibility.md)
- [Data card](docs/data_card.md)
- [Model card](docs/model_card.md)
- [Troubleshooting](docs/troubleshooting.md)

## Ethical limitations

- **"Dataset gender-label prediction"**, not "gender prediction" -- the
  output reflects a label defined by whichever dataset you train on, not
  a determination of a person's gender identity. Class names default to
  the neutral `gender_label_0` / `gender_label_1`.
- Dataset labels may be binary, incomplete, inaccurate, self-reported,
  annotator-assigned, or culturally limited.
- Race/ethnicity metadata (when present, e.g. in UTKFace) is **never**
  used as a feature, prediction target, or split criterion.
- This system has not been validated for, and must not be used for:
  employment, policing, surveillance, identity verification, medical
  diagnosis, admissions, insurance, or any other high-impact decision.
- Grad-CAM output is a gradient-weighted visualization, **not proof of
  causality** and not an explanation of the model's reasoning.

See [docs/model_card.md](docs/model_card.md) and
[docs/data_card.md](docs/data_card.md) for the full discussion.

## Reproducibility and scope

Every reported number is a property of one specific dataset, split, seed,
and evaluation design -- not a universal statement about the underlying
task. All splits are fixed once and reused by every experiment; every
checkpoint/seed gets its own isolated artifact tree; calibration artifacts
record and verify provenance (checkpoint/split hashes) before being
applied; and nothing in this repository hardcodes example metrics as if
they were real results. Supervised training on a few thousand 128px
images is feasible on a single consumer GPU in well under an hour per
experiment. See [docs/reproducibility.md](docs/reproducibility.md) for
seeds, splits, compute expectations, and notebook details, and
[docs/data_card.md](docs/data_card.md) for demographic-coverage caveats.

## License / authors

MIT License -- see [LICENSE](LICENSE).
