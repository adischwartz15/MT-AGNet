# Experiment Plan

This document describes the config-driven ablation suite in
`configs/experiments.yaml` and the specific hypotheses each experiment is
designed to test. All experiments train/evaluate against the same
`data/splits/full_metadata_with_splits.csv` (see `docs/reproducibility.md`),
so differences in outcome are attributable to the architecture/training
change under test.

## Research question

Does a single shared Custom ResNet-18 backbone learn visual features
useful for *both* age estimation and dataset gender-label classification,
and do task-specific bottleneck adapters plus learned uncertainty-based
loss balancing reduce negative transfer relative to naive sharing or fully
independent backbones? 
Separately: how does a non-parametric k-NN classifier/regressor in the learned embedding space compare to the parametric heads?

1. **SimpleCNN vs. Custom ResNet-18 (Experiment 0 vs. D)** - "is the
   larger residual architecture justified relative to a compact CNN?"
   SimpleCNN differs from Custom ResNet-18 in depth, stage widths, *and*
   the presence of skip connections all at once, so this is an
   **efficiency/accuracy trade-off** comparison, not a clean ablation of
   residual connections specifically. A ResNet win here could be explained
   by depth/width alone, with skip connections contributing nothing.
2. **PlainDeep18NoSkip vs. Custom ResNet-18 (Experiment 0b vs. D)** -
   "what is the contribution of residual skip connections when depth and
   width are held fixed?" PlainDeep18NoSkip
   (`src/models/plain_deep18_no_skip.py`) copies Custom ResNet-18's stem,
   stage widths, block layout, embedding size, and training recipe exactly,
   removing only the residual additions (and, unavoidably, the
   downsample-shortcut parameters that only exist to support them - see
   Experiment 0b below for the exact count). This *is* a clean ablation of
   residual connections specifically.


## Experiment 0 - Plain CNN backbone baseline (`exp_0_simple_cnn_shared_adapters_learned_balance`)

A **controlled baseline, not a general CNN benchmark**: a conventional,
non-residual CNN (`src/models/simple_cnn.py` - stacked Conv+BN+ReLU+MaxPool
blocks, no skip connections) substituted for the Custom ResNet-18 backbone,
with everything else held identical to Experiment D - the same shared
multi-task structure, task-specific adapters, learned uncertainty loss
balancing, training setup, data split, and evaluation pipeline. The plain
CNN uses the same 512-d embedding output, so the adapters/heads/losses are
byte-for-byte the same code path regardless of which backbone feeds them.

This isolates one variable: **residual connections, present or absent**.
It deliberately does not compare a weak CNN with fixed losses against a
ResNet with adapters and learned balancing - that would change too many
variables at once to attribute any difference to the backbone. This is
not intended to be tuned into a competitive standalone architecture, and
the plain CNN must never be described as this project's main backbone;
`CustomResNet18` remains that throughout.

All three backbones (`custom_resnet18 | simple_cnn | plain_deep18_no_skip`)
expose the same `forward` / `forward_features` (`layer1`-`layer4`, for
Grad-CAM compatibility) / `num_parameters` interface, so the rest of the
pipeline (adapters, heads, trainer, evaluation, inference, Grad-CAM) is
unmodified by which one is active.

## Experiment 0b -- Plain, depth/width-matched, no-skip-connection backbone (`exp_0b_plain_deep18_no_skip_shared_adapters_learned_balance`)

The controlled residual-connections ablation Experiment 0 cannot provide
(see "Research question" above): `src/models/plain_deep18_no_skip.py`'s
`PlainDeep18NoSkip` uses the **same** stem, stage widths (64/128/256/512),
block layout `[2, 2, 2, 2]` (two 3x3 convolutions per block), BatchNorm,
ReLU placement, embedding size, adapters, heads, learned-uncertainty loss
balancing, and training recipe as Custom ResNet-18 (Experiment D) -- the
only change is that `PlainBlock.forward` never adds an identity/projection
shortcut.

**Unavoidable parameter difference.** Because there is no residual addition,
there is also no need for the three downsample shortcuts (1x1 conv +
BatchNorm) Custom ResNet-18 uses at the layer2/layer3/layer4 channel/stride
transitions -- `PlainDeep18NoSkip` has exactly **173,824 fewer parameters**
than `CustomResNet18` (11,002,688 vs. 11,176,512 with default
`stem_channels=64`), matching those three shortcuts' parameter count
exactly (verified in `tests/test_plain_deep18_no_skip.py`). This is not a
design choice that favors either architecture; it is what "remove the skip
connections and nothing else" necessarily implies.

## Experiment 0c -- Custom ResNet-18, no zero-init residual (`exp_0c_custom_resnet18_no_zero_init_shared_adapters_learned_balance`)

The recommended architecture control: identical to Experiment D in every
respect (architecture, adapters, learned loss balancing, seeds, training
setup) except `model.backbone.zero_init_residual=false` - each residual
branch's final BatchNorm keeps its default init (weight=1) instead of
being zeroed. This is orthogonal to "does the residual connection exist at
all" (Experiment 0b's question) and isolates a specific, common ResNet
training trick instead:

- **PlainDeep18NoSkip vs. Experiment 0c** tests residual shortcuts more
  cleanly than PlainDeep18NoSkip vs. Experiment D, because PlainDeep18NoSkip
  and Experiment 0c both use non-zero-init residual-branch normalization --
  Experiment D additionally differs by zero-initializing its residual
  branches, which is a second, confounding variable if PlainDeep18NoSkip is
  only ever compared against Experiment D.
- **Experiment D vs. Experiment 0c** isolates the effect of zero-initialized
  residual branches on their own, holding the presence of the residual
  connections themselves fixed.

Experiment D (the project's actual reported ResNet configuration, with
`zero_init_residual: true`) is never changed by adding this control.


## Experiment A - Separate models (`exp_a_separate`)

Two independent Custom ResNet-18 backbones, one trained only for age, one
trained only for dataset gender-label classification. 

This is the "no sharing" baseline: it isolates what each task can achieve with a
dedicated backbone and establishes the parameter-cost ceiling (2x backbone
parameters) against which sharing is judged.

## Experiment B - Shared backbone, no adapters (`exp_b_shared_no_adapters`)

One shared backbone feeds both heads directly, fixed loss weights. Tests
whether naive parameter sharing causes **negative transfer** (worse
per-task performance than Experiment A) or **positive transfer** (better,
due to shared low-level visual features like edges/texture/illumination
invariance).

## Experiment C - Shared backbone + adapters (`exp_c_shared_adapters`)

Adds task-specific residual bottleneck adapters on top of Experiment B's
shared backbone. Tests whether adapters recover per-task specialization
lost in Experiment B while keeping most parameters shared (adapters are
configured to be a small fraction of backbone size, see
`docs/architecture_analysis.md`).

## Experiment D - Shared + adapters + learned loss balancing (`exp_d_shared_adapters_learned_balance`)

Same architecture as C, but replaces fixed loss weights with learned
homoscedastic-uncertainty weighting (trainable log-variances per task).
Tests whether automatic loss balancing improves on manually fixed weights
once adapters already address representational conflict.

## Experiment E - Parametric vs. k-NN (`exp_e_parametric_vs_knn`)
Not a separate training run: reuses Experiment D's (or the best-performing
experiment's) checkpoint.

## Experiment F - Pretrained vs. scratch (`exp_f_pretrained_vs_scratch`, optional)

Compares a backbone initialized from this repository's own SimCLR-style
self-supervised pretraining (`scripts/pretrain.py`) against the same
architecture trained from scratch. Skipped automatically (with a logged
message) if no pretrained checkpoint exists -- this experiment is opt-in
because self-supervised pretraining is comparatively compute-hungry (see
`docs/reproducibility.md`).

