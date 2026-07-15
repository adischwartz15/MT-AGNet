# Execution Modes and Notebook Configuration

Covers every flag in the "USER CONFIGURATION" cell shared by both
notebooks (`notebooks/train_evaluate_colab.ipynb`,
`notebooks/train_evaluate_kaggle.ipynb`).

## What runs, always

Both notebooks always train and evaluate the full architecture ablation
suite in one pass: Experiments 0 (SimpleCNN), 0b (PlainDeep18NoSkip --
residual-connection control), 0c (no-zero-init-residual control), A
(separate backbones), B (shared, no adapters), C (shared + adapters,
fixed weights), and D (shared + adapters + learned balancing), then
Experiment E (D's parametric heads vs. a k-NN baseline over D's own
embeddings). Already-complete experiments (checkpoint + metrics already
on disk) are skipped automatically -- see "Resume safety" below.

There is no more per-run experiment *selection* (the old `RUN_PROFILE`
system): everything above always runs. What's still configurable is
*how long* each experiment trains, and which *optional* analyses run on
top.

## The "USER CONFIGURATION" cell, flag by flag

| Flag | Controls | Notes |
|---|---|---|
| `SMOKE_TEST` | Caps every experiment to 1 epoch / patience 1 / 3 batches per epoch | Fast pipeline-validation check only -- confirms every stage runs without crashing. **Results are never scientific findings**; the CNN-vs-ResNet table and detailed report are skipped for this run. Use before a real run, after any code change, or to validate a new environment/dataset path. |
| `FORCE_RERUN` | Whether to retrain/recalibrate/re-evaluate a stage even if its artifact already exists | Default `False` (restart-safe: an already-complete stage is skipped). |
| `ALLOW_TEST_FAILURES` | Whether the notebook continues past a failing `pytest` run | Default `False` -- a real test failure should stop the notebook, not be silently ignored. |
| `RUN_EXPERIMENT_E` | Whether Experiment E (parametric vs. k-NN over D's embeddings) runs | Default `True`. Reuses D's checkpoint; never retrains. |
| `RUN_NONPARAMETRIC_BASELINES` | Whether the raw-pixel/PCA and frozen-backbone k-NN/KDE baselines run | Default `True`. CPU-only, no trained checkpoint needed. |
| `RUN_ROBUSTNESS` | Whether the 11-corruption-type robustness sweep runs for Experiments 0 / 0b / D | Default `True`. |
| `RUN_GRADCAM` | Whether Grad-CAM attention maps are generated for Experiments 0 / 0b / D | Default `True`. |
| `RUN_MULTI_SEED` | Whether Experiments 0 / 0b / D are re-run at every seed in `SEEDS` for a mean +/- std | Default `False` since it multiplies training time by `len(SEEDS)`. |
| `SEEDS` | The pre-registered seed list for multi-seed runs | Default `[42, 123, 2026]`, matching `docs/final_evaluation_protocol.md`. `SEEDS[0]` (`PRIMARY_SEED`) is always used for the single-seed run regardless of `RUN_MULTI_SEED`. |
| `MAX_EPOCHS` | Cap passed as `training.warm_up_from_scratch.epochs` | Default 40. Hard-capped to 1 automatically under `SMOKE_TEST=True`. |
| `EARLY_STOPPING_PATIENCE` | Epochs of no `val_loss` improvement before stopping a stage early | Default 12. Hard-capped to 1 under `SMOKE_TEST=True`. |
| `MAX_BATCHES_PER_EPOCH` | Optional hard cap on batches per epoch, independent of epoch count | Default `None` (unlimited). Auto-set to 3 under `SMOKE_TEST=True`. |
| `REPO_URL` / `REPO_BRANCH` | Where the notebook clones/pulls this repository from | Must match the real remote (verified: `https://github.com/adischwartz15/AgeGender.git`, branch `main`). |
| `RESUME_RUN_ID` | Continue a previous run instead of starting fresh | Works across a genuine session restart, not just same-session reruns -- see "Resume safety" below. |
| `USE_GOOGLE_DRIVE` (Colab only) | Whether the run directory is synced to Drive after every major phase, and restored from Drive when resuming | Requires Drive to actually be mounted (handled automatically when this is `True`). |
| `KAGGLE_DATASET_SLUG` | Kaggle API dataset slug to download | Pre-filled with `jangedoo/utkface-new` (the standard UTKFace-on-Kaggle dataset this project's `utkface` adapter expects) in the Kaggle notebook; `None` by default in the Colab notebook. |
| `DRIVE_DATASET_DIR` (Colab) / `KAGGLE_INPUT_DATASET_DIR` (Kaggle) | Use an already-available local/Drive/attached-input dataset instead of downloading | Exactly one dataset source (this or `KAGGLE_DATASET_SLUG`) must be set, or the dataset-setup cell raises. |
| `PREVIOUS_RUN_KAGGLE_INPUT_DIR` (Kaggle only) | Path to a previous session's own attached output, used together with `RESUME_RUN_ID` to resume across a real session restart | Kaggle has no persistent local disk between sessions -- without this, `RESUME_RUN_ID` only skips already-complete stages within the same still-alive session. |

Model-architecture, loss-balancing, and tau (confidence-threshold) values
are **not** set in this notebook cell at all -- they live in
`configs/model.yaml` (`model.gender_head.confidence_threshold`,
`model.loss_balancing.mode`) and `configs/experiments.yaml` (per-experiment
architecture overrides), consistent with the project's "config-driven, not
hardcoded" design. The notebook only selects *how long* each experiment
trains, never redefines its architecture inline. Robustness corruption
types/severities are likewise entirely config-driven from
`configs/robustness.yaml`, not the notebook.

## Resume safety

Every stage (train / calibrate / build k-NN index / evaluate) is checked
and skipped independently, based only on whether *that stage's own*
artifact already exists -- so a crash partway through never redoes
already-complete work when the notebook is re-run.

This works two ways:

- **Same session** (e.g. a cell raised an exception): just re-run from
  the top. `RUN_DIR` and everything already written to it are still on
  disk, so already-complete stages are skipped immediately.
- **New session** (a Colab VM recycled, or a Kaggle session ended): set
  `RESUME_RUN_ID` to the previous run's printed `RUN_ID`. Since a fresh
  VM/session has no local disk from before, the notebook restores
  whatever was already synced -- from Google Drive on Colab
  (`USE_GOOGLE_DRIVE=True`), or from `PREVIOUS_RUN_KAGGLE_INPUT_DIR` on
  Kaggle (attach the previous session's own output as an input dataset
  first). Without that restore step, a new session has nothing to
  resume from and every experiment reruns from scratch.

A fresh run (no `RESUME_RUN_ID`) never overwrites an existing run
directory -- a numeric suffix is appended instead.
