# Reproducibility

## Stratified, Locked Split

`scripts/prepare_data.py` creates one file,
`data/splits/full_metadata_with_splits.csv`, with four splits: `train`,
`validation`, `calibration`, and `test`. Every experiment in
`configs/experiments.yaml` reads this same file, so results are always
comparable — any difference comes from the model/training change being
tested, not from a different data split.

- **Stratified**: samples are split so that each age group x gender label
  keeps roughly the same train/val/calibration/test proportions, not just
  a random shuffle. See `src/data/split_utils.py`.
- **Locked**: once a split is created, its SHA-256 hash is saved in
  `data/splits/split_manifest.json`. Every script that reads the split
  checks this hash first. If it doesn't match, something changed the
  split file — so it's either an error or a deliberate re-split. You can
  force a new split with `--force-resplit`. Either way, the old split is
  always backed up (never deleted) to `data/splits/.backup/`.
- **Safe to interrupt**: the split file and manifest are written to a
  temporary path first, then swapped in. A crash mid-write can't corrupt
  them.
- **The manifest records**: split method/seed, the split file's SHA-256,
  per-split sample counts, a near-duplicate check summary, the git commit
  that created it, and a timestamp.

### Near-duplicate check

`src/data/near_duplicate_audit.py` flags images that are probably
near-duplicates of each other (e.g. the same photo resized or
re-compressed) using perceptual hashing. It only reports candidates in
the manifest — it never removes anything automatically.

Every experiment records the split's SHA-256 in its own output, and
downstream steps like calibration refuse to run if that hash doesn't
match the currently locked split.

## Config-driven, not hardcoded

All architecture, training, and evaluation settings live in
`configs/*.yaml`. 
Every saved checkpoint embeds a full copy of the config that produced it.


## Environment

- Python 3.10+ 
- PyTorch, CPU or CUDA — see `requirements.txt`. 

## Running on Kaggle or Google Colab

Two ready-to-run notebooks, one per platform, each covering the entire
pipeline end to end in a single pass: setup, data prep, tests, the full
architecture ablation suite (Experiments 0/0b/0c/A/B/C/D), Experiment E
(parametric vs. k-NN), optional non-parametric baselines/robustness/
Grad-CAM/multi-seed runs, a detailed results report, and a final summary.
See `docs/execution_modes.md` for every configuration flag.

- `notebooks/train_evaluate_colab.ipynb` -- Google Colab. Syncs the run
  directory to Google Drive after every phase, and restores from Drive
  when resuming a previous run in a fresh session.
- `notebooks/train_evaluate_kaggle.ipynb` -- Kaggle Notebooks. Uses an
  attached Kaggle input dataset (or the Kaggle API, pre-filled with the
  standard `jangedoo/utkface-new` dataset slug), never mounts Google
  Drive, and produces a downloadable zip archive under Kaggle's Output
  tab. Resuming across a session restart requires attaching the previous
  session's own output as an input dataset (see `PREVIOUS_RUN_KAGGLE_INPUT_DIR`
  in `docs/execution_modes.md`).

Both notebooks are restart-safe: every stage (train / calibrate / build
k-NN index / evaluate) is checked and skipped independently if its
artifact already exists, so an interruption never redoes already-complete
work -- see `docs/execution_modes.md`'s "Resume safety" section.
