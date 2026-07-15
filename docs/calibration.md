# Conformal Calibration

Practical guide to fitting and using split-conformal age intervals. For
the conceptual explanation of raw vs. calibrated intervals and what a
marginal coverage guarantee does (and does not) mean, see
`docs/model_card.md` ("Uncertainty interpretation"). For the pre-registered
target coverage/alpha used in the project's reported run, see
`docs/final_evaluation_protocol.md`.

## Running it

```bash
make calibrate CHECKPOINT=checkpoints/<your_checkpoint>.pt
# or, with explicit isolation/provenance (what run_experiments.py/run_seeds.py do internally):
python scripts/calibrate.py --checkpoint <checkpoint>.pt --calibration-dir <isolated_dir> --experiment-name <name> --seed <seed>
```

Fits split-conformal calibration (`src/evaluation/calibration.py`) on the
**dedicated calibration split only** -- deliberately not the validation
split, which is reserved for early stopping/checkpoint selection, so no
single split is used for two different decisions (see `docs/data_card.md`
for the full four-way split protocol). Saves the offset to
`conformal_calibration.json` under `--calibration-dir` (default:
`configs/training.yaml: calibration.output_dir`). Reports coverage/width
before and after calibration on the test set (touched only here, once)
(`calibration_test_effect.json` alongside it). An interval is only ever
described as "calibrated" when this artifact exists and loaded
successfully.

## Provenance and mismatch protection

Every calibration artifact records the checkpoint's SHA-256, the split
CSV's SHA-256, an ordered test-sample-ID hash, the experiment name, seed,
alpha, and target coverage. `scripts/evaluate.py` and
`scripts/run_robustness.py` validate a loaded calibration artifact's
recorded provenance against the checkpoint and split actually being
evaluated (`validate_calibration_artifact`) and raise
`CalibrationMismatchError` loudly on any mismatch -- e.g. applying seed
42's calibration to seed 123's checkpoint, or a SimpleCNN checkpoint to a
calibration artifact fit for ResNet.

`scripts/run_seeds.py` and `scripts/run_experiments.py` never fall back to
a shared global `outputs/calibration/` -- every checkpoint gets calibrated
into its own isolated
`experiments/<experiment>/seed_<seed>/calibration/` directory
(`src/utils/experiment_paths.py`) before being evaluated with calibration
applied.
