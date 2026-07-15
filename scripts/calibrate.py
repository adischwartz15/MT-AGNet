#!/usr/bin/env python
"""CLI: fit split conformal calibration for age intervals using the dedicated calibration split.

Per the 4-way split protocol (train/validation/calibration/test), this
intentionally does NOT reuse the validation split (which is reserved for
early stopping / checkpoint selection) -- fitting conformal intervals on
the same data used for model/checkpoint selection would let that data
influence both decisions, muddying the calibration guarantee. The test
split is only ever used afterward, once, to report the calibration's
effect (coverage/width before vs. after).

The saved calibration artifact records provenance (checkpoint SHA256,
split CSV SHA256, ordered test-sample-ID hash, experiment name, seed,
alpha, target coverage) so scripts/evaluate.py can refuse (loudly) to
apply it to a mismatched checkpoint or split later -- see
src/evaluation/calibration.py:validate_calibration_artifact.

Usage:
    python scripts/calibrate.py --checkpoint checkpoints/multitask_best_balanced_score.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import FaceMultiTaskDataset
from src.data.transforms import resolve_eval_transform
from src.evaluation.calibration import compute_preprocessing_fingerprint, evaluate_calibration_effect, fit_and_save_calibration
from src.inference.artifacts import load_model_checkpoint
from src.utils.config import REPO_ROOT, resolve_device
from src.utils.io import checkpoint_experiment_name, save_json
from src.utils.logging import get_logger

logger = get_logger("scripts.calibrate")


@torch.no_grad()
def _predict_age(model, dataset, device, batch_size=64):
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    q10s, q90s, ages, masks = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        outputs = model(images)
        q10s.append(outputs["age_output"]["q10"].cpu().numpy())
        q90s.append(outputs["age_output"]["q90"].cpu().numpy())
        ages.append(batch["age"].numpy())
        masks.append(batch["age_mask"].numpy())
    return np.concatenate(q10s), np.concatenate(q90s), np.concatenate(ages), np.concatenate(masks).astype(bool)


def calibrate_checkpoint(
    checkpoint_path: str,
    calibration_dir: str | None = None,
    alpha: float | None = None,
    experiment_name: str | None = None,
    seed: int | None = None,
) -> dict | None:
    """Fit + save a conformal calibration artifact for one checkpoint, with full provenance.

    This is the callable core used both by this script's CLI and by
    ``scripts/run_seeds.py`` / ``scripts/run_experiments.py``, which must
    calibrate every trained checkpoint into its own isolated
    ``calibration_dir`` before evaluating it with calibration applied.
    Returns None (after logging an error) if no prepared split or
    calibration split exists yet.
    """
    device = resolve_device("auto")
    model, config, _ = load_model_checkpoint(checkpoint_path, device)
    alpha = alpha if alpha is not None else config["calibration"]["alpha"]
    experiment_name = experiment_name or checkpoint_experiment_name(checkpoint_path)
    seed = seed if seed is not None else config.get("training", {}).get("seed", config.get("seed"))

    splits_path = REPO_ROOT / config["paths"]["splits_dir"] / "full_metadata_with_splits.csv"
    if not splits_path.exists():
        logger.error("No prepared split found at %s.", splits_path)
        return None
    df = pd.read_csv(splits_path)

    # Model-aware preprocessing: a pretrained-ResNet checkpoint's own
    # resolved transform (input size, mean/std, interpolation, crop_pct),
    # never this project's 128px/IMAGENET-constant default for such a model
    # -- see src/data/transforms.py::resolve_eval_transform.
    eval_transform = resolve_eval_transform(model, config)
    calibration_dataset = FaceMultiTaskDataset(df[df["split"] == "calibration"], eval_transform)
    if len(calibration_dataset) == 0:
        logger.error(
            "Calibration split is empty. Re-run 'make prepare-data' with the current "
            "4-way split config (configs/data.yaml: split.calibration_fraction) if this "
            "split was prepared before the calibration split existed."
        )
        return None
    q10_cal, q90_cal, ages_cal, mask_cal = _predict_age(model, calibration_dataset, device)
    if not mask_cal.any():
        logger.error("Calibration split has no age labels; cannot calibrate.")
        return None

    test_df = df[df["split"] == "test"]
    calibration_dir_resolved = REPO_ROOT / calibration_dir if calibration_dir else REPO_ROOT / config["calibration"]["output_dir"]
    preprocessing_fingerprint = compute_preprocessing_fingerprint(
        eval_transform.image_size, eval_transform.mean, eval_transform.std,
        eval_transform.interpolation, getattr(eval_transform, "crop_pct", 1.0),
    )
    artifact = fit_and_save_calibration(
        ages_cal[mask_cal], q10_cal[mask_cal], q90_cal[mask_cal], alpha, calibration_dir_resolved,
        checkpoint_path=checkpoint_path, split_csv_path=splits_path,
        test_sample_ids=test_df["image_path"].tolist(), experiment=experiment_name, seed=seed,
        model_id=getattr(model, "model_id", None), pretrained_source=getattr(model, "pretrained_source", None),
        preprocessing_fingerprint=preprocessing_fingerprint,
    )
    logger.info("Calibration artifact: %s", artifact)

    test_dataset = FaceMultiTaskDataset(test_df, eval_transform)
    q10_test, q90_test, ages_test, mask_test = _predict_age(model, test_dataset, device)
    if mask_test.any():
        effect = evaluate_calibration_effect(ages_test[mask_test], q10_test[mask_test], q90_test[mask_test], artifact["offset"])
        logger.info("Calibration effect on test set: %s", effect)
        save_json(effect, calibration_dir_resolved / "calibration_test_effect.json")

    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--alpha", type=float, default=None, help="Target miscoverage (default from configs/training.yaml)")
    parser.add_argument("--calibration-dir", default=None, help="Default: config['calibration']['output_dir']")
    parser.add_argument("--experiment-name", default=None, help="Default: derived from the checkpoint filename")
    parser.add_argument("--seed", type=int, default=None, help="Default: the seed recorded in the checkpoint's config")
    args = parser.parse_args()

    artifact = calibrate_checkpoint(
        args.checkpoint, args.calibration_dir, args.alpha, args.experiment_name, args.seed,
    )
    if artifact is None:
        return 1
    calibration_dir = args.calibration_dir or "the configured calibration output_dir"
    print(f"Saved calibration artifact to {calibration_dir}/conformal_calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
