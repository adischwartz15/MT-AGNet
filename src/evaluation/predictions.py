"""Per-sample prediction export -- the exact prediction arrays used to
compute a checkpoint's aggregate test metrics, persisted to disk so:

* aggregate metrics can be recomputed and cross-checked from the saved file
  (see tests/test_predictions_export.py) instead of only ever trusted as a
  single reported number;
* paired statistical comparisons (bootstrap CIs, McNemar's test) have an
  exact, ordered, sample-identified source to align against;
* a later re-analysis (subgroup breakdown, a new metric) never needs a
  second inference pass through the model.

Never re-runs inference: every function here operates on the same
prediction arrays ``scripts/evaluate.py::run_inference`` already produced
for the aggregate metrics, so the exported file and the reported numbers
are guaranteed to come from the identical forward pass.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.calibration import apply_conformal_offset
from src.utils.io import file_sha256, save_json

PREDICTION_COLUMNS = [
    "sample_id", "image_path", "split",
    "age_mask", "true_age", "pred_q10", "pred_q50", "pred_q90",
    "absolute_age_error", "squared_age_error",
    "raw_interval_width", "raw_interval_contains_target",
    "calibrated_q10", "calibrated_q90", "calibrated_interval_width", "calibrated_interval_contains_target",
    "gender_mask", "true_gender_label", "probability_class_0", "probability_class_1",
    "predicted_gender_label", "gender_confidence", "gender_abstained", "gender_correct",
]


def build_predictions_dataframe(
    preds: dict[str, np.ndarray],
    split: str = "test",
    calibration: dict | None = None,
    confidence_threshold: float = 0.80,
) -> pd.DataFrame:
    """Build the per-sample prediction table from an already-computed
    ``preds`` dict (``scripts/evaluate.py::run_inference``'s return value).

    Uses NaN (never a fabricated 0 or an arbitrary sentinel) for any column
    that doesn't apply to a given row -- e.g. every age-related column for a
    row with ``age_mask == False``, and every calibrated-interval column
    when no calibration artifact was supplied at all.
    """
    n = len(preds["age"])
    age_mask = preds["age_mask"].astype(bool)
    gender_mask = preds["gender_mask"].astype(bool)

    image_path = preds.get("sample_id")
    sample_id = np.arange(n)

    q10, q50, q90 = preds["q10"], preds["q50"], preds["q90"]
    true_age = preds["age"]

    absolute_age_error = np.where(age_mask, np.abs(q50 - true_age), np.nan)
    squared_age_error = np.where(age_mask, (q50 - true_age) ** 2, np.nan)
    raw_interval_width = q90 - q10
    raw_contains = np.where(age_mask, ((true_age >= q10) & (true_age <= q90)).astype(float), np.nan)

    calibrated_q10 = np.full(n, np.nan)
    calibrated_q90 = np.full(n, np.nan)
    calibrated_width = np.full(n, np.nan)
    calibrated_contains = np.full(n, np.nan)
    if calibration is not None:
        offset = calibration["offset"]
        calibrated_q10, calibrated_q90 = apply_conformal_offset(q10, q90, offset)
        calibrated_width = calibrated_q90 - calibrated_q10
        calibrated_contains = np.where(
            age_mask, ((true_age >= calibrated_q10) & (true_age <= calibrated_q90)).astype(float), np.nan,
        )

    probs = preds["probs"]
    predicted_gender = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    abstained = confidence < confidence_threshold
    gender_correct = np.where(
        gender_mask, (predicted_gender == preds["gender"].astype(int)).astype(float), np.nan,
    )

    df = pd.DataFrame({
        "sample_id": sample_id,
        "image_path": image_path if image_path is not None else [None] * n,
        "split": split,
        "age_mask": age_mask,
        "true_age": np.where(age_mask, true_age, np.nan),
        "pred_q10": q10, "pred_q50": q50, "pred_q90": q90,
        "absolute_age_error": absolute_age_error, "squared_age_error": squared_age_error,
        "raw_interval_width": raw_interval_width, "raw_interval_contains_target": raw_contains,
        "calibrated_q10": calibrated_q10, "calibrated_q90": calibrated_q90,
        "calibrated_interval_width": calibrated_width, "calibrated_interval_contains_target": calibrated_contains,
        "gender_mask": gender_mask,
        "true_gender_label": np.where(gender_mask, preds["gender"], np.nan),
        "probability_class_0": probs[:, 0],
        "probability_class_1": probs[:, 1] if probs.shape[1] > 1 else np.nan,
        "predicted_gender_label": predicted_gender,
        "gender_confidence": confidence,
        "gender_abstained": abstained,
        "gender_correct": gender_correct,
    })
    return df[PREDICTION_COLUMNS]


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def export_predictions(
    preds: dict[str, np.ndarray],
    output_path: str | Path,
    split: str = "test",
    calibration: dict | None = None,
    confidence_threshold: float = 0.80,
    manifest_path: str | Path | None = None,
    provenance: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build and atomically write the per-sample prediction table, plus an
    optional sidecar provenance manifest recording exactly what produced it
    (checkpoint/split/calibration hashes, model identity, preprocessing,
    confidence threshold, git commit, dependency versions, timestamp).

    Returns the built DataFrame (so a caller can also use it in-process,
    e.g. to recompute aggregate metrics, without re-reading the file).
    """
    output_path = Path(output_path)
    df = build_predictions_dataframe(preds, split=split, calibration=calibration, confidence_threshold=confidence_threshold)
    _atomic_write_csv(df, output_path)

    if manifest_path is not None:
        manifest = dict(provenance or {})
        manifest["predictions_csv_sha256"] = file_sha256(output_path)
        manifest["n_rows"] = len(df)
        manifest["confidence_threshold"] = confidence_threshold
        manifest_path = Path(manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        save_json(manifest, tmp_manifest)
        os.replace(tmp_manifest, manifest_path)

    return df


def recompute_aggregate_metrics_from_predictions(predictions_df: pd.DataFrame) -> dict:
    """Recompute the same headline aggregate metrics
    ``scripts/evaluate.py::compute_parametric_metrics`` reports, but purely
    from a saved predictions table -- used to prove the exported file
    actually reproduces the reported numbers (see
    tests/test_predictions_export.py), and reusable for any later
    re-analysis that shouldn't need a second inference pass.
    """
    age_rows = predictions_df[predictions_df["age_mask"].astype(bool)]
    gender_rows = predictions_df[predictions_df["gender_mask"].astype(bool)]

    metrics: dict[str, Any] = {}
    if len(age_rows) > 0:
        metrics["age_mae"] = float(age_rows["absolute_age_error"].mean())
        metrics["age_rmse"] = float(np.sqrt(age_rows["squared_age_error"].mean()))
        metrics["interval_coverage"] = float(age_rows["raw_interval_contains_target"].mean())
        metrics["mean_interval_width"] = float(age_rows["raw_interval_width"].mean())
        calibrated_rows = age_rows.dropna(subset=["calibrated_interval_contains_target"])
        if len(calibrated_rows) > 0:
            metrics["interval_coverage_calibrated"] = float(calibrated_rows["calibrated_interval_contains_target"].mean())
            metrics["mean_interval_width_calibrated"] = float(calibrated_rows["calibrated_interval_width"].mean())

    if len(gender_rows) > 0:
        accepted = gender_rows[~gender_rows["gender_abstained"].astype(bool)]
        metrics["gender_accuracy"] = (
            float(accepted["gender_correct"].mean()) if len(accepted) > 0 else float("nan")
        )
        metrics["abstention_rate"] = float(gender_rows["gender_abstained"].astype(bool).mean())
        metrics["gender_balanced_accuracy_inputs"] = {
            "y_true": gender_rows["true_gender_label"].astype(int).tolist(),
            "y_pred": gender_rows["predicted_gender_label"].astype(int).tolist(),
        }

    return metrics
