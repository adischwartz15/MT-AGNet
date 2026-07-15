"""Split conformal calibration for age q10/q90 prediction intervals.

Uses the validation set only (never the test set) to learn a single
scalar interval-expansion offset via conformalized quantile regression
(CQR, Romano et al. 2019):

    score_i = max(q10_i - y_i, y_i - q90_i)
    offset  = the ceil((n+1)(1-alpha))/n empirical quantile of {score_i}
    calibrated interval = [q10 - offset, q90 + offset]

This guarantees (marginally, under exchangeability) that the calibrated
interval covers the true value with probability >= 1 - alpha on held-out
data drawn from the same distribution. Intervals are only ever described
as "calibrated" when a calibration artifact produced by this procedure
actually exists and loaded successfully.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from src.utils.io import file_sha256, load_json, save_json


class CalibrationMismatchError(RuntimeError):
    """Raised when a loaded calibration artifact's recorded provenance does
    not match the checkpoint / split file / test-sample-set it is about to
    be applied to (see :func:`validate_calibration_artifact`)."""


def compute_nonconformity_scores(y_true: np.ndarray, q10: np.ndarray, q90: np.ndarray) -> np.ndarray:
    return np.maximum(q10 - y_true, y_true - q90)


def compute_ordered_id_hash(ids: Sequence) -> str:
    """SHA-256 of an ordered sequence of sample identifiers (e.g. image paths).

    Used to detect when a calibration artifact is being applied to a test
    split whose row order differs from the one it was fit against --
    equal *counts* are not sufficient evidence the two enumerations are
    the same set in the same order.
    """
    hasher = hashlib.sha256()
    for sample_id in ids:
        hasher.update(str(sample_id).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def fit_conformal_offset(scores: np.ndarray, alpha: float = 0.10) -> float:
    """Compute the split-conformal offset for miscoverage level ``alpha``."""
    n = len(scores)
    if n == 0:
        raise ValueError("Cannot fit conformal calibration on an empty validation set")
    level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level))


def apply_conformal_offset(q10: np.ndarray, q90: np.ndarray, offset: float) -> tuple[np.ndarray, np.ndarray]:
    return q10 - offset, q90 + offset


def compute_preprocessing_fingerprint(
    input_size: int, mean: Sequence[float], std: Sequence[float], interpolation, crop_pct: float = 1.0,
) -> str:
    """SHA-256 fingerprint of the exact preprocessing a checkpoint's
    predictions were produced with -- input size, normalization mean/std,
    interpolation mode, and crop_pct. Recorded on a calibration artifact so
    a later mismatch (e.g. applying a calibration fit under one crop_pct to
    predictions produced under a different one) is caught explicitly,
    in addition to (not instead of) the checkpoint-hash check, which
    already implies this for any checkpoint this repository trained but is
    otherwise opaque to inspect.
    """
    payload = f"{input_size}|{list(mean)}|{list(std)}|{interpolation}|{crop_pct}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fit_and_save_calibration(
    y_true_val: np.ndarray,
    q10_val: np.ndarray,
    q90_val: np.ndarray,
    alpha: float,
    output_dir: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    split_csv_path: str | Path | None = None,
    test_sample_ids: Sequence | None = None,
    experiment: str | None = None,
    seed: int | None = None,
    model_id: str | None = None,
    pretrained_source: str | None = None,
    preprocessing_fingerprint: str | None = None,
) -> dict:
    """Fit conformal calibration on the validation set and save the artifact.

    ``checkpoint_path`` / ``split_csv_path`` / ``test_sample_ids`` /
    ``experiment`` / ``seed`` / ``model_id`` / ``pretrained_source`` /
    ``preprocessing_fingerprint`` are optional provenance recorded into the
    artifact so a later :func:`validate_calibration_artifact` call can
    detect (and fail loudly on) cross-seed/cross-model/cross-preprocessing
    contamination -- e.g. evaluating one checkpoint against a calibration
    artifact actually fit for a different checkpoint, a differently-ordered
    test split, or a different resolved preprocessing (input size/mean/std/
    interpolation/crop_pct). Omitting them keeps this function usable in
    contexts (tests, ad-hoc analysis) that don't have a real checkpoint/
    split file on disk; no validation is performed against fields that were
    never recorded.
    """
    scores = compute_nonconformity_scores(y_true_val, q10_val, q90_val)
    offset = fit_conformal_offset(scores, alpha)
    artifact = {
        "method": "split_conformal_cqr",
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "offset": offset,
        "n_calibration_samples": int(len(y_true_val)),
        "experiment": experiment,
        "seed": seed,
        "model_id": model_id,
        "pretrained_source": pretrained_source,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "checkpoint_sha256": file_sha256(checkpoint_path) if checkpoint_path is not None else None,
        "split_csv_sha256": file_sha256(split_csv_path) if split_csv_path is not None else None,
        "test_sample_id_hash": compute_ordered_id_hash(test_sample_ids) if test_sample_ids is not None else None,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(artifact, output_dir / "conformal_calibration.json")
    return artifact


def load_calibration(output_dir: str | Path) -> dict | None:
    """Load a previously saved calibration artifact, or None if it doesn't exist."""
    path = Path(output_dir) / "conformal_calibration.json"
    if not path.exists():
        return None
    return load_json(path)


def validate_calibration_artifact(
    artifact: dict,
    *,
    checkpoint_path: str | Path | None = None,
    split_csv_path: str | Path | None = None,
    test_sample_ids: Sequence | None = None,
    model_id: str | None = None,
    pretrained_source: str | None = None,
    preprocessing_fingerprint: str | None = None,
) -> None:
    """Fail loudly if ``artifact`` was fit against a different checkpoint, split
    file, ordered test-sample set, model identifier, pretrained source, or
    resolved preprocessing than the ones supplied here.

    This is what stops cross-seed/cross-model/cross-preprocessing
    calibration contamination: silently applying seed 42's (or model A's,
    or a different crop_pct's) conformal offset to seed 123's (or model
    B's) test predictions just because both calibration artifacts happen
    to live in a similarly-named directory or have the same array length.
    Only fields the artifact actually recorded are checked -- an older
    artifact fit before a given provenance field existed has it as
    ``None`` and is intentionally not validated against that field (there
    is nothing on disk yet to compare against).
    """
    checks: list[tuple[str, str | None, str]] = []
    if checkpoint_path is not None:
        checks.append(("checkpoint_sha256", file_sha256(checkpoint_path), "checkpoint"))
    if split_csv_path is not None:
        checks.append(("split_csv_sha256", file_sha256(split_csv_path), "split CSV"))
    if test_sample_ids is not None:
        checks.append(("test_sample_id_hash", compute_ordered_id_hash(test_sample_ids), "ordered test-sample IDs"))
    if model_id is not None:
        checks.append(("model_id", model_id, "model identifier"))
    if pretrained_source is not None:
        checks.append(("pretrained_source", pretrained_source, "pretrained source"))
    if preprocessing_fingerprint is not None:
        checks.append(("preprocessing_fingerprint", preprocessing_fingerprint, "resolved preprocessing"))

    for field, actual_value, label in checks:
        recorded_value = artifact.get(field)
        if recorded_value is None:
            continue  # artifact predates this provenance field -- nothing to compare
        if recorded_value != actual_value:
            raise CalibrationMismatchError(
                f"Calibration artifact mismatch on {label} (field '{field}'): this artifact was "
                f"fit against a different {label} than the one being evaluated now. Applying it "
                "would silently use the wrong conformal offset. Re-run scripts/calibrate.py "
                "against this exact checkpoint and split before evaluating with calibration."
            )


def evaluate_calibration_effect(
    y_true_test: np.ndarray, q10_test: np.ndarray, q90_test: np.ndarray, offset: float
) -> dict:
    """Report coverage/width before and after applying the conformal offset."""
    from src.evaluation.metrics import interval_coverage, mean_interval_width

    coverage_before = interval_coverage(y_true_test, q10_test, q90_test)
    width_before = mean_interval_width(q10_test, q90_test)

    q10_cal, q90_cal = apply_conformal_offset(q10_test, q90_test, offset)
    coverage_after = interval_coverage(y_true_test, q10_cal, q90_cal)
    width_after = mean_interval_width(q10_cal, q90_cal)

    return {
        "coverage_before_calibration": coverage_before,
        "coverage_after_calibration": coverage_after,
        "mean_width_before_calibration": width_before,
        "mean_width_after_calibration": width_after,
    }
