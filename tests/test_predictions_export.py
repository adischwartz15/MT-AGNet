"""Tests for src/evaluation/predictions.py (T5) -- per-sample prediction
export and the proof that recomputed aggregate metrics from the saved file
match scripts/evaluate.py::compute_parametric_metrics's reported numbers.

Synthetic prediction arrays only -- no real model, no real data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = str(REPO_ROOT / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from src.evaluation.predictions import (
    PREDICTION_COLUMNS,
    build_predictions_dataframe,
    export_predictions,
    recompute_aggregate_metrics_from_predictions,
)


def _synthetic_preds(n=50, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.uniform(0, 90, size=n)
    q50 = age + rng.normal(0, 3, size=n)
    q10 = q50 - np.abs(rng.normal(5, 1, size=n))
    q90 = q50 + np.abs(rng.normal(5, 1, size=n))
    age_mask = rng.random(n) > 0.1
    gender = rng.integers(0, 2, size=n)
    gender_mask = rng.random(n) > 0.1
    logits = rng.normal(0, 2, size=(n, 2))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    sample_id = np.array([f"img_{i}.jpg" for i in range(n)])
    return {
        "q10": q10, "q50": q50, "q90": q90, "probs": probs,
        "age": age, "age_mask": age_mask.astype(float),
        "gender": gender.astype(float), "gender_mask": gender_mask.astype(float),
        "sample_id": sample_id, "latency_ms_per_image": 1.0,
    }


# -- build_predictions_dataframe -----------------------------------------------------


def test_dataframe_has_all_required_columns():
    preds = _synthetic_preds()
    df = build_predictions_dataframe(preds)
    assert list(df.columns) == PREDICTION_COLUMNS


def test_nan_for_rows_outside_age_mask():
    preds = _synthetic_preds()
    df = build_predictions_dataframe(preds)
    masked_out = df[~df["age_mask"]]
    assert masked_out["true_age"].isna().all()
    assert masked_out["absolute_age_error"].isna().all()
    assert masked_out["squared_age_error"].isna().all()


def test_nan_for_rows_outside_gender_mask():
    preds = _synthetic_preds()
    df = build_predictions_dataframe(preds)
    masked_out = df[~df["gender_mask"]]
    assert masked_out["true_gender_label"].isna().all()
    assert masked_out["gender_correct"].isna().all()


def test_calibrated_columns_all_nan_without_calibration():
    preds = _synthetic_preds()
    df = build_predictions_dataframe(preds, calibration=None)
    assert df["calibrated_q10"].isna().all()
    assert df["calibrated_interval_contains_target"].isna().all()


def test_calibrated_columns_populated_with_calibration():
    preds = _synthetic_preds()
    calibration = {"offset": 2.0}
    df = build_predictions_dataframe(preds, calibration=calibration)
    age_rows = df[df["age_mask"]]
    assert not age_rows["calibrated_q10"].isna().any()
    assert (age_rows["calibrated_q10"] == age_rows["pred_q10"] - 2.0).all()
    assert (age_rows["calibrated_q90"] == age_rows["pred_q90"] + 2.0).all()


def test_no_second_inference_pass_columns_derived_purely_from_preds():
    """Regression guard: build_predictions_dataframe must be a pure function
    of `preds` (no model/dataset argument at all) -- structurally impossible
    to accidentally re-run inference."""
    import inspect

    sig = inspect.signature(build_predictions_dataframe)
    assert "model" not in sig.parameters
    assert "dataset" not in sig.parameters


# -- export_predictions (atomic write + manifest) -------------------------------------


def test_export_predictions_writes_csv_atomically(tmp_path):
    preds = _synthetic_preds()
    out_path = tmp_path / "predictions.csv"
    export_predictions(preds, out_path)
    assert out_path.exists()
    assert not out_path.with_suffix(".csv.tmp").exists()


def test_export_predictions_writes_sidecar_manifest(tmp_path):
    preds = _synthetic_preds()
    out_path = tmp_path / "predictions.csv"
    manifest_path = tmp_path / "predictions_manifest.json"
    export_predictions(
        preds, out_path, manifest_path=manifest_path,
        provenance={"experiment": "test_exp", "checkpoint_sha256": "abc123"},
    )
    from src.utils.io import load_json

    manifest = load_json(manifest_path)
    assert manifest["experiment"] == "test_exp"
    assert manifest["checkpoint_sha256"] == "abc123"
    assert "predictions_csv_sha256" in manifest
    assert manifest["n_rows"] == 50


# -- recompute_aggregate_metrics_from_predictions: parity with compute_parametric_metrics --


def test_recomputed_metrics_match_compute_parametric_metrics():
    """The core proof requirement: aggregate metrics recomputed from the
    saved predictions table must equal (within floating-point tolerance)
    the metrics scripts/evaluate.py::compute_parametric_metrics reports
    from the same underlying preds dict."""
    from evaluate import compute_parametric_metrics

    preds = _synthetic_preds(n=200, seed=42)
    confidence_threshold = 0.6

    original_metrics = compute_parametric_metrics(preds, confidence_threshold, calibration=None)
    df = build_predictions_dataframe(preds, calibration=None, confidence_threshold=confidence_threshold)
    recomputed = recompute_aggregate_metrics_from_predictions(df)

    assert recomputed["age_mae"] == pytest.approx(original_metrics["age_mae"], abs=1e-9)
    assert recomputed["age_rmse"] == pytest.approx(original_metrics["age_rmse"], abs=1e-9)
    assert recomputed["interval_coverage"] == pytest.approx(original_metrics["interval_coverage"], abs=1e-9)
    assert recomputed["mean_interval_width"] == pytest.approx(original_metrics["mean_interval_width"], abs=1e-9)
    assert recomputed["abstention_rate"] == pytest.approx(original_metrics["abstention_rate"], abs=1e-9)

    # gender_accuracy is NaN-safe compare (both may be NaN if all abstained; here shouldn't be)
    if original_metrics["gender_accuracy"] == original_metrics["gender_accuracy"]:
        assert recomputed["gender_accuracy"] == pytest.approx(original_metrics["gender_accuracy"], abs=1e-9)


def test_recomputed_metrics_match_with_calibration():
    from evaluate import compute_parametric_metrics

    preds = _synthetic_preds(n=150, seed=7)
    calibration = {"offset": 1.5}
    original_metrics = compute_parametric_metrics(preds, 0.8, calibration)
    df = build_predictions_dataframe(preds, calibration=calibration, confidence_threshold=0.8)
    recomputed = recompute_aggregate_metrics_from_predictions(df)

    assert recomputed["interval_coverage_calibrated"] == pytest.approx(
        original_metrics["interval_coverage_calibrated"], abs=1e-9
    )
    assert recomputed["mean_interval_width_calibrated"] == pytest.approx(
        original_metrics["mean_interval_width_calibrated"], abs=1e-9
    )


def test_gender_balanced_accuracy_from_saved_predictions_matches():
    from evaluate import compute_parametric_metrics
    from src.evaluation.metrics import gender_balanced_accuracy

    preds = _synthetic_preds(n=100, seed=3)
    original_metrics = compute_parametric_metrics(preds, 0.5, calibration=None)
    df = build_predictions_dataframe(preds, calibration=None, confidence_threshold=0.5)
    recomputed = recompute_aggregate_metrics_from_predictions(df)

    inputs = recomputed["gender_balanced_accuracy_inputs"]
    recomputed_balanced_acc = gender_balanced_accuracy(np.array(inputs["y_true"]), np.array(inputs["y_pred"]))
    assert recomputed_balanced_acc == pytest.approx(original_metrics["gender_balanced_accuracy"], abs=1e-9)
