"""Tests for split conformal calibration of age prediction intervals."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.calibration import (
    CalibrationMismatchError, apply_conformal_offset, compute_nonconformity_scores, compute_ordered_id_hash,
    evaluate_calibration_effect, fit_and_save_calibration, fit_conformal_offset, load_calibration,
    validate_calibration_artifact,
)


def test_nonconformity_scores_zero_when_inside_interval():
    y = np.array([5.0])
    q10 = np.array([0.0])
    q90 = np.array([10.0])
    scores = compute_nonconformity_scores(y, q10, q90)
    assert scores[0] <= 0.0


def test_nonconformity_scores_positive_when_outside_interval():
    y = np.array([15.0])
    q10 = np.array([0.0])
    q90 = np.array([10.0])
    scores = compute_nonconformity_scores(y, q10, q90)
    assert scores[0] == 5.0


def test_fit_conformal_offset_widens_narrow_intervals():
    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.uniform(0, 100, n)
    # A noisy point estimate with a narrow fixed-width interval around it:
    # since the interval doesn't track the true value exactly, it misses
    # (undercovers) often enough for conformal calibration to need a
    # positive expansion offset.
    predicted_center = y_true + rng.normal(0, 3.0, n)
    q10 = predicted_center - 1.0
    q90 = predicted_center + 1.0
    scores = compute_nonconformity_scores(y_true, q10, q90)
    offset = fit_conformal_offset(scores, alpha=0.10)
    assert offset > 0

    q10_cal, q90_cal = apply_conformal_offset(q10, q90, offset)
    coverage_before = np.mean((y_true >= q10) & (y_true <= q90))
    coverage_after = np.mean((y_true >= q10_cal) & (y_true <= q90_cal))
    assert coverage_after >= coverage_before


def test_fit_and_save_and_load_calibration(tmp_path):
    rng = np.random.default_rng(1)
    n = 200
    y_true = rng.uniform(0, 100, n)
    q10 = y_true - 5
    q90 = y_true + 5
    artifact = fit_and_save_calibration(y_true, q10, q90, alpha=0.1, output_dir=tmp_path)
    assert (tmp_path / "conformal_calibration.json").exists()
    loaded = load_calibration(tmp_path)
    assert loaded is not None
    assert loaded["offset"] == artifact["offset"]
    assert loaded["target_coverage"] == 0.9


def test_load_calibration_returns_none_when_missing(tmp_path):
    assert load_calibration(tmp_path) is None


def test_evaluate_calibration_effect_reports_before_and_after(tmp_path):
    rng = np.random.default_rng(2)
    n = 300
    y_true = rng.uniform(0, 100, n)
    q10 = y_true - 2
    q90 = y_true + 2
    effect = evaluate_calibration_effect(y_true, q10, q90, offset=3.0)
    assert effect["mean_width_after_calibration"] > effect["mean_width_before_calibration"]
    assert effect["coverage_after_calibration"] >= effect["coverage_before_calibration"]


def test_compute_ordered_id_hash_is_order_sensitive():
    assert compute_ordered_id_hash(["a", "b", "c"]) != compute_ordered_id_hash(["c", "b", "a"])
    assert compute_ordered_id_hash(["a", "b", "c"]) == compute_ordered_id_hash(["a", "b", "c"])


def _fit_calibration_with_provenance(tmp_path, checkpoint_bytes, split_rows, experiment, seed, subdir):
    """Fit+save a calibration artifact with real provenance (a fake checkpoint
    file and a fake split CSV on disk, since fit_and_save_calibration hashes
    whatever file paths it is given)."""
    checkpoint_path = tmp_path / f"{experiment}.pt"
    checkpoint_path.write_bytes(checkpoint_bytes)
    split_csv_path = tmp_path / f"{experiment}_split.csv"
    split_csv_path.write_text(split_rows, encoding="utf-8")

    rng = np.random.default_rng(seed)
    n = 50
    y_true = rng.uniform(0, 100, n)
    q10, q90 = y_true - 5, y_true + 5
    output_dir = tmp_path / subdir
    artifact = fit_and_save_calibration(
        y_true, q10, q90, alpha=0.1, output_dir=output_dir,
        checkpoint_path=checkpoint_path, split_csv_path=split_csv_path,
        test_sample_ids=["img_0.jpg", "img_1.jpg", "img_2.jpg"],
        experiment=experiment, seed=seed,
    )
    return artifact, checkpoint_path, split_csv_path


def test_fit_and_save_calibration_records_full_provenance(tmp_path):
    artifact, checkpoint_path, split_csv_path = _fit_calibration_with_provenance(
        tmp_path, b"seed-42-checkpoint", "row1\nrow2\n", "exp_d_shared_adapters_learned_balance", 42, "cal_a",
    )
    assert artifact["experiment"] == "exp_d_shared_adapters_learned_balance"
    assert artifact["seed"] == 42
    assert artifact["alpha"] == 0.1
    assert artifact["target_coverage"] == 0.9
    assert artifact["checkpoint_sha256"] is not None
    assert artifact["split_csv_sha256"] is not None
    assert artifact["test_sample_id_hash"] == compute_ordered_id_hash(["img_0.jpg", "img_1.jpg", "img_2.jpg"])


def test_validate_calibration_artifact_passes_for_the_matching_checkpoint_and_split(tmp_path):
    artifact, checkpoint_path, split_csv_path = _fit_calibration_with_provenance(
        tmp_path, b"seed-42-checkpoint", "row1\nrow2\n", "exp_d_shared_adapters_learned_balance", 42, "cal_a",
    )
    validate_calibration_artifact(
        artifact, checkpoint_path=checkpoint_path, split_csv_path=split_csv_path,
        test_sample_ids=["img_0.jpg", "img_1.jpg", "img_2.jpg"],
    )  # must not raise


def test_validate_calibration_artifact_rejects_cross_seed_contamination(tmp_path):
    """Regression test: seed 42's calibration artifact must never be silently
    applied to seed 123's checkpoint, even if both are named/shaped identically."""
    artifact_seed42, _, _ = _fit_calibration_with_provenance(
        tmp_path, b"seed-42-checkpoint-bytes", "row1\nrow2\n", "exp_d_shared_adapters_learned_balance", 42, "cal_seed42",
    )
    # A different checkpoint (seed 123) evaluated against seed 42's artifact.
    other_checkpoint = tmp_path / "exp_d_seed123.pt"
    other_checkpoint.write_bytes(b"seed-123-checkpoint-bytes")

    with pytest.raises(CalibrationMismatchError):
        validate_calibration_artifact(artifact_seed42, checkpoint_path=other_checkpoint)


def test_validate_calibration_artifact_rejects_cross_model_contamination(tmp_path):
    """Regression test: a SimpleCNN checkpoint must never be silently
    evaluated with a calibration artifact actually fit for the ResNet
    checkpoint, even though both artifacts live under similarly-shaped
    per-experiment calibration directories."""
    resnet_artifact, _, resnet_split = _fit_calibration_with_provenance(
        tmp_path, b"resnet-checkpoint-bytes", "row1\nrow2\n", "exp_d_shared_adapters_learned_balance", 42, "cal_resnet",
    )
    simple_cnn_checkpoint = tmp_path / "exp_0_simple_cnn.pt"
    simple_cnn_checkpoint.write_bytes(b"simple-cnn-checkpoint-bytes")

    with pytest.raises(CalibrationMismatchError):
        validate_calibration_artifact(resnet_artifact, checkpoint_path=simple_cnn_checkpoint, split_csv_path=resnet_split)


def test_validate_calibration_artifact_rejects_reordered_test_sample_ids(tmp_path):
    """Equal sample *count* is not sufficient evidence of alignment -- a
    reordered (or differently filtered) test split must also be rejected."""
    artifact, checkpoint_path, split_csv_path = _fit_calibration_with_provenance(
        tmp_path, b"seed-42-checkpoint", "row1\nrow2\n", "exp_d_shared_adapters_learned_balance", 42, "cal_order",
    )
    with pytest.raises(CalibrationMismatchError):
        validate_calibration_artifact(
            artifact, checkpoint_path=checkpoint_path, split_csv_path=split_csv_path,
            test_sample_ids=["img_1.jpg", "img_0.jpg", "img_2.jpg"],  # same set, reordered
        )


def test_validate_calibration_artifact_is_a_noop_for_artifacts_without_provenance(tmp_path):
    """Older artifacts fit before provenance existed (no recorded hashes)
    must not be rejected outright -- there is nothing on disk to compare
    against, so validation is skipped for those fields."""
    rng = np.random.default_rng(0)
    n = 50
    y_true = rng.uniform(0, 100, n)
    legacy_artifact = fit_and_save_calibration(y_true, y_true - 5, y_true + 5, alpha=0.1, output_dir=tmp_path / "legacy")
    checkpoint_path = tmp_path / "some_checkpoint.pt"
    checkpoint_path.write_bytes(b"anything")

    validate_calibration_artifact(legacy_artifact, checkpoint_path=checkpoint_path)  # must not raise
