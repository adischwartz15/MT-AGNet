"""Tests for scripts/evaluate.py's compute_parametric_metrics wiring.

Guards against a regression where the module docstring described
per-bucket uncertainty metrics but the function body still called a
now-removed helper (age_error_by_bucket) that wasn't imported -- the kind
of inconsistency that only surfaces at runtime, not at import time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate import compute_parametric_metrics  # noqa: E402


def _synthetic_preds(n=40, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.uniform(0, 80, size=n)
    q50 = age + rng.normal(0, 2, size=n)
    q10 = q50 - rng.uniform(5, 10, size=n)
    q90 = q50 + rng.uniform(5, 10, size=n)
    probs = rng.dirichlet([1, 1], size=n)
    return {
        "q10": q10, "q50": q50, "q90": q90, "probs": probs,
        "age": age, "age_mask": np.ones(n, dtype=bool),
        "gender": rng.integers(0, 2, size=n), "gender_mask": np.ones(n, dtype=bool),
        "latency_ms_per_image": 1.23,
    }


def test_compute_parametric_metrics_includes_age_metrics_by_bucket():
    preds = _synthetic_preds()
    metrics = compute_parametric_metrics(preds, confidence_threshold=0.80, calibration=None)
    assert "age_metrics_by_bucket" in metrics
    assert "age_error_by_bucket" not in metrics
    bucket = metrics["age_metrics_by_bucket"]
    assert any(v["count"] > 0 for v in bucket.values())
    for label, stats in bucket.items():
        if stats["count"] > 0:
            assert stats["mae"] is not None
            assert stats["coverage"] is not None
            assert stats["mean_width"] is not None


def test_compute_parametric_metrics_adds_calibrated_bucket_metrics_when_calibration_present():
    preds = _synthetic_preds()
    calibration = {"offset": 1.5}
    metrics = compute_parametric_metrics(preds, confidence_threshold=0.80, calibration=calibration)
    assert "age_metrics_by_bucket_calibrated" in metrics
    assert "interval_coverage_calibrated" in metrics
    assert "mean_interval_width_calibrated" in metrics
    # A positive offset widens intervals, which can only raise or hold coverage.
    assert metrics["mean_interval_width_calibrated"] > metrics["mean_interval_width"]


def test_compute_parametric_metrics_omits_calibrated_keys_when_calibration_absent():
    preds = _synthetic_preds()
    metrics = compute_parametric_metrics(preds, confidence_threshold=0.80, calibration=None)
    assert "age_metrics_by_bucket_calibrated" not in metrics
    assert "interval_coverage_calibrated" not in metrics


import copy  # noqa: E402

import torch  # noqa: E402

import evaluate as evaluate_module  # noqa: E402
from evaluate import evaluate_checkpoint  # noqa: E402

from src.data.dataset import FaceMultiTaskDataset  # noqa: E402
from src.data.split_utils import split_dataframe  # noqa: E402
from src.data.transforms import EvalTransform, TrainTransform  # noqa: E402
from src.evaluation.calibration import CalibrationMismatchError, fit_and_save_calibration  # noqa: E402
from src.evaluation.knn_baseline import KNNEmbeddingBaseline  # noqa: E402
from src.models.multitask_model import build_multitask_model  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402


def _train_and_prepare_one_experiment(tmp_path, synthetic_metadata_df, tiny_config, experiment_name, seed):
    """Train a tiny real checkpoint end-to-end, plus a matching k-NN index and
    conformal calibration artifact, all isolated under tmp_path/experiment_name."""
    df = split_dataframe(synthetic_metadata_df, 0.4, 0.2, 0.2, 0.2, seed=seed, subject_level_if_available=False)
    exp_root = tmp_path / experiment_name
    splits_dir = exp_root / "splits"
    splits_dir.mkdir(parents=True)
    df.to_csv(splits_dir / "full_metadata_with_splits.csv", index=False)

    config = copy.deepcopy(tiny_config)
    output_dir = exp_root / "output"
    checkpoint_dir = exp_root / "checkpoints"
    config["paths"]["splits_dir"] = str(splits_dir)
    config["paths"]["output_dir"] = str(output_dir)
    config["paths"]["checkpoint_dir"] = str(checkpoint_dir)
    config["calibration"]["output_dir"] = str(exp_root / "calibration")
    config["knn"]["index_dir"] = str(exp_root / "knn")

    image_size = config["dataset"]["image_size"]
    train_dataset = FaceMultiTaskDataset(df[df["split"] == "train"], TrainTransform(image_size))
    val_dataset = FaceMultiTaskDataset(df[df["split"] == "validation"], EvalTransform(image_size))

    model = build_multitask_model(config)
    trainer = Trainer(
        model, config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=checkpoint_dir, experiment_name=experiment_name,
    )
    trainer.train()
    checkpoint_path = checkpoint_dir / f"{experiment_name}_best_balanced_score.pt"
    assert checkpoint_path.exists()

    # Fit a real (if tiny) k-NN index over the train split's embeddings.
    calibration_dataset = FaceMultiTaskDataset(df[df["split"] == "calibration"], EvalTransform(image_size))
    train_eval_dataset = FaceMultiTaskDataset(df[df["split"] == "train"], EvalTransform(image_size))
    embeds, ages, age_masks, genders, gender_masks = [], [], [], [], []
    with torch.no_grad():
        for i in range(len(train_eval_dataset)):
            row = train_eval_dataset[i]
            emb = model.encode(row["image"].unsqueeze(0))["age_embedding"].numpy()[0]
            embeds.append(emb)
            ages.append(row["age"].item())
            age_masks.append(bool(row["age_mask"].item()))
            genders.append(row["gender_label"].item())
            gender_masks.append(bool(row["gender_mask"].item()))

    knn_baseline = KNNEmbeddingBaseline(k=3)
    knn_baseline.fit(
        np.array(embeds), np.array(ages), np.array(age_masks),
        np.array(genders), np.array(gender_masks), num_classes=2,
    )
    knn_path = exp_root / "knn" / "knn_baseline.pkl"
    knn_baseline.save(knn_path)

    # Fit a real conformal calibration artifact (alpha=0.10 -> target_coverage=0.90),
    # with full provenance so cross-experiment contamination can be detected.
    cal_q10, cal_q90, cal_ages = [], [], []
    with torch.no_grad():
        for i in range(len(calibration_dataset)):
            row = calibration_dataset[i]
            out = model(row["image"].unsqueeze(0))["age_output"]
            cal_q10.append(out["q10"].item())
            cal_q90.append(out["q90"].item())
            cal_ages.append(row["age"].item())
    test_df = df[df["split"] == "test"]
    fit_and_save_calibration(
        np.array(cal_ages), np.array(cal_q10), np.array(cal_q90),
        alpha=0.10, output_dir=exp_root / "calibration",
        checkpoint_path=checkpoint_path, split_csv_path=splits_dir / "full_metadata_with_splits.csv",
        test_sample_ids=test_df["image_path"].tolist(), experiment=experiment_name, seed=seed,
    )

    return checkpoint_path, knn_path


def test_evaluate_checkpoint_knn_artifacts_isolated_across_experiments(tmp_path, synthetic_metadata_df, tiny_config):
    """Regression test: two experiments evaluated with --compare-knn must
    write to distinct, isolated k-NN comparison table paths -- never a
    single shared global outputs/knn/parametric_vs_knn.csv that the second
    experiment's evaluation would silently overwrite."""
    checkpoint_a, knn_a = _train_and_prepare_one_experiment(tmp_path, synthetic_metadata_df, tiny_config, "exp_a", seed=1)
    checkpoint_b, knn_b = _train_and_prepare_one_experiment(tmp_path, synthetic_metadata_df, tiny_config, "exp_b", seed=2)

    metrics_a = evaluate_checkpoint(
        str(checkpoint_a), output_name="exp_a_test_metrics", compare_knn=True,
        knn_path=str(knn_a), calibration_dir=str(tmp_path / "exp_a" / "calibration"),
    )
    metrics_b = evaluate_checkpoint(
        str(checkpoint_b), output_name="exp_b_test_metrics", compare_knn=True,
        knn_path=str(knn_b), calibration_dir=str(tmp_path / "exp_b" / "calibration"),
    )

    path_a = Path(metrics_a["knn_comparison_table_path"])
    path_b = Path(metrics_b["knn_comparison_table_path"])
    assert path_a.exists()
    assert path_b.exists()
    assert path_a != path_b
    assert path_a.parent != path_b.parent


def test_evaluate_checkpoint_uses_calibration_target_coverage_not_hardcoded(
    tmp_path, synthetic_metadata_df, tiny_config, monkeypatch,
):
    """Regression test: the calibrated coverage-width tradeoff plot's target
    line must come from the calibration artifact's own target_coverage
    (1 - alpha), not a hardcoded 0.80 -- alpha=0.10 here means 0.90."""
    checkpoint_path, _ = _train_and_prepare_one_experiment(
        tmp_path, synthetic_metadata_df, tiny_config, "exp_target", seed=3
    )

    captured = {}
    original = evaluate_module.plot_coverage_width_tradeoff

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(evaluate_module, "plot_coverage_width_tradeoff", _capture)

    evaluate_checkpoint(
        str(checkpoint_path), output_name="exp_target_test_metrics",
        calibration_dir=str(tmp_path / "exp_target" / "calibration"),
    )

    assert captured.get("target_coverage") == 0.90


def test_evaluate_checkpoint_rejects_cross_experiment_calibration_contamination(
    tmp_path, synthetic_metadata_df, tiny_config,
):
    """Regression test: evaluating experiment A's checkpoint against
    experiment B's calibration artifact (e.g. from an accidental shared
    calibration_dir, or copy-paste of the wrong path) must fail loudly
    instead of silently applying the wrong conformal offset."""
    checkpoint_a, _ = _train_and_prepare_one_experiment(tmp_path, synthetic_metadata_df, tiny_config, "exp_cross_a", seed=11)
    checkpoint_b, _ = _train_and_prepare_one_experiment(tmp_path, synthetic_metadata_df, tiny_config, "exp_cross_b", seed=22)

    with pytest.raises(CalibrationMismatchError):
        evaluate_checkpoint(
            str(checkpoint_a), output_name="exp_cross_a_test_metrics",
            calibration_dir=str(tmp_path / "exp_cross_b" / "calibration"),
        )

    # The matching calibration_dir must still work (sanity check that the
    # rejection above is really about the mismatch, not a broken checkpoint/split).
    metrics = evaluate_checkpoint(
        str(checkpoint_b), output_name="exp_cross_b_test_metrics",
        calibration_dir=str(tmp_path / "exp_cross_b" / "calibration"),
    )
    assert metrics is not None


def test_evaluate_checkpoint_rejects_stale_calibration_after_retraining(
    tmp_path, synthetic_metadata_df, tiny_config,
):
    """Regression test: retraining the same experiment/seed (new random
    init, e.g. after a code change) produces a new checkpoint whose weights
    differ from the one the old calibration.json on disk was fit against.
    Evaluating with the stale calibration artifact must fail loudly rather
    than silently reuse an offset fit for different model weights."""
    checkpoint_v1, _ = _train_and_prepare_one_experiment(tmp_path, synthetic_metadata_df, tiny_config, "exp_retrain", seed=7)
    stale_calibration_dir = tmp_path / "exp_retrain" / "calibration"

    # Simulate retraining in-place: overwrite the checkpoint file's bytes
    # without re-running scripts/calibrate.py.
    checkpoint_v1.write_bytes(checkpoint_v1.read_bytes() + b"\x00")

    with pytest.raises(CalibrationMismatchError):
        evaluate_checkpoint(
            str(checkpoint_v1), output_name="exp_retrain_test_metrics",
            calibration_dir=str(stale_calibration_dir),
        )
