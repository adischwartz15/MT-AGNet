"""Tests for live training observability: incremental history.csv/json and
the atomic live-status file written after every epoch (src/training/trainer.py).

These exist so a notebook (or any external process) can inspect training
progress -- or recover a partial run's history -- without waiting for
Trainer.train() to return, which matters for long-running Colab/Kaggle
sessions that can be interrupted mid-training.
"""

from __future__ import annotations

import csv
import json

from src.data.dataset import FaceMultiTaskDataset
from src.data.split_utils import split_dataframe
from src.data.transforms import EvalTransform, TrainTransform
from src.models.multitask_model import build_multitask_model
from src.training.trainer import Trainer


def _build_datasets(synthetic_metadata_df, tiny_config, seed=0):
    df = split_dataframe(synthetic_metadata_df, 0.5, 0.2, 0.1, 0.2, seed=seed, subject_level_if_available=False)
    image_size = tiny_config["dataset"]["image_size"]
    train_dataset = FaceMultiTaskDataset(df[df["split"] == "train"], TrainTransform(image_size))
    val_dataset = FaceMultiTaskDataset(df[df["split"] == "validation"], EvalTransform(image_size))
    return train_dataset, val_dataset


def test_trainer_writes_incremental_history_csv_and_json(tmp_path, synthetic_metadata_df, tiny_config):
    tiny_config["training"]["warm_up_from_scratch"]["epochs"] = 2
    train_dataset, val_dataset = _build_datasets(synthetic_metadata_df, tiny_config)

    model = build_multitask_model(tiny_config)
    output_dir = tmp_path / "output"
    trainer = Trainer(
        model, tiny_config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="obs_test", output_dir=output_dir,
    )
    result = trainer.train()

    history_csv = output_dir / "metrics" / "obs_test_history.csv"
    history_json = output_dir / "metrics" / "obs_test_history.json"
    assert history_csv.exists()
    assert history_json.exists()

    with open(history_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    n_epochs = len(result["history"]["train_loss"])
    assert len(rows) == n_epochs + 1  # header + one row per epoch
    assert "val_age_rmse" in rows[0]
    assert "val_gender_selective_accuracy" in rows[0]
    assert "lr" in rows[0]

    with open(history_json, encoding="utf-8") as fh:
        history_from_disk = json.load(fh)
    # Compare via serialized form rather than dict `==`: NaN != NaN under
    # normal equality, and this history legitimately contains NaN entries
    # (e.g. gender metrics before any sample clears the confidence threshold).
    assert json.dumps(history_from_disk, sort_keys=True) == json.dumps(result["history"], sort_keys=True)


def test_trainer_writes_atomic_status_file_with_expected_fields(tmp_path, synthetic_metadata_df, tiny_config):
    tiny_config["training"]["warm_up_from_scratch"]["epochs"] = 1
    train_dataset, val_dataset = _build_datasets(synthetic_metadata_df, tiny_config, seed=1)

    model = build_multitask_model(tiny_config)
    output_dir = tmp_path / "output"
    trainer = Trainer(
        model, tiny_config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="status_test", output_dir=output_dir,
    )
    trainer.train()

    status_path = output_dir / "logs" / "status_test_status.json"
    assert status_path.exists()
    # The atomic write-then-rename must never leave a stray .tmp file behind.
    assert not status_path.with_suffix(".json.tmp").exists()

    status = json.loads(status_path.read_text(encoding="utf-8"))
    for key in ("experiment_name", "stage", "epoch", "total_epochs_planned", "best_scores", "early_stopping_bad_epochs", "early_stopping_patience", "updated_at_utc"):
        assert key in status
    assert status["experiment_name"] == "status_test"
    assert status["epoch"] == 1
    assert status["total_epochs_planned"] == 1


def test_maybe_checkpoint_returns_true_only_on_improvement(tmp_path, synthetic_metadata_df, tiny_config):
    train_dataset, val_dataset = _build_datasets(synthetic_metadata_df, tiny_config, seed=2)
    model = build_multitask_model(tiny_config)
    trainer = Trainer(
        model, tiny_config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="tracker_test", output_dir=tmp_path / "output",
    )

    assert trainer._maybe_checkpoint("age_mae", 10.0, epoch=1, metrics={}) is True  # first value is always "best"
    assert trainer._maybe_checkpoint("age_mae", 12.0, epoch=2, metrics={}) is False  # worse (mode="min")
    assert trainer._maybe_checkpoint("age_mae", 8.0, epoch=3, metrics={}) is True  # better
    assert trainer._maybe_checkpoint("age_mae", float("nan"), epoch=4, metrics={}) is False  # NaN never checkpoints


def test_max_batches_per_epoch_caps_training_and_validation_batches(tmp_path, synthetic_metadata_df, tiny_config):
    """Regression test for the smoke-mode speed cap: with
    max_train_batches_per_epoch / max_val_batches_per_epoch set, a run must
    still complete correctly (checkpoints/history written) even though it
    only ever sees a handful of batches per epoch -- this is what lets a
    "smoke test" validate the pipeline quickly on a large dataset instead of
    still iterating it fully once per epoch just because epochs=1."""
    tiny_config["training"]["warm_up_from_scratch"]["epochs"] = 1
    tiny_config["training"]["max_train_batches_per_epoch"] = 1
    tiny_config["training"]["max_val_batches_per_epoch"] = 1
    tiny_config["training"]["batch_size"] = 2  # small enough that "1 batch" << full dataset
    train_dataset, val_dataset = _build_datasets(synthetic_metadata_df, tiny_config, seed=4)

    model = build_multitask_model(tiny_config)
    output_dir = tmp_path / "output"
    trainer = Trainer(
        model, tiny_config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="capped_test", output_dir=output_dir,
    )
    assert trainer.max_train_batches == 1
    assert trainer.max_val_batches == 1
    trainer.train()  # must not raise despite the tiny per-epoch batch cap

    assert (output_dir / "metrics" / "capped_test_history.csv").exists()


def test_max_batches_per_epoch_defaults_to_unlimited(tmp_path, synthetic_metadata_df, tiny_config):
    train_dataset, val_dataset = _build_datasets(synthetic_metadata_df, tiny_config, seed=5)
    model = build_multitask_model(tiny_config)
    trainer = Trainer(
        model, tiny_config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="uncapped_test", output_dir=tmp_path / "output",
    )
    assert trainer.max_train_batches is None
    assert trainer.max_val_batches is None


def test_trainer_writes_run_manifest_once_at_start(tmp_path, synthetic_metadata_df, tiny_config):
    tiny_config["training"]["warm_up_from_scratch"]["epochs"] = 2
    train_dataset, val_dataset = _build_datasets(synthetic_metadata_df, tiny_config, seed=6)
    model = build_multitask_model(tiny_config)
    output_dir = tmp_path / "output"
    trainer = Trainer(
        model, tiny_config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="manifest_test", output_dir=output_dir,
    )
    trainer.train()

    manifest_path = output_dir / "logs" / "manifest_test_run_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in (
        "experiment_name", "seed", "device", "train_samples", "val_samples", "trainable_params",
        "total_params", "stages", "total_epochs_planned", "checkpoint_selection_metric", "started_at_utc",
    ):
        assert key in manifest
    assert manifest["experiment_name"] == "manifest_test"
    assert manifest["total_epochs_planned"] == 2


def test_trainer_writes_last_checkpoint_every_epoch(tmp_path, synthetic_metadata_df, tiny_config):
    tiny_config["training"]["warm_up_from_scratch"]["epochs"] = 2
    train_dataset, val_dataset = _build_datasets(synthetic_metadata_df, tiny_config, seed=7)
    model = build_multitask_model(tiny_config)
    trainer = Trainer(
        model, tiny_config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="last_ckpt_test", output_dir=tmp_path / "output",
    )
    trainer.train()

    last_path = tmp_path / "checkpoints" / "last_ckpt_test_last.pt"
    assert last_path.exists()
    assert not last_path.with_suffix(last_path.suffix + ".tmp").exists()
    import torch

    payload = torch.load(last_path, map_location="cpu", weights_only=False)
    assert payload["epoch"] == 2
    assert "model_state_dict" in payload


def test_history_includes_gender_balanced_accuracy_and_f1(tmp_path, synthetic_metadata_df, tiny_config):
    train_dataset, val_dataset = _build_datasets(synthetic_metadata_df, tiny_config, seed=8)
    model = build_multitask_model(tiny_config)
    trainer = Trainer(
        model, tiny_config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="balanced_acc_test", output_dir=tmp_path / "output",
    )
    result = trainer.train()
    assert "val_gender_balanced_accuracy" in result["history"]
    assert "val_gender_f1" in result["history"]
    assert len(result["history"]["val_gender_balanced_accuracy"]) == len(result["history"]["train_loss"])


def test_best_metric_tracker_records_best_epoch(tmp_path, synthetic_metadata_df, tiny_config):
    train_dataset, val_dataset = _build_datasets(synthetic_metadata_df, tiny_config, seed=9)
    model = build_multitask_model(tiny_config)
    trainer = Trainer(
        model, tiny_config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="best_epoch_test", output_dir=tmp_path / "output",
    )
    assert trainer._maybe_checkpoint("age_mae", 10.0, epoch=1, metrics={}) is True
    assert trainer.trackers["age_mae"].best_epoch == 1
    assert trainer._maybe_checkpoint("age_mae", 12.0, epoch=2, metrics={}) is False
    assert trainer.trackers["age_mae"].best_epoch == 1  # unchanged -- epoch 2 was worse
    assert trainer._maybe_checkpoint("age_mae", 8.0, epoch=3, metrics={}) is True
    assert trainer.trackers["age_mae"].best_epoch == 3


def test_epoch_progress_is_printed_unbuffered(tmp_path, synthetic_metadata_df, tiny_config, capsys):
    tiny_config["training"]["warm_up_from_scratch"]["epochs"] = 1
    train_dataset, val_dataset = _build_datasets(synthetic_metadata_df, tiny_config, seed=10)
    model = build_multitask_model(tiny_config)
    trainer = Trainer(
        model, tiny_config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="print_test", output_dir=tmp_path / "output",
    )
    trainer.train()
    captured = capsys.readouterr()
    assert "print_test" in captured.out
    assert "Epoch 01/01" in captured.out
    assert "balanced_acc=" in captured.out
    assert "log_var:" in captured.out
    assert "checkpoint:" in captured.out


def test_backward_compatible_without_explicit_output_dir(tmp_path, synthetic_metadata_df, tiny_config):
    """Trainer must still work when output_dir is omitted (existing callers /
    older test code), defaulting sensibly to checkpoint_dir.parent."""
    train_dataset, val_dataset = _build_datasets(synthetic_metadata_df, tiny_config, seed=3)
    model = build_multitask_model(tiny_config)
    checkpoint_dir = tmp_path / "nested" / "checkpoints"
    trainer = Trainer(
        model, tiny_config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=checkpoint_dir, experiment_name="no_output_dir_test",
    )
    trainer.train()
    assert (tmp_path / "nested" / "metrics" / "no_output_dir_test_history.csv").exists()
