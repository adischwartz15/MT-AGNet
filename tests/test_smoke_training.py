"""End-to-end smoke test: tiny synthetic dataset, one training epoch, real checkpoint saved.

Synthetic data only -- never mixed with real Kaggle experiment results.
This test exercises the full path: dataset -> model -> masked multi-task
loss -> optimizer step -> checkpoint save, to catch integration breakage
that unit tests alone would miss.
"""

from __future__ import annotations

from src.data.dataset import FaceMultiTaskDataset
from src.data.split_utils import split_dataframe
from src.data.transforms import EvalTransform, TrainTransform
from src.models.multitask_model import build_multitask_model
from src.training.trainer import Trainer


def test_smoke_training_runs_and_saves_checkpoints(tmp_path, synthetic_metadata_df, tiny_config):
    df = split_dataframe(synthetic_metadata_df, 0.5, 0.2, 0.1, 0.2, seed=0, subject_level_if_available=False)

    image_size = tiny_config["dataset"]["image_size"]
    train_dataset = FaceMultiTaskDataset(df[df["split"] == "train"], TrainTransform(image_size))
    val_dataset = FaceMultiTaskDataset(df[df["split"] == "validation"], EvalTransform(image_size))

    model = build_multitask_model(tiny_config)
    checkpoint_dir = tmp_path / "checkpoints"

    trainer = Trainer(
        model, tiny_config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=checkpoint_dir, experiment_name="smoke",
    )
    result = trainer.train()

    assert "history" in result
    assert len(result["history"]["train_loss"]) >= 1
    assert not any(v != v for v in result["history"]["train_loss"])  # no NaNs

    saved_checkpoints = list(checkpoint_dir.glob("smoke_best_*.pt"))
    assert len(saved_checkpoints) >= 1


def test_smoke_training_with_simple_cnn_backbone(tmp_path, synthetic_metadata_df, tiny_config):
    """The plain-CNN controlled baseline must be trainable end-to-end too."""
    tiny_config["model"]["backbone"]["name"] = "simple_cnn"
    tiny_config["model"]["loss_balancing"]["mode"] = "learned_uncertainty"

    df = split_dataframe(synthetic_metadata_df, 0.5, 0.2, 0.1, 0.2, seed=2, subject_level_if_available=False)
    image_size = tiny_config["dataset"]["image_size"]
    train_dataset = FaceMultiTaskDataset(df[df["split"] == "train"], TrainTransform(image_size))
    val_dataset = FaceMultiTaskDataset(df[df["split"] == "validation"], EvalTransform(image_size))

    model = build_multitask_model(tiny_config)
    from src.models.simple_cnn import SimpleCNNBackbone

    assert isinstance(model.backbone, SimpleCNNBackbone)

    trainer = Trainer(
        model, tiny_config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="smoke_simple_cnn",
    )
    result = trainer.train()

    assert len(result["history"]["train_loss"]) >= 1
    assert not any(v != v for v in result["history"]["train_loss"])  # no NaNs
    saved_checkpoints = list((tmp_path / "checkpoints").glob("smoke_simple_cnn_best_*.pt"))
    assert len(saved_checkpoints) >= 1


def test_smoke_training_with_missing_gender_labels(tmp_path, synthetic_metadata_df, tiny_config):
    """Some samples have age-only labels; masked loss must not crash or NaN."""
    df = synthetic_metadata_df.copy()
    df.loc[df.index[::3], "gender_label"] = float("nan")
    df = split_dataframe(df, 0.5, 0.2, 0.1, 0.2, seed=1, subject_level_if_available=False)

    image_size = tiny_config["dataset"]["image_size"]
    train_dataset = FaceMultiTaskDataset(df[df["split"] == "train"], TrainTransform(image_size))
    val_dataset = FaceMultiTaskDataset(df[df["split"] == "validation"], EvalTransform(image_size))

    model = build_multitask_model(tiny_config)
    trainer = Trainer(
        model, tiny_config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="smoke_partial_labels",
    )
    result = trainer.train()
    assert len(result["history"]["train_loss"]) >= 1
