"""End-to-end integration test for scripts/compare_backbones.py.

Trains two tiny real checkpoints (SimpleCNN + CustomResNet18) on synthetic
data, then runs the full comparison CLI against them, checking that every
expected artifact is produced and that the final interpretation is
generated (not just that individual functions work in isolation, as
tests/test_backbone_comparison.py already covers at the unit level).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from compare_backbones import main as compare_backbones_main  # noqa: E402

from src.data.dataset import FaceMultiTaskDataset  # noqa: E402
from src.data.split_utils import split_dataframe  # noqa: E402
from src.data.transforms import EvalTransform, TrainTransform  # noqa: E402
from src.models.multitask_model import build_multitask_model  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402


def _train_tiny_checkpoint(tmp_path, splits_dir, tiny_config, experiment_name, backbone_name):
    # Every checkpoint being compared must share the identical test split --
    # a paired bootstrap comparison is only valid when both models were
    # evaluated on the exact same, index-aligned test samples (see
    # src/evaluation/backbone_comparison.py:_assert_paired_alignment).
    exp_root = tmp_path / experiment_name

    config = copy.deepcopy(tiny_config)
    config["model"]["backbone"]["name"] = backbone_name
    config["paths"]["splits_dir"] = str(splits_dir)
    config["paths"]["output_dir"] = str(exp_root / "output")
    config["paths"]["checkpoint_dir"] = str(exp_root / "checkpoints")

    df = pd.read_csv(splits_dir / "full_metadata_with_splits.csv")
    image_size = config["dataset"]["image_size"]
    train_dataset = FaceMultiTaskDataset(df[df["split"] == "train"], TrainTransform(image_size))
    val_dataset = FaceMultiTaskDataset(df[df["split"] == "validation"], EvalTransform(image_size))

    model = build_multitask_model(config)
    trainer = Trainer(
        model, config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=Path(config["paths"]["checkpoint_dir"]), experiment_name=experiment_name,
    )
    trainer.train()
    checkpoint_path = Path(config["paths"]["checkpoint_dir"]) / f"{experiment_name}_best_balanced_score.pt"
    assert checkpoint_path.exists()
    return checkpoint_path


def test_compare_backbones_end_to_end(tmp_path, synthetic_metadata_df, tiny_config, monkeypatch):
    shared_splits_dir = tmp_path / "shared_splits"
    shared_splits_dir.mkdir(parents=True)
    df = split_dataframe(synthetic_metadata_df, 0.4, 0.2, 0.2, 0.2, seed=10, subject_level_if_available=False)
    df.to_csv(shared_splits_dir / "full_metadata_with_splits.csv", index=False)

    cnn_checkpoint = _train_tiny_checkpoint(tmp_path, shared_splits_dir, tiny_config, "cmp_cnn", "simple_cnn")
    resnet_checkpoint = _train_tiny_checkpoint(tmp_path, shared_splits_dir, tiny_config, "cmp_resnet", "custom_resnet18")

    output_dir = tmp_path / "backbone_comparison"
    argv = [
        "compare_backbones.py",
        "--checkpoint", f"simple_cnn={cnn_checkpoint}",
        "--checkpoint", f"custom_resnet18={resnet_checkpoint}",
        "--resnet-name", "custom_resnet18",
        "--output-dir", str(output_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    exit_code = compare_backbones_main()
    assert exit_code == 0

    for expected in (
        "clean_test_summary.csv", "gender_risk_at_coverage.csv", "gender_aurc.json",
        "gender_pairwise_bootstrap.json", "gender_aurc_bootstrap.json",
        "age_selective_mae_at_coverage.csv", "age_selective_aurc.json",
        "age_pairwise_bootstrap.json", "age_aurc_bootstrap.json",
        "age_bucket_mae.csv", "age_error_percentiles.json", "final_interpretation.md",
    ):
        assert (output_dir / expected).exists(), f"missing artifact: {expected}"

    for expected_plot in (
        "gender_risk_coverage.png", "age_risk_coverage_mae.png", "age_risk_coverage_rmse.png",
        "age_error_cdf.png", "age_tail_error_rates.png",
    ):
        assert (output_dir / "plots" / expected_plot).exists(), f"missing plot: {expected_plot}"

    interpretation = (output_dir / "final_interpretation.md").read_text(encoding="utf-8")
    assert "Is Additional Residual Complexity Justified" in interpretation
