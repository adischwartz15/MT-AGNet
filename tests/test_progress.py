"""Tests for src/training/progress.py -- the shared live-progress formatting
used by the trainer."""

from __future__ import annotations

import torch
from torch import nn

from src.training.progress import (
    describe_trainable_backbone_parts, format_epoch_report, format_lr_groups,
    format_multi_seed_preflight, format_resume_announcement, format_stage_announcement,
)


def test_format_lr_groups_renders_none_as_frozen_not_zero():
    text = format_lr_groups({"backbone": None, "adapters": 3e-4})
    assert "backbone=frozen/inactive" in text
    assert "adapters=3.00e-04" in text
    assert "0.00e+00" not in text  # never fabricate a numeric LR for a frozen group


def test_format_lr_groups_empty_is_na():
    assert format_lr_groups({}) == "n/a"


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.stage1 = nn.Linear(4, 4)
        self.stage2 = nn.Linear(4, 4)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _TinyBackbone()
        self.head = nn.Linear(4, 2)


def test_describe_trainable_backbone_parts_reports_frozen_and_trainable():
    model = _TinyModel()
    for p in model.backbone.stem.parameters():
        p.requires_grad = False
    for p in model.backbone.stage1.parameters():
        p.requires_grad = True
    for p in model.backbone.stage2.parameters():
        p.requires_grad = True
    text = describe_trainable_backbone_parts(model)
    assert "stem=frozen" in text
    assert "stage1=trainable" in text
    assert "stage2=trainable" in text


def test_describe_trainable_backbone_parts_reports_partial():
    model = _TinyModel()
    params = list(model.backbone.stage1.parameters())
    params[0].requires_grad = False
    for p in params[1:]:
        p.requires_grad = True
    text = describe_trainable_backbone_parts(model)
    assert "stage1=partially-trainable" in text


def test_describe_trainable_backbone_parts_handles_no_backbone_attribute():
    model = nn.Linear(4, 2)
    assert "n/a" in describe_trainable_backbone_parts(model)


def test_format_stage_announcement_includes_param_counts_and_backbone_parts():
    model = _TinyModel()
    text = format_stage_announcement("exp_x", 42, "Stage 1: frozen backbone", model)
    assert "exp_x" in text
    assert "seed=42" in text
    assert "Stage 1: frozen backbone" in text
    assert "trainable_params=" in text
    assert "backbone parts:" in text


def test_format_epoch_report_includes_every_required_field():
    train_metrics = {
        "loss": 1.234, "age_loss": 0.5, "gender_loss": 0.6,
        "log_var_age": -0.1, "log_var_gender": 0.2,
        "effective_age_weight": 1.1, "effective_gender_weight": 0.9,
    }
    val_metrics = {
        "loss": 1.0, "age_mae": 5.5, "age_rmse": 6.6,
        "gender_accuracy": 0.9, "gender_balanced_accuracy": 0.88, "gender_f1": 0.87,
        "gender_selective_accuracy": 0.93, "gender_coverage": 0.8, "gender_abstention": 0.2,
    }
    text = format_epoch_report(
        experiment_name="exp_d", seed=123, stage_name="warm_up_from_scratch",
        epoch=7, total_epochs=30, train_metrics=train_metrics, val_metrics=val_metrics,
        lr_groups={"backbone": 3e-5, "head": 3e-4}, selection_score=0.654,
        is_best=True, best_score=0.654, best_epoch=7,
        early_stopping_bad_epochs=0, early_stopping_patience=8, epoch_seconds=12.3,
        checkpoint_path="checkpoints/exp_d_best_balanced_score.pt",
    )
    for expected in (
        "exp_d", "seed=123", "warm_up_from_scratch", "Epoch 07/30", "12.3s",
        "total=1.2340", "age=0.5000", "gender=0.6000",  # train
        "age_mae=5.500", "age_rmse=6.600",  # val age
        "gender_acc=0.900", "balanced_acc=0.880", "f1=0.870",  # val gender
        "selective_acc=0.930", "coverage=0.800", "abstention=0.200",
        "backbone=3.00e-05", "head=3.00e-04",  # lr
        "log_var: age=-0.1000", "gender=0.2000",
        "loss_weights: age=1.1000", "gender=0.9000",
        "selection_score=0.6540", "best=yes", "best_score=0.6540", "@ epoch 7",
        "early_stop=0/8",
        "checkpoint: checkpoints/exp_d_best_balanced_score.pt",
    ):
        assert expected in text, f"missing {expected!r} in:\n{text}"


def test_format_epoch_report_renders_missing_metrics_as_na_not_fabricated():
    text = format_epoch_report(
        experiment_name="exp_x", seed=None, stage_name="Stage 1: frozen backbone",
        epoch=1, total_epochs=3, train_metrics={}, val_metrics={},
        lr_groups={}, selection_score=float("nan"), is_best=False, best_score=None, best_epoch=None,
        early_stopping_bad_epochs=0, early_stopping_patience=8, epoch_seconds=1.0,
        checkpoint_path=None,
    )
    assert "n/a" in text
    assert "checkpoint: n/a" in text
    assert "@ epoch n/a" in text
    assert "[exp_x]" in text  # no "seed=None" when seed is not given


def test_format_multi_seed_preflight_lists_every_category():
    text = format_multi_seed_preflight(
        "exp_d_shared_adapters_learned_balance",
        requested_seeds=[42, 123, 2026], completed_seeds=[42],
        incomplete_resumable_seeds=[123], missing_seeds=[2026], will_run_now_seeds=[123, 2026],
    )
    assert "requested seeds:            [42, 123, 2026]" in text
    assert "already completed (reused): [42]" in text
    assert "incomplete (will resume):   [123]" in text
    assert "missing (will start fresh): [2026]" in text
    assert "will run now:               [123, 2026]" in text


def test_format_resume_announcement_includes_every_required_field():
    text = format_resume_announcement(
        "exp_d_shared_adapters_learned_balance", 123, "local", "Stage 2: fine-tune", 5, 340, 0.87,
        "checkpoints/.../last.pt", "abc123", "def456",
    )
    for expected in (
        "seed=123", "resume source:     local", "stage:             Stage 2: fine-tune",
        "epoch:             5", "global step:       340", "best score so far: 0.8700",
        "checkpoint path:   checkpoints/.../last.pt", "checkpoint sha256: abc123", "split sha256:      def456",
    ):
        assert expected in text, f"missing {expected!r} in:\n{text}"


def test_format_resume_announcement_handles_missing_fields_as_na():
    text = format_resume_announcement("exp_x", None, "local", None, None, None, None, None, None, None)
    assert "stage:             n/a" in text
    assert "epoch:             n/a" in text
    assert "global step:       n/a" in text
    assert "checkpoint sha256: n/a" in text
    assert "split sha256:      n/a" in text


def test_emit_prints_to_stdout(capsys):
    from src.training.progress import emit

    emit("hello progress")
    captured = capsys.readouterr()
    assert "hello progress" in captured.out
