"""Tests for the training-quality improvements applied symmetrically to every
experiment (src/training/trainer.py): differential (discriminative) learning
rates, the warmup-then-cosine-annealing LR schedule, and the loss-balancing
warmup for learned homoscedastic-uncertainty weighting.

These are deliberately applied identically regardless of which backbone is
active (see configs/training.yaml: differential_lr, configs/model.yaml:
loss_balancing.learned_uncertainty.warmup_epochs) -- none of this changes
the *relative* comparison between architectures, only the training quality
of every experiment equally.
"""

from __future__ import annotations

import copy

import torch

from src.data.dataset import FaceMultiTaskDataset
from src.data.split_utils import split_dataframe
from src.data.transforms import EvalTransform, TrainTransform
from src.models.multitask_model import build_multitask_model
from src.training.trainer import Trainer, _build_optimizer, _build_scheduler, resolve_loss_balancing


def _build_datasets(synthetic_metadata_df, tiny_config, seed=0):
    df = split_dataframe(synthetic_metadata_df, 0.5, 0.2, 0.1, 0.2, seed=seed, subject_level_if_available=False)
    image_size = tiny_config["dataset"]["image_size"]
    train_dataset = FaceMultiTaskDataset(df[df["split"] == "train"], TrainTransform(image_size))
    val_dataset = FaceMultiTaskDataset(df[df["split"] == "validation"], EvalTransform(image_size))
    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# resolve_loss_balancing (pure function, no training needed)
# ---------------------------------------------------------------------------

def test_resolve_loss_balancing_forces_equal_fixed_weights_during_warmup():
    loss_cfg = {"mode": "learned_uncertainty", "learned_uncertainty": {"warmup_epochs": 3}}
    for epoch in (1, 2, 3):
        mode, fixed = resolve_loss_balancing(loss_cfg, current_epoch=epoch)
        assert mode == "fixed"
        assert fixed == {"age_weight": 1.0, "gender_weight": 1.0}


def test_resolve_loss_balancing_switches_to_learned_uncertainty_after_warmup():
    loss_cfg = {"mode": "learned_uncertainty", "learned_uncertainty": {"warmup_epochs": 3}}
    mode, _ = resolve_loss_balancing(loss_cfg, current_epoch=4)
    assert mode == "learned_uncertainty"


def test_resolve_loss_balancing_zero_warmup_uses_learned_uncertainty_from_epoch_1():
    loss_cfg = {"mode": "learned_uncertainty", "learned_uncertainty": {"warmup_epochs": 0}}
    mode, _ = resolve_loss_balancing(loss_cfg, current_epoch=1)
    assert mode == "learned_uncertainty"


def test_resolve_loss_balancing_ignores_warmup_epochs_when_mode_is_fixed():
    """A learned_uncertainty.warmup_epochs key must not affect mode='fixed'
    experiments (e.g. exp_c) -- warmup only makes sense for the mode it's nested under."""
    loss_cfg = {"mode": "fixed", "fixed": {"age_weight": 2.0, "gender_weight": 0.5}, "learned_uncertainty": {"warmup_epochs": 5}}
    mode, fixed = resolve_loss_balancing(loss_cfg, current_epoch=1)
    assert mode == "fixed"
    assert fixed == {"age_weight": 2.0, "gender_weight": 0.5}  # the *real* fixed weights, not the warmup override


def test_resolve_loss_balancing_uses_configured_fixed_weights_after_warmup():
    loss_cfg = {
        "mode": "learned_uncertainty", "learned_uncertainty": {"warmup_epochs": 1},
        "fixed": {"age_weight": 9.0, "gender_weight": 9.0},  # should be irrelevant once mode != fixed
    }
    mode, _ = resolve_loss_balancing(loss_cfg, current_epoch=2)
    assert mode == "learned_uncertainty"


# ---------------------------------------------------------------------------
# _build_optimizer: differential learning rates
# ---------------------------------------------------------------------------

def test_build_optimizer_without_differential_lr_puts_every_param_at_base_lr(tiny_config):
    """Without differential LR, every trainable parameter uses the base LR.
    (Group *count* is no longer 1 because build_param_groups now splits each
    LR into decay / no-decay sub-groups -- see src/training/optim.py.)"""
    model = build_multitask_model(tiny_config)
    optimizer = _build_optimizer(model, lr=1e-3, weight_decay=0.05, differential_lr_cfg={"enabled": False})
    assert {group["lr"] for group in optimizer.param_groups} == {1e-3}


def test_build_optimizer_with_differential_lr_uses_two_distinct_lrs(tiny_config):
    model = build_multitask_model(tiny_config)
    optimizer = _build_optimizer(
        model, lr=1e-3, weight_decay=0.05,
        differential_lr_cfg={"enabled": True, "backbone_lr_multiplier": 0.1},
    )
    assert {group["lr"] for group in optimizer.param_groups} == {1e-3 * 0.1, 1e-3}

    backbone_param_ids = {id(p) for p in model.backbone_parameters()}
    for group in optimizer.param_groups:
        for p in group["params"]:
            expected = 1e-3 * 0.1 if id(p) in backbone_param_ids else 1e-3
            assert abs(group["lr"] - expected) < 1e-12


def test_build_optimizer_differential_lr_covers_every_trainable_parameter_exactly_once(tiny_config):
    model = build_multitask_model(tiny_config)
    optimizer = _build_optimizer(
        model, lr=1e-3, weight_decay=0.05, differential_lr_cfg={"enabled": True, "backbone_lr_multiplier": 0.1},
    )
    covered_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    expected_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert covered_ids == expected_ids


def test_build_optimizer_differential_lr_respects_frozen_backbone(tiny_config):
    """When the backbone is fully frozen (requires_grad=False for all its
    params), no parameter uses the low backbone LR -- every remaining
    trainable parameter is at the full base LR."""
    model = build_multitask_model(tiny_config)
    model.set_stage_trainable(freeze_backbone=True, unfreeze_layers=[])
    optimizer = _build_optimizer(
        model, lr=1e-3, weight_decay=0.05, differential_lr_cfg={"enabled": True, "backbone_lr_multiplier": 0.1},
    )
    assert {group["lr"] for group in optimizer.param_groups} == {1e-3}
    # And no frozen backbone parameter leaked into any group.
    backbone_param_ids = {id(p) for p in model.backbone_parameters()}
    grouped_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    trainable_backbone = {id(p) for p in model.backbone_parameters() if p.requires_grad}
    assert grouped_ids & backbone_param_ids == trainable_backbone


# ---------------------------------------------------------------------------
# _build_scheduler: linear warmup + cosine annealing
# ---------------------------------------------------------------------------

def test_scheduler_warms_up_then_decays_to_near_zero():
    model_param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([model_param], lr=1.0)
    scheduler = _build_scheduler(optimizer, total_epochs=10, warmup_epochs=2)

    lrs = []
    for _ in range(10):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    assert lrs[0] < lrs[1]  # still warming up
    assert lrs[1] <= 1.0 + 1e-9  # reaches ~full LR by the end of warmup
    assert lrs[-1] < lrs[1]  # cosine decay brings it back down after warmup


def test_scheduler_handles_zero_warmup_epochs():
    model_param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([model_param], lr=1.0)
    scheduler = _build_scheduler(optimizer, total_epochs=5, warmup_epochs=0)  # must not raise
    for _ in range(5):
        scheduler.step()


def test_scheduler_handles_single_epoch_total():
    """Regression guard: a 1-epoch run (e.g. SMOKE_TEST=True) must not
    raise even though there's no room for both warmup and decay."""
    model_param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([model_param], lr=1.0)
    scheduler = _build_scheduler(optimizer, total_epochs=1, warmup_epochs=1)
    scheduler.step()  # must not raise


# ---------------------------------------------------------------------------
# End-to-end: a real (tiny) training run with these features enabled
# ---------------------------------------------------------------------------

def test_trainer_runs_end_to_end_with_differential_lr_and_loss_balancing_warmup(
    tmp_path, synthetic_metadata_df, tiny_config,
):
    config = copy.deepcopy(tiny_config)
    config["training"]["warm_up_from_scratch"]["epochs"] = 3
    config["training"]["differential_lr"] = {"enabled": True, "backbone_lr_multiplier": 0.1}
    config["model"]["loss_balancing"]["mode"] = "learned_uncertainty"
    config["model"]["loss_balancing"]["learned_uncertainty"]["warmup_epochs"] = 1

    train_dataset, val_dataset = _build_datasets(synthetic_metadata_df, config, seed=7)
    model = build_multitask_model(config)
    output_dir = tmp_path / "output"
    trainer = Trainer(
        model, config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="opt_test", output_dir=output_dir,
    )
    result = trainer.train()  # must not raise

    history = result["history"]
    assert len(history["train_loss"]) == 3
    # Epoch 1 is within the loss-balancing warmup window (warmup_epochs=1) --
    # forced equal fixed weights, so log_var accumulators stay exactly 0.0
    # (the learned-uncertainty log-variance parameters were never used in
    # the loss that epoch).
    assert history["log_var_age"][0] == 0.0
    assert history["log_var_gender"][0] == 0.0
    assert history["effective_age_weight"][0] == 1.0
    assert history["effective_gender_weight"][0] == 1.0


# ---------------------------------------------------------------------------
# Learned-uncertainty correction knobs: gender_loss_scale, log_var_clamp_*
# (added after diagnosing that gender's cross-entropy loss lives on a much
# smaller numeric scale than age's pinball loss, biasing the learned weights
# toward gender regardless of real task difficulty -- see
# src/losses/multitask_loss.py's module docstring)
# ---------------------------------------------------------------------------

def test_trainer_respects_log_var_clamp_bounds(tmp_path, synthetic_metadata_df, tiny_config):
    """A tight log_var_clamp must keep every epoch's reported log_var inside
    bounds, end to end through the real Trainer (not just the pure loss
    function in tests/test_losses.py)."""
    config = copy.deepcopy(tiny_config)
    config["training"]["warm_up_from_scratch"]["epochs"] = 3
    config["model"]["loss_balancing"]["mode"] = "learned_uncertainty"
    config["model"]["loss_balancing"]["learned_uncertainty"]["warmup_epochs"] = 0
    config["model"]["loss_balancing"]["learned_uncertainty"]["log_var_clamp_min"] = -0.2
    config["model"]["loss_balancing"]["learned_uncertainty"]["log_var_clamp_max"] = 0.2

    train_dataset, val_dataset = _build_datasets(synthetic_metadata_df, config, seed=3)
    model = build_multitask_model(config)
    trainer = Trainer(
        model, config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="clamp_test", output_dir=tmp_path / "output",
    )
    result = trainer.train()

    history = result["history"]
    assert all(-0.2 - 1e-6 <= v <= 0.2 + 1e-6 for v in history["log_var_age"])
    assert all(-0.2 - 1e-6 <= v <= 0.2 + 1e-6 for v in history["log_var_gender"])


def test_trainer_passes_gender_loss_scale_from_config_to_loss_function(
    monkeypatch, tmp_path, synthetic_metadata_df, tiny_config,
):
    """The trainer must read gender_loss_scale out of
    model.loss_balancing.learned_uncertainty and forward it to every
    compute_multitask_loss call -- checked deterministically by intercepting
    the real call, rather than by asserting on where log_var ends up after
    a full (statistically noisy, on this tiny synthetic setup) training run.
    """
    import src.training.trainer as trainer_module

    config = copy.deepcopy(tiny_config)
    config["training"]["warm_up_from_scratch"]["epochs"] = 1
    config["model"]["loss_balancing"]["mode"] = "learned_uncertainty"
    config["model"]["loss_balancing"]["learned_uncertainty"]["warmup_epochs"] = 0
    config["model"]["loss_balancing"]["learned_uncertainty"]["gender_loss_scale"] = 8.5

    train_dataset, val_dataset = _build_datasets(synthetic_metadata_df, config, seed=1)
    model = build_multitask_model(config)
    trainer = Trainer(
        model, config, train_dataset, val_dataset, device="cpu",
        checkpoint_dir=tmp_path / "checkpoints", experiment_name="scale_wiring_test", output_dir=tmp_path / "output",
    )

    real_compute_multitask_loss = trainer_module.compute_multitask_loss
    seen_scales = []

    def _spy(*args, **kwargs):
        seen_scales.append(kwargs.get("gender_loss_scale"))
        return real_compute_multitask_loss(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "compute_multitask_loss", _spy)
    trainer.train()

    assert seen_scales  # at least one batch ran
    assert all(scale == 8.5 for scale in seen_scales)
