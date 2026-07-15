"""Tests for Experiment 0c (Custom ResNet-18, zero_init_residual=false).

This is the recommended architecture control from the final evaluation
protocol: same architecture/seeds/training setup as Experiment D, differing
only in whether each residual branch's final BatchNorm is zero-initialized.
Combined with PlainDeep18NoSkip (Experiment 0b), it lets
"PlainDeep18NoSkip vs. Experiment 0c" isolate residual shortcuts more
cleanly than "PlainDeep18NoSkip vs. Experiment D" (both non-zero-init), and
"Experiment D vs. Experiment 0c" isolate the zero-init trick on its own.
"""

from __future__ import annotations

import torch

from src.models.custom_resnet import CustomResNet18
from src.models.multitask_model import build_multitask_model
from src.utils.config import REPO_ROOT, load_config, load_full_config

_EXP_0C = "exp_0c_custom_resnet18_no_zero_init_shared_adapters_learned_balance"
_EXP_D = "exp_d_shared_adapters_learned_balance"


def test_exp_0c_is_registered_in_experiments_yaml_and_run_order():
    experiments_cfg = load_config(REPO_ROOT / "configs" / "experiments.yaml")
    assert _EXP_0C in experiments_cfg["experiments"]
    assert _EXP_0C in experiments_cfg["run_order"]
    # Runs right after 0b (both plain-vs-ResNet-family controls) and before
    # the main A-D ablation sequence.
    run_order = experiments_cfg["run_order"]
    assert run_order.index(_EXP_0C) > run_order.index("exp_0b_plain_deep18_no_skip_shared_adapters_learned_balance")
    assert run_order.index(_EXP_0C) < run_order.index("exp_a_separate")


def test_exp_0c_overrides_only_zero_init_residual_relative_to_exp_d():
    experiments_cfg = load_config(REPO_ROOT / "configs" / "experiments.yaml")["experiments"]
    exp_0c_overrides = experiments_cfg[_EXP_0C]["overrides"]
    exp_d_overrides = experiments_cfg[_EXP_D]["overrides"]

    assert exp_0c_overrides["model"]["backbone"]["zero_init_residual"] is False
    assert exp_0c_overrides["model"]["architecture"] == exp_d_overrides["model"]["architecture"]
    assert exp_0c_overrides["model"]["adapters"] == exp_d_overrides["model"]["adapters"]
    assert exp_0c_overrides["model"]["loss_balancing"] == exp_d_overrides["model"]["loss_balancing"]


def test_exp_0c_config_resolves_to_zero_init_residual_false():
    experiments_cfg = load_config(REPO_ROOT / "configs" / "experiments.yaml")["experiments"]
    config = load_full_config(overrides=experiments_cfg[_EXP_0C]["overrides"])
    assert config["model"]["backbone"]["zero_init_residual"] is False
    assert config["model"]["backbone"]["name"] == "custom_resnet18"

    default_config = load_full_config(overrides=experiments_cfg[_EXP_D]["overrides"])
    assert default_config["model"]["backbone"].get("zero_init_residual", True) is True


def test_zero_init_residual_false_leaves_last_bn_weight_at_one_not_zero():
    """Direct behavioral check: with zero_init_residual=False, each residual
    block's final BatchNorm weight must stay at its default init (1.0),
    not be zeroed -- otherwise the config override would be inert."""
    torch.manual_seed(0)
    model_default = CustomResNet18(block_layout=(1, 1, 1, 1), zero_init_residual=True)
    model_no_zero_init = CustomResNet18(block_layout=(1, 1, 1, 1), zero_init_residual=False)

    default_bn2_weights = [m.bn2.weight for m in model_default.modules() if type(m).__name__ == "BasicBlock"]
    no_zero_init_bn2_weights = [m.bn2.weight for m in model_no_zero_init.modules() if type(m).__name__ == "BasicBlock"]

    assert all(torch.all(w == 0.0) for w in default_bn2_weights)
    assert all(torch.all(w == 1.0) for w in no_zero_init_bn2_weights)


def test_exp_0c_builds_a_working_multitask_model():
    experiments_cfg = load_config(REPO_ROOT / "configs" / "experiments.yaml")["experiments"]
    config = load_full_config(overrides=experiments_cfg[_EXP_0C]["overrides"])
    model = build_multitask_model(config)
    images = torch.randn(2, 3, config["dataset"]["image_size"], config["dataset"]["image_size"])
    outputs = model(images)
    assert outputs["age_output"]["q50"].shape == (2,)
    assert outputs["gender_logits"].shape[0] == 2
