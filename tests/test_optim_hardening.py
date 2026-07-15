"""Tests for src/training/optim.py -- the shared optimizer no-decay grouping
(no weight decay on biases / normalization / scalar loss-balancing params,
every trainable parameter in exactly one group) and the real-warmup
scheduler -- plus the seeded DataLoader generator wired into both trainers.

All CPU-only, synthetic, no network / pretrained downloads.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training.optim import build_param_groups, build_warmup_cosine_scheduler, is_no_decay_param


class _ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)          # weight 4-D (decay), bias 1-D (no-decay)
        self.bn = nn.BatchNorm2d(4)             # weight/bias 1-D (no-decay)
        self.fc = nn.Linear(4, 2)               # weight 2-D (decay), bias 1-D (no-decay)
        self.log_var = nn.Parameter(torch.zeros(()))  # 0-D scalar (no-decay)


def _wd_of(groups, param):
    for g in groups:
        if any(p is param for p in g["params"]):
            return g["weight_decay"]
    raise AssertionError("parameter not found in any group")


def test_is_no_decay_classification():
    assert is_no_decay_param(nn.Parameter(torch.zeros(())))       # scalar
    assert is_no_decay_param(nn.Parameter(torch.zeros(5)))        # 1-D (bias/norm)
    assert not is_no_decay_param(nn.Parameter(torch.zeros(5, 5)))  # 2-D weight
    assert not is_no_decay_param(nn.Parameter(torch.zeros(4, 3, 3, 3)))  # conv weight


def test_biases_norms_and_scalars_get_zero_decay():
    model = _ToyNet()
    groups = build_param_groups(model.named_parameters(), lambda n, p: 1e-3, weight_decay=0.05)

    assert _wd_of(groups, model.conv.weight) == 0.05
    assert _wd_of(groups, model.fc.weight) == 0.05
    assert _wd_of(groups, model.conv.bias) == 0.0
    assert _wd_of(groups, model.bn.weight) == 0.0
    assert _wd_of(groups, model.bn.bias) == 0.0
    assert _wd_of(groups, model.fc.bias) == 0.0
    assert _wd_of(groups, model.log_var) == 0.0


def test_every_trainable_param_in_exactly_one_group():
    model = _ToyNet()
    groups = build_param_groups(model.named_parameters(), lambda n, p: 1e-3, weight_decay=0.05)
    grouped = [p for g in groups for p in g["params"]]
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(grouped) == len(trainable)
    # No duplicates (compare by identity).
    assert len({id(p) for p in grouped}) == len(grouped)
    assert {id(p) for p in grouped} == {id(p) for p in trainable}


def test_frozen_params_excluded():
    model = _ToyNet()
    model.conv.weight.requires_grad = False
    groups = build_param_groups(model.named_parameters(), lambda n, p: 1e-3, weight_decay=0.05)
    grouped_ids = {id(p) for g in groups for p in g["params"]}
    assert id(model.conv.weight) not in grouped_ids


def test_differential_lr_expressed_via_lr_for():
    model = _ToyNet()
    conv_ids = {id(model.conv.weight), id(model.conv.bias)}

    def lr_for(_name, p):
        return 1e-4 if id(p) in conv_ids else 1e-3

    groups = build_param_groups(model.named_parameters(), lr_for, weight_decay=0.05)
    # conv.weight -> lr 1e-4 with decay; conv.bias -> lr 1e-4 no decay; both present.
    lrs_for_conv_weight = [g["lr"] for g in groups if any(p is model.conv.weight for p in g["params"])]
    assert lrs_for_conv_weight == [1e-4]
    lrs_for_fc_weight = [g["lr"] for g in groups if any(p is model.fc.weight for p in g["params"])]
    assert lrs_for_fc_weight == [1e-3]


def test_warmup_scheduler_starts_below_base_and_rises():
    model = nn.Linear(4, 2)
    base_lr = 1e-2
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    scheduler = build_warmup_cosine_scheduler(optimizer, total_epochs=10, warmup_epochs=3, warmup_start_factor=0.1)

    lr_epoch0 = optimizer.param_groups[0]["lr"]
    assert lr_epoch0 < base_lr  # real warmup: does NOT start at base LR
    assert abs(lr_epoch0 - base_lr * 0.1) < 1e-9

    lrs = [lr_epoch0]
    for _ in range(3):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    # LR strictly increases across the warmup epochs then reaches base.
    assert lrs[1] > lrs[0]
    assert lrs[2] > lrs[1]
    assert abs(lrs[3] - base_lr) < 1e-6  # base LR reached at end of warmup


def test_single_warmup_epoch_still_warms():
    """The exact bug the fix targets: warmup_epochs=1 used to give
    start_factor=1.0 (no warmup). It must now start below the base LR."""
    optimizer = torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=1e-2)
    build_warmup_cosine_scheduler(optimizer, total_epochs=5, warmup_epochs=1, warmup_start_factor=0.1)
    assert optimizer.param_groups[0]["lr"] < 1e-2


def test_warmup_start_factor_validated():
    optimizer = torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=1e-2)
    for bad in (0.0, -0.1, 1.5):
        try:
            build_warmup_cosine_scheduler(optimizer, total_epochs=5, warmup_epochs=2, warmup_start_factor=bad)
            raise AssertionError(f"expected ValueError for warmup_start_factor={bad}")
        except ValueError:
            pass


def test_cosine_decay_follows_warmup():
    optimizer = torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=1e-2)
    scheduler = build_warmup_cosine_scheduler(optimizer, total_epochs=6, warmup_epochs=2, warmup_start_factor=0.1)
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(5):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    peak = max(lrs)
    peak_idx = lrs.index(peak)
    # After the warmup peak, LR decays (cosine).
    assert lrs[-1] < peak
    assert peak_idx <= 2


def test_seeded_dataloader_generator_is_deterministic():
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(torch.arange(20).unsqueeze(1).float())

    def _order(seed):
        g = torch.Generator()
        g.manual_seed(seed)
        loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=g)
        return [int(x[0].item()) for batch in loader for x in batch[0]]

    assert _order(123) == _order(123)      # same seed -> same order
    assert _order(123) != _order(999)      # different seed -> (almost surely) different order


def test_core_build_optimizer_zero_decays_log_var(tiny_config):
    """The real MultiTaskFaceModel + Trainer._build_optimizer path: the
    learned-uncertainty log-variance parameters must never receive weight
    decay."""
    from src.models.multitask_model import build_multitask_model
    from src.training.trainer import _build_optimizer

    tiny_config["model"]["loss_balancing"]["mode"] = "learned_uncertainty"
    model = build_multitask_model(tiny_config)
    optimizer = _build_optimizer(
        model, lr=1e-3, weight_decay=0.05,
        differential_lr_cfg=tiny_config["training"].get("differential_lr", {}),
    )
    assert model.log_var_age is not None
    assert _wd_of(optimizer.param_groups, model.log_var_age) == 0.0
    assert _wd_of(optimizer.param_groups, model.log_var_gender) == 0.0
    # A conv/linear weight tensor somewhere still decays.
    weight_tensors = [p for n, p in model.named_parameters() if p.ndim >= 2 and p.requires_grad]
    assert any(_wd_of(optimizer.param_groups, w) == 0.05 for w in weight_tensors)


def test_core_build_optimizer_differential_backbone_lr(tiny_config):
    from src.models.multitask_model import build_multitask_model
    from src.training.trainer import _build_optimizer

    model = build_multitask_model(tiny_config)
    optimizer = _build_optimizer(
        model, lr=1e-3, weight_decay=0.05,
        differential_lr_cfg={"enabled": True, "backbone_lr_multiplier": 0.1},
    )
    backbone_ids = {id(p) for p in model.backbone_parameters()}
    for g in optimizer.param_groups:
        for p in g["params"]:
            if id(p) in backbone_ids:
                assert abs(g["lr"] - 1e-4) < 1e-12
            else:
                assert abs(g["lr"] - 1e-3) < 1e-12
