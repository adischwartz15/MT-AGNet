"""Tests for quantile ordering, pinball loss, masking, and multi-task loss balancing."""

from __future__ import annotations

import torch

from src.losses.multitask_loss import compute_multitask_loss
from src.losses.quantile_loss import multi_quantile_pinball_loss, pinball_loss
from src.models.heads import AgeQuantileHead


def test_age_quantile_head_ordering():
    head = AgeQuantileHead(input_dim=32, hidden_dim=16, age_min=0, age_max=120)
    z = torch.randn(20, 32)
    out = head(z)
    assert torch.all(out["q10"] <= out["q50"] + 1e-4)
    assert torch.all(out["q50"] <= out["q90"] + 1e-4)


def test_age_quantile_head_respects_age_range():
    head = AgeQuantileHead(input_dim=16, hidden_dim=8, age_min=0, age_max=120)
    z = torch.randn(50, 16) * 5
    out = head(z)
    assert torch.all(out["q10"] >= 0 - 1e-4)
    assert torch.all(out["q90"] <= 120 + 1e-4)


def test_pinball_loss_known_values():
    pred = torch.tensor([10.0])
    target = torch.tensor([12.0])
    # error = target - pred = 2; tau=0.5 -> loss = 0.5*2 = 1.0
    loss = pinball_loss(pred, target, 0.5)
    assert torch.allclose(loss, torch.tensor([1.0]))

    # over-prediction: target - pred = -2; tau=0.1 -> max(0.1*-2, -0.9*-2) = max(-0.2, 1.8) = 1.8
    loss2 = pinball_loss(torch.tensor([14.0]), torch.tensor([12.0]), 0.1)
    assert torch.allclose(loss2, torch.tensor([1.8]))


def test_masked_pinball_loss_ignores_unlabeled_samples():
    q10 = torch.tensor([1.0, 100.0])
    q50 = torch.tensor([1.0, 100.0])
    q90 = torch.tensor([1.0, 100.0])
    target = torch.tensor([1.0, 999.0])  # second sample's target is nonsense but masked out
    mask = torch.tensor([True, False])
    loss = multi_quantile_pinball_loss(q10, q50, q90, target, mask)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)


def test_masked_pinball_loss_all_masked_returns_zero():
    q10 = q50 = q90 = torch.zeros(3)
    target = torch.zeros(3)
    mask = torch.zeros(3, dtype=torch.bool)
    loss = multi_quantile_pinball_loss(q10, q50, q90, target, mask)
    assert loss.item() == 0.0


def _make_age_output(batch_size):
    return {
        "q10_raw": torch.rand(batch_size) * 10,
        "q50_raw": torch.rand(batch_size) * 10 + 10,
        "q90_raw": torch.rand(batch_size) * 10 + 20,
    }


def test_fixed_loss_balancing_combines_both_tasks():
    batch_size = 6
    age_output = _make_age_output(batch_size)
    gender_logits = torch.randn(batch_size, 2)
    age_target = torch.rand(batch_size) * 80
    gender_target = torch.randint(0, 2, (batch_size,))
    age_mask = torch.ones(batch_size, dtype=torch.bool)
    gender_mask = torch.ones(batch_size, dtype=torch.bool)

    result = compute_multitask_loss(
        age_output, gender_logits, age_target, age_mask, gender_target, gender_mask,
        mode="fixed", fixed_age_weight=2.0, fixed_gender_weight=3.0,
    )
    assert result.age_loss is not None and result.gender_loss is not None
    expected_total = 2.0 * result.age_loss.item() + 3.0 * result.gender_loss.item()
    assert abs(result.total_loss.item() - expected_total) < 1e-4
    assert result.effective_age_weight == 2.0
    assert result.effective_gender_weight == 3.0


def test_learned_uncertainty_balancing_uses_log_variances():
    batch_size = 6
    age_output = _make_age_output(batch_size)
    gender_logits = torch.randn(batch_size, 2)
    age_target = torch.rand(batch_size) * 80
    gender_target = torch.randint(0, 2, (batch_size,))
    age_mask = torch.ones(batch_size, dtype=torch.bool)
    gender_mask = torch.ones(batch_size, dtype=torch.bool)
    log_var_age = torch.tensor(0.5, requires_grad=True)
    log_var_gender = torch.tensor(-0.3, requires_grad=True)

    result = compute_multitask_loss(
        age_output, gender_logits, age_target, age_mask, gender_target, gender_mask,
        mode="learned_uncertainty", log_var_age=log_var_age, log_var_gender=log_var_gender,
    )
    expected_total = (
        torch.exp(-log_var_age).item() * result.age_loss.item() + log_var_age.item()
        + torch.exp(-log_var_gender).item() * result.gender_loss.item() + log_var_gender.item()
    )
    assert abs(result.total_loss.item() - expected_total) < 1e-4
    assert abs(result.log_var_age - 0.5) < 1e-6
    assert abs(result.log_var_gender - (-0.3)) < 1e-6


def test_gender_loss_scale_affects_total_but_not_reported_gender_loss():
    """gender_loss_scale must scale the gender term inside total_loss (and
    therefore where log_var_gender's gradient pushes it), but result.gender_loss
    itself must stay the raw, unscaled cross-entropy value (used for
    metrics/plots that should remain in real "nats" units)."""
    batch_size = 6
    age_output = _make_age_output(batch_size)
    gender_logits = torch.randn(batch_size, 2)
    age_target = torch.rand(batch_size) * 80
    gender_target = torch.randint(0, 2, (batch_size,))
    age_mask = torch.ones(batch_size, dtype=torch.bool)
    gender_mask = torch.ones(batch_size, dtype=torch.bool)
    log_var_age = torch.tensor(0.5, requires_grad=True)
    log_var_gender = torch.tensor(-0.3, requires_grad=True)

    unscaled = compute_multitask_loss(
        age_output, gender_logits, age_target, age_mask, gender_target, gender_mask,
        mode="learned_uncertainty", log_var_age=log_var_age, log_var_gender=log_var_gender,
    )
    scaled = compute_multitask_loss(
        age_output, gender_logits, age_target, age_mask, gender_target, gender_mask,
        mode="learned_uncertainty", log_var_age=log_var_age, log_var_gender=log_var_gender,
        gender_loss_scale=8.5,
    )

    assert abs(scaled.gender_loss.item() - unscaled.gender_loss.item()) < 1e-6
    precision_gender = torch.exp(-log_var_gender).item()
    expected_extra = precision_gender * unscaled.gender_loss.item() * (8.5 - 1.0)
    assert abs(scaled.total_loss.item() - unscaled.total_loss.item() - expected_extra) < 1e-4
    # effective_gender_weight reports the learned precision only, not the scale.
    assert abs(scaled.effective_gender_weight - unscaled.effective_gender_weight) < 1e-6


def test_log_var_clamp_bounds_effective_weight():
    """With log_var_gender initialized far outside [-1, 1], the clamp must
    cap both the reported log_var and the resulting effective weight -- the
    raw nn.Parameter is untouched (clamping happens only inside the loss)."""
    batch_size = 6
    age_output = _make_age_output(batch_size)
    gender_logits = torch.randn(batch_size, 2)
    age_target = torch.rand(batch_size) * 80
    gender_target = torch.randint(0, 2, (batch_size,))
    age_mask = torch.ones(batch_size, dtype=torch.bool)
    gender_mask = torch.ones(batch_size, dtype=torch.bool)
    log_var_age = torch.tensor(0.5, requires_grad=True)
    log_var_gender = torch.tensor(-1.75, requires_grad=True)  # far below -1.0

    result = compute_multitask_loss(
        age_output, gender_logits, age_target, age_mask, gender_target, gender_mask,
        mode="learned_uncertainty", log_var_age=log_var_age, log_var_gender=log_var_gender,
        log_var_clamp_min=-1.0, log_var_clamp_max=1.0,
    )

    assert abs(result.log_var_gender - (-1.0)) < 1e-6  # clamped, not the raw -1.75
    assert abs(result.effective_gender_weight - torch.exp(torch.tensor(1.0)).item()) < 1e-4
    assert abs(log_var_gender.item() - (-1.75)) < 1e-9  # raw parameter itself untouched


def test_log_var_clamp_one_sided_leaves_other_bound_open():
    batch_size = 6
    age_output = _make_age_output(batch_size)
    gender_logits = torch.randn(batch_size, 2)
    age_target = torch.rand(batch_size) * 80
    gender_target = torch.randint(0, 2, (batch_size,))
    age_mask = torch.ones(batch_size, dtype=torch.bool)
    gender_mask = torch.ones(batch_size, dtype=torch.bool)
    log_var_age = torch.tensor(5.0, requires_grad=True)  # would only be capped by a max bound
    log_var_gender = torch.tensor(-0.3, requires_grad=True)

    result = compute_multitask_loss(
        age_output, gender_logits, age_target, age_mask, gender_target, gender_mask,
        mode="learned_uncertainty", log_var_age=log_var_age, log_var_gender=log_var_gender,
        log_var_clamp_max=2.0,  # no min bound
    )
    assert abs(result.log_var_age - 2.0) < 1e-6
    assert abs(result.log_var_gender - (-0.3)) < 1e-6  # untouched, no min bound was set


def test_loss_omits_task_when_batch_has_no_labels():
    batch_size = 4
    age_output = _make_age_output(batch_size)
    gender_logits = torch.randn(batch_size, 2)
    age_target = torch.rand(batch_size) * 80
    gender_target = torch.randint(0, 2, (batch_size,))
    age_mask = torch.zeros(batch_size, dtype=torch.bool)  # no age labels in this batch
    gender_mask = torch.ones(batch_size, dtype=torch.bool)

    result = compute_multitask_loss(
        age_output, gender_logits, age_target, age_mask, gender_target, gender_mask, mode="fixed",
    )
    assert result.age_loss is None
    assert result.gender_loss is not None
    assert result.effective_age_weight == 0.0
    assert abs(result.total_loss.item() - result.gender_loss.item()) < 1e-4
