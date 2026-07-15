"""Tests for task-specific residual bottleneck adapters."""

from __future__ import annotations

import torch

from src.models.adapters import AgeAdapter, BottleneckAdapter, GenderAdapter, IdentityAdapter


def test_adapter_output_shape_matches_input():
    adapter = BottleneckAdapter(input_dim=512, bottleneck_dim=128)
    z = torch.randn(4, 512)
    out = adapter(z)
    assert out.shape == z.shape


def test_adapter_is_near_identity_at_init():
    """Zero-initialized up-projection means adapter_output == z at init."""
    adapter = BottleneckAdapter(input_dim=64, bottleneck_dim=16, dropout=0.0)
    adapter.eval()
    z = torch.randn(3, 64)
    out = adapter(z)
    assert torch.allclose(out, z, atol=1e-6)


def test_adapter_residual_formula():
    adapter = BottleneckAdapter(input_dim=32, bottleneck_dim=8, dropout=0.0)
    adapter.eval()
    z = torch.randn(2, 32)
    expected_delta = adapter.up_proj(adapter.activation(adapter.down_proj(z)))
    out = adapter(z)
    assert torch.allclose(out, z + expected_delta, atol=1e-5)


def test_age_and_gender_adapters_are_independent_modules():
    age_adapter = AgeAdapter(input_dim=64, bottleneck_dim=16)
    gender_adapter = GenderAdapter(input_dim=64, bottleneck_dim=16)
    for p1, p2 in zip(age_adapter.parameters(), gender_adapter.parameters()):
        assert p1.data_ptr() != p2.data_ptr()


def test_identity_adapter_is_true_noop():
    adapter = IdentityAdapter()
    z = torch.randn(5, 512)
    assert torch.equal(adapter(z), z)
    assert adapter.num_parameters() == 0


def test_adapter_parameter_count_much_smaller_than_backbone():
    from src.models.custom_resnet import CustomResNet18

    backbone = CustomResNet18()
    adapter = BottleneckAdapter(input_dim=512, bottleneck_dim=128)
    assert adapter.num_parameters() < backbone.num_parameters() * 0.05
