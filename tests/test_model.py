"""Tests for the manually implemented Custom ResNet-18 backbone."""

from __future__ import annotations

import torch

from src.models.custom_resnet import BasicBlock, CustomResNet18, build_backbone


def test_custom_resnet_output_shape():
    model = CustomResNet18(embedding_dim=512)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 512)


def test_custom_resnet_variable_input_size():
    # batch_size=2 (not 1): at 32px input, layer4's spatial resolution collapses to
    # 1x1, and BatchNorm2d cannot compute batch statistics from a single sample
    # per channel in training mode -- this is expected BatchNorm behavior, not a
    # backbone bug.
    model = CustomResNet18(embedding_dim=512)
    for size in (32, 96, 128):
        x = torch.randn(2, 3, size, size)
        out = model(x)
        assert out.shape == (2, 512)


def test_basic_block_identity_shortcut_preserves_shape():
    block = BasicBlock(in_channels=32, out_channels=32, stride=1)
    x = torch.randn(2, 32, 16, 16)
    out = block(x)
    assert out.shape == x.shape
    assert block.downsample is None


def test_basic_block_downsampling_changes_shape():
    block = BasicBlock(in_channels=32, out_channels=64, stride=2)
    x = torch.randn(2, 32, 16, 16)
    out = block(x)
    assert out.shape == (2, 64, 8, 8)
    assert block.downsample is not None


def test_resnet18_block_layout_and_param_count():
    model = build_backbone({"block_layout": [2, 2, 2, 2], "embedding_dim": 512})
    assert len(model.layer1) == 2
    assert len(model.layer2) == 2
    assert len(model.layer3) == 2
    assert len(model.layer4) == 2
    # A standard ResNet-18 has ~11.1M parameters.
    assert 10_000_000 < model.num_parameters() < 12_500_000


def test_forward_features_returns_all_layers():
    model = CustomResNet18()
    x = torch.randn(1, 3, 64, 64)
    features = model.forward_features(x)
    assert set(features.keys()) == {"layer1", "layer2", "layer3", "layer4"}
    assert features["layer4"].shape[1] == 512
