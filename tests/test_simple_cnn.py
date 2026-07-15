"""Tests for the plain-CNN controlled baseline (SimpleCNNBackbone) and its
integration with the backbone factory, MultiTaskFaceModel, and Grad-CAM.

This backbone exists only to answer whether the Custom ResNet-18's
residual connections provide measurable value -- it is never the
project's main architecture, so these tests focus on interface parity
with CustomResNet18 (shape, forward_features naming, Grad-CAM
compatibility) rather than on standalone architecture quality.
"""

from __future__ import annotations

import torch

from src.models.backbone_factory import build_backbone
from src.models.custom_resnet import CustomResNet18
from src.models.multitask_model import MultiTaskFaceModel, build_multitask_model
from src.models.simple_cnn import ConvBlock, SimpleCNNBackbone
from src.evaluation.gradcam import GradCAM


def test_simple_cnn_output_shape():
    model = SimpleCNNBackbone(embedding_dim=512)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 512)


def test_simple_cnn_variable_input_size():
    model = SimpleCNNBackbone(embedding_dim=512)
    for size in (64, 96, 128):
        x = torch.randn(2, 3, size, size)
        out = model(x)
        assert out.shape == (2, 512)


def test_simple_cnn_forward_features_returns_layer1_to_4():
    model = SimpleCNNBackbone()
    x = torch.randn(2, 3, 128, 128)
    features = model.forward_features(x)
    assert set(features.keys()) == {"layer1", "layer2", "layer3", "layer4"}
    # layer4 is the final 512-channel feature map (matches CustomResNet18's layer4).
    assert features["layer4"].shape[1] == 512
    # Spatial resolution shrinks monotonically through the pooled stages.
    assert features["layer1"].shape[-1] > features["layer2"].shape[-1] > features["layer3"].shape[-1]


def test_simple_cnn_has_no_residual_additions():
    """Sanity check that this really is a plain CNN: no submodule adds a shortcut."""
    model = SimpleCNNBackbone()
    for module in model.modules():
        assert not hasattr(module, "downsample"), "SimpleCNNBackbone must not have residual shortcuts"


def test_conv_block_no_pool_variant_preserves_spatial_size():
    block = ConvBlock(8, 16, pool=False)
    x = torch.randn(1, 8, 20, 20)
    out = block(x)
    assert out.shape == (1, 16, 20, 20)


def test_conv_block_pool_variant_halves_spatial_size():
    block = ConvBlock(8, 16, pool=True)
    x = torch.randn(1, 8, 20, 20)
    out = block(x)
    assert out.shape == (1, 16, 10, 10)


def test_simple_cnn_num_parameters_matches_manual_count():
    model = SimpleCNNBackbone()
    manual_count = sum(p.numel() for p in model.parameters())
    assert model.num_parameters() == manual_count
    assert model.num_parameters() > 0


def test_simple_cnn_embedding_dim_configurable():
    model = SimpleCNNBackbone(embedding_dim=256)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert out.shape == (1, 256)


def test_backbone_factory_default_builds_custom_resnet18():
    backbone = build_backbone({"block_layout": [1, 1, 1, 1], "embedding_dim": 32, "stem_channels": 8})
    assert isinstance(backbone, CustomResNet18)


def test_backbone_factory_builds_simple_cnn_when_named():
    backbone = build_backbone({"name": "simple_cnn", "embedding_dim": 64})
    assert isinstance(backbone, SimpleCNNBackbone)
    x = torch.randn(1, 3, 64, 64)
    assert backbone(x).shape == (1, 64)


def test_backbone_factory_rejects_unknown_name():
    import pytest

    with pytest.raises(ValueError):
        build_backbone({"name": "does_not_exist"})


def _tiny_cnn_config(backbone_name: str = "simple_cnn") -> dict:
    return {
        "model": {
            "architecture": "shared_adapters",
            "backbone": {"name": backbone_name, "embedding_dim": 32, "block_layout": [1, 1, 1, 1], "stem_channels": 8},
            "adapters": {"enabled": True, "bottleneck_dim": 8, "dropout": 0.0},
            "age_head": {"hidden_dim": 8, "dropout": 0.0, "age_min": 0, "age_max": 120},
            "gender_head": {"hidden_dim": 8, "dropout": 0.0, "num_classes": 2},
            "loss_balancing": {
                "mode": "learned_uncertainty",
                "learned_uncertainty": {"init_log_var_age": 0.0, "init_log_var_gender": 0.0},
            },
            "pretrained_checkpoint": None,
        }
    }


def test_build_multitask_model_with_simple_cnn_backbone():
    config = _tiny_cnn_config()
    model = build_multitask_model(config)
    assert isinstance(model, MultiTaskFaceModel)
    assert isinstance(model.backbone, SimpleCNNBackbone)
    assert model.backbone_name == "simple_cnn"

    images = torch.randn(2, 3, 64, 64)
    outputs = model(images)
    assert outputs["age_output"]["q50"].shape == (2,)
    assert outputs["gender_logits"].shape == (2, 2)


def test_parameter_breakdown_reports_backbone_name_and_split_heads():
    config = _tiny_cnn_config()
    model = build_multitask_model(config)
    breakdown = model.parameter_breakdown().as_dict()
    assert breakdown["backbone_name"] == "simple_cnn"
    assert breakdown["age_head_parameters"] > 0
    assert breakdown["gender_head_parameters"] > 0
    assert breakdown["total_parameters"] == (
        breakdown["backbone_parameters"] + breakdown["adapter_parameters"]
        + breakdown["age_head_parameters"] + breakdown["gender_head_parameters"]
        + breakdown["log_variance_parameters"]
    )


def test_default_config_still_builds_custom_resnet18_without_regression():
    config = _tiny_cnn_config(backbone_name="custom_resnet18")
    model = build_multitask_model(config)
    assert isinstance(model.backbone, CustomResNet18)
    assert model.backbone_name == "custom_resnet18"
    images = torch.randn(2, 3, 64, 64)
    outputs = model(images)
    assert outputs["gender_logits"].shape == (2, 2)


def test_gradcam_compatible_with_simple_cnn_backbone():
    config = _tiny_cnn_config()
    model = build_multitask_model(config)
    model.eval()
    gradcam = GradCAM(model, target_layer_name="layer4")

    image = torch.randn(1, 3, 64, 64)
    age_result = gradcam.generate(image.clone(), task="age")
    gender_result = gradcam.generate(image.clone(), task="gender")

    assert age_result["heatmap"].ndim == 2
    assert gender_result["heatmap"].ndim == 2
    assert age_result["heatmap"].min() >= 0.0 and age_result["heatmap"].max() <= 1.0 + 1e-6
