"""Tests for the depth/width-matched no-skip-connection backbone (PlainDeep18NoSkip).

Unlike SimpleCNNBackbone (a differently-shaped, smaller CNN used for an
efficiency/accuracy trade-off comparison), this backbone exists to isolate
the causal contribution of residual skip connections specifically: same
stem, stage widths, block layout, and embedding size as CustomResNet18,
with only the identity/projection-shortcut additions removed. These tests
focus on that parity (matching parameter count modulo the unavoidable
downsample-shortcut difference, matching feature-map shapes, no residual
additions) plus interface parity with the backbone factory / model /
Grad-CAM, mirroring tests/test_simple_cnn.py's conventions.
"""

from __future__ import annotations

import torch

from src.evaluation.gradcam import GradCAM
from src.models.backbone_factory import build_backbone
from src.models.custom_resnet import BasicBlock, CustomResNet18
from src.models.multitask_model import MultiTaskFaceModel, build_multitask_model
from src.models.plain_deep18_no_skip import PlainBlock, PlainDeep18NoSkip


def test_plain_deep18_output_shape():
    model = PlainDeep18NoSkip(embedding_dim=512)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 512)


def test_plain_deep18_variable_input_size():
    model = PlainDeep18NoSkip(embedding_dim=512)
    for size in (64, 96, 128):
        x = torch.randn(2, 3, size, size)
        out = model(x)
        assert out.shape == (2, 512)


def test_plain_deep18_forward_features_returns_layer1_to_4():
    model = PlainDeep18NoSkip()
    x = torch.randn(2, 3, 128, 128)
    features = model.forward_features(x)
    assert set(features.keys()) == {"layer1", "layer2", "layer3", "layer4"}
    assert features["layer4"].shape[1] == 512
    assert features["layer1"].shape[-1] > features["layer2"].shape[-1] > features["layer3"].shape[-1]


def test_plain_deep18_has_no_residual_additions():
    """Sanity check that this really has no shortcuts: no submodule adds a downsample."""
    model = PlainDeep18NoSkip()
    for module in model.modules():
        assert not hasattr(module, "downsample"), "PlainDeep18NoSkip must not have residual shortcuts"


def test_plain_deep18_matches_custom_resnet18_shapes_stage_by_stage():
    """Same stem/stage widths/block layout as CustomResNet18 -- only the skip connections differ."""
    resnet = CustomResNet18()
    plain = PlainDeep18NoSkip()
    x = torch.randn(2, 3, 128, 128)
    resnet_features = resnet.forward_features(x)
    plain_features = plain.forward_features(x)
    for name in ("layer1", "layer2", "layer3", "layer4"):
        assert resnet_features[name].shape == plain_features[name].shape


def test_plain_deep18_parameter_difference_matches_downsample_shortcuts_exactly():
    """The only unavoidable parameter difference is CustomResNet18's three
    1x1-conv+BatchNorm downsample shortcuts (layer2/3/4 transitions);
    PlainDeep18NoSkip has strictly fewer parameters by exactly that amount."""
    resnet = CustomResNet18()
    plain = PlainDeep18NoSkip()
    resnet_params = sum(p.numel() for p in resnet.parameters())
    plain_params = sum(p.numel() for p in plain.parameters())

    downsample_params = 0
    for module in resnet.modules():
        if isinstance(module, BasicBlock) and module.downsample is not None:
            downsample_params += sum(p.numel() for p in module.downsample.parameters())

    assert downsample_params > 0
    assert resnet_params - plain_params == downsample_params


def test_plain_block_has_no_downsample_attribute():
    block = PlainBlock(16, 32, stride=2)
    assert not hasattr(block, "downsample")
    x = torch.randn(1, 16, 8, 8)
    out = block(x)
    assert out.shape == (1, 32, 4, 4)


def test_plain_deep18_num_parameters_matches_manual_count():
    model = PlainDeep18NoSkip()
    manual_count = sum(p.numel() for p in model.parameters())
    assert model.num_parameters() == manual_count
    assert model.num_parameters() > 0


def test_backbone_factory_builds_plain_deep18_no_skip_when_named():
    backbone = build_backbone({"name": "plain_deep18_no_skip", "embedding_dim": 64, "block_layout": [1, 1, 1, 1], "stem_channels": 8})
    assert isinstance(backbone, PlainDeep18NoSkip)
    x = torch.randn(1, 3, 64, 64)
    assert backbone(x).shape == (1, 64)


def _tiny_plain_config() -> dict:
    return {
        "model": {
            "architecture": "shared_adapters",
            "backbone": {"name": "plain_deep18_no_skip", "embedding_dim": 32, "block_layout": [1, 1, 1, 1], "stem_channels": 8},
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


def test_build_multitask_model_with_plain_deep18_no_skip_backbone():
    config = _tiny_plain_config()
    model = build_multitask_model(config)
    assert isinstance(model, MultiTaskFaceModel)
    assert isinstance(model.backbone, PlainDeep18NoSkip)
    assert model.backbone_name == "plain_deep18_no_skip"

    images = torch.randn(2, 3, 64, 64)
    outputs = model(images)
    assert outputs["age_output"]["q50"].shape == (2,)
    assert outputs["gender_logits"].shape == (2, 2)


def test_parameter_breakdown_reports_plain_deep18_backbone_name():
    config = _tiny_plain_config()
    model = build_multitask_model(config)
    breakdown = model.parameter_breakdown().as_dict()
    assert breakdown["backbone_name"] == "plain_deep18_no_skip"
    assert breakdown["total_parameters"] == (
        breakdown["backbone_parameters"] + breakdown["adapter_parameters"]
        + breakdown["age_head_parameters"] + breakdown["gender_head_parameters"]
        + breakdown["log_variance_parameters"]
    )


def test_gradcam_compatible_with_plain_deep18_no_skip_backbone():
    config = _tiny_plain_config()
    model = build_multitask_model(config)
    model.eval()
    gradcam = GradCAM(model, target_layer_name="layer4")

    image = torch.randn(1, 3, 64, 64)
    age_result = gradcam.generate(image.clone(), task="age")
    gender_result = gradcam.generate(image.clone(), task="gender")

    assert age_result["heatmap"].ndim == 2
    assert gender_result["heatmap"].ndim == 2
    assert age_result["heatmap"].min() >= 0.0 and age_result["heatmap"].max() <= 1.0 + 1e-6


def test_backbone_factory_still_rejects_unknown_name():
    import pytest

    with pytest.raises(ValueError):
        build_backbone({"name": "does_not_exist"})
