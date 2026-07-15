"""Tests for src/models/pretrained_resnet.py -- the required pretrained-
ResNet-18 bridge baseline (+ optional ResNet-50), final-run hardening T3.

Requires a real (offline, ``pretrained=False``) ``torchvision`` install --
skipped entirely where the optional ``requirements-transfer.txt`` extra
isn't installed. Never downloads real ImageNet weights.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("torchvision")

import torch  # noqa: E402

from src.models.pretrained_resnet import (  # noqa: E402
    ALLOWED_PRETRAINED_SOURCES,
    PretrainedResNetFaceOnlyMultiTask,
    PretrainedSourceNotAllowedError,
    UnsupportedResNetModelError,
    build_pretrained_resnet_model,
    validate_pretrained_source,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tiny_config(model_id="resnet18", pretrained_source="imagenet1k_v1"):
    return {
        "model": {
            "family": "pretrained_resnet",
            "pretrained_resnet": {"model_id": model_id, "pretrained": False, "pretrained_source": pretrained_source},
            "adapters": {"enabled": True, "bottleneck_ratio": 4, "dropout": 0.1},
            "age_head": {"hidden_dim": 16, "dropout": 0.1, "age_min": 0, "age_max": 120},
            "gender_head": {"hidden_dim": 16, "dropout": 0.1, "num_classes": 2, "confidence_threshold": 0.80},
            "loss_balancing": {
                "mode": "learned_uncertainty",
                "learned_uncertainty": {"init_log_var_age": 0.0, "init_log_var_gender": 0.0},
            },
        }
    }


def test_resnet18_constructs_offline_and_discovers_embedding_dim():
    model = build_pretrained_resnet_model(_tiny_config("resnet18"))
    assert model.embedding_dim == 512
    assert model.input_size == 224
    assert 0.0 < model.crop_pct <= 1.0


def test_resnet50_constructs_offline_and_discovers_embedding_dim():
    model = build_pretrained_resnet_model(_tiny_config("resnet50", "imagenet1k_v2"))
    assert model.embedding_dim == 2048


def test_forward_pass_shapes():
    model = build_pretrained_resnet_model(_tiny_config())
    dummy = torch.zeros(2, 3, model.input_size, model.input_size)
    out = model(dummy)
    assert out["age_output"]["q50"].shape == (2,)
    assert out["gender_logits"].shape == (2, 2)


def test_unsupported_model_id_rejected():
    config = _tiny_config()
    config["model"]["pretrained_resnet"]["model_id"] = "resnet101"
    with pytest.raises(UnsupportedResNetModelError):
        build_pretrained_resnet_model(config)


def test_pretrained_source_allow_list_enforced():
    validate_pretrained_source("imagenet1k_v1")  # does not raise
    with pytest.raises(PretrainedSourceNotAllowedError):
        validate_pretrained_source("some_face_dataset")
    assert "imagenet1k_v1" in ALLOWED_PRETRAINED_SOURCES
    assert "imagenet1k_v2" in ALLOWED_PRETRAINED_SOURCES


def test_freeze_and_unfreeze_backbone():
    model = build_pretrained_resnet_model(_tiny_config())
    model.freeze_backbone()
    assert all(not p.requires_grad for p in model.backbone.parameters())
    model.unfreeze_backbone()
    assert all(p.requires_grad for p in model.backbone.parameters())


def test_unfreeze_last_stages_leaves_earlier_stages_frozen():
    model = build_pretrained_resnet_model(_tiny_config())
    model.unfreeze_last_stages(1)
    assert all(not p.requires_grad for p in model.backbone.layer1.parameters())
    assert all(p.requires_grad for p in model.backbone.layer4.parameters())


def test_unfreeze_last_stages_rejects_invalid_n():
    model = build_pretrained_resnet_model(_tiny_config())
    with pytest.raises(Exception):
        model.unfreeze_last_stages(0)
    with pytest.raises(Exception):
        model.unfreeze_last_stages(99)


def test_get_parameter_groups_zero_decay_on_log_var():
    model = build_pretrained_resnet_model(_tiny_config())
    groups = model.get_parameter_groups(1e-5, 1e-4, 1e-4, 1e-4, weight_decay=0.05)
    log_var_group = next(g for g in groups if any(p is model.log_var_age for p in g["params"]))
    assert log_var_group["weight_decay"] == 0.0


def test_build_transforms_uses_official_torchvision_preprocessing():
    model = build_pretrained_resnet_model(_tiny_config())
    train_t, eval_t = model.build_transforms()
    assert eval_t.image_size == model.input_size
    assert eval_t.mean == model.mean
    assert eval_t.crop_pct == model.crop_pct


def test_parameter_breakdown_no_adapters():
    config = _tiny_config()
    config["model"]["adapters"]["enabled"] = False
    model = build_pretrained_resnet_model(config)
    breakdown = model.parameter_breakdown().as_dict()
    assert breakdown["adapter_parameters"] == 0


def test_module_never_imports_torchvision_at_module_scope():
    """torchvision must only ever be imported inside a function body, never
    at module scope -- this is what keeps every core experiment importable
    with torchvision completely absent."""
    source = Path(__file__).resolve().parents[1].joinpath("src", "models", "pretrained_resnet.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("torchvision")
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("torchvision")


# -- offline checkpoint loading (via src/inference/artifacts.py) -------------------


def test_offline_reconstruction_never_requests_pretrained_weights(monkeypatch):
    from src.inference.artifacts import _construct_model_offline

    saved_config = _tiny_config()
    saved_config["model"]["pretrained_resnet"]["pretrained"] = True  # as if actually trained pretrained

    import torchvision

    calls = []
    real_resnet18 = torchvision.models.resnet18

    def _spy_resnet18(weights=None, **kwargs):
        calls.append(weights)
        return real_resnet18(weights=None, **kwargs)

    monkeypatch.setattr(torchvision.models, "resnet18", _spy_resnet18)

    import copy as _copy

    before = _copy.deepcopy(saved_config)
    model = _construct_model_offline("pretrained_resnet", saved_config)

    assert calls == [None]  # never requested real pretrained weights
    assert saved_config == before  # original saved config untouched
    assert model is not None


def test_checkpoint_round_trip_reproduces_outputs(tmp_path):
    from src.inference.artifacts import load_model_checkpoint
    from src.training.checkpointing import save_checkpoint

    config = _tiny_config()
    model = build_pretrained_resnet_model(config)
    checkpoint_path = tmp_path / "resnet_best.pt"
    save_checkpoint(checkpoint_path, model, None, epoch=1, metrics={}, config=config)

    reloaded_model, reloaded_config, _ = load_model_checkpoint(checkpoint_path, device="cpu")
    assert reloaded_config["model"]["family"] == "pretrained_resnet"

    model.eval()
    reloaded_model.eval()
    dummy = torch.zeros(2, 3, model.input_size, model.input_size)
    with torch.no_grad():
        out_original = model(dummy)
        out_reloaded = reloaded_model(dummy)
    assert torch.allclose(out_original["age_output"]["q50"], out_reloaded["age_output"]["q50"], atol=1e-5)
