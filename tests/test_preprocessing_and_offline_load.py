"""Tests for T6 (final-run hardening): centralized model-aware preprocessing
(src/data/transforms.py::resolve_eval_transform), crop_pct support, offline
local-checkpoint loading (no pretrained download), and conformal artifact
provenance/rejection (src/evaluation/calibration.py).

CPU-only, synthetic where possible.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch.nn as nn
from PIL import Image

from src.data.transforms import (
    EvalTransform,
    TrainTransform,
    resize_and_center_crop,
    resolve_eval_transform,
    resolve_train_transform,
)
from src.evaluation.calibration import (
    CalibrationMismatchError,
    compute_preprocessing_fingerprint,
    fit_and_save_calibration,
    validate_calibration_artifact,
)


# -- crop_pct ---------------------------------------------------------------------


def test_crop_pct_1_0_is_unchanged_behaviour():
    img = Image.new("RGB", (300, 200), color=(10, 20, 30))
    default = resize_and_center_crop(img, 128)
    explicit = resize_and_center_crop(img, 128, crop_pct=1.0)
    assert list(default.getdata()) == list(explicit.getdata())
    assert default.size == (128, 128)


def test_crop_pct_below_1_resizes_larger_before_cropping():
    """With crop_pct=0.96 (a real value from a pretrained backbone's own
    preprocessing config), the intermediate resize target is
    round(size / crop_pct) > size -- a larger field of view is kept before
    the final crop than with crop_pct=1.0."""
    img = Image.new("RGB", (300, 300), color=(50, 60, 70))
    out = resize_and_center_crop(img, 224, crop_pct=0.96)
    assert out.size == (224, 224)  # final crop size is always `size`
    # round(224 / 0.96) = 233 -- larger than 224, proving a bigger
    # intermediate resize actually happened (indirect check: a crop_pct=1.0
    # crop of a uniformly-colored image is identical in content, so compare
    # against a directly-cropped-smaller reference where content differs at
    # the edges for a non-uniform image).
    gradient = Image.new("RGB", (300, 300))
    for x in range(300):
        for y in range(300):
            gradient.putpixel((x, y), (x % 256, y % 256, 0))
    out_pct = resize_and_center_crop(gradient, 224, crop_pct=0.96)
    out_full = resize_and_center_crop(gradient, 224, crop_pct=1.0)
    assert list(out_pct.getdata()) != list(out_full.getdata())


def test_crop_pct_validated():
    img = Image.new("RGB", (100, 100))
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            resize_and_center_crop(img, 64, crop_pct=bad)


def test_eval_transform_stores_and_uses_crop_pct():
    transform = EvalTransform(224, crop_pct=0.96)
    assert transform.crop_pct == 0.96
    img = Image.new("RGB", (300, 300), color=(1, 2, 3))
    out = transform(img)
    assert out.shape == (3, 224, 224)


# -- resolve_eval_transform / resolve_train_transform ------------------------------


class _FakeModelWithOwnTransforms:
    """Duck-types the build_transforms() contract without needing timm."""

    def build_transforms(self):
        return TrainTransform(224, crop_pct=0.9), EvalTransform(224, crop_pct=0.9)


class _FakeCoreModel(nn.Module):
    pass


def test_resolve_eval_transform_uses_models_own_transform_when_present():
    model = _FakeModelWithOwnTransforms()
    transform = resolve_eval_transform(model, config={"dataset": {"image_size": 128}})
    assert isinstance(transform, EvalTransform)
    assert transform.image_size == 224
    assert transform.crop_pct == 0.9


def test_resolve_eval_transform_falls_back_to_core_default():
    model = _FakeCoreModel()
    transform = resolve_eval_transform(model, config={"dataset": {"image_size": 128}})
    assert isinstance(transform, EvalTransform)
    assert transform.image_size == 128


def test_resolve_train_transform_mirrors_eval_resolution():
    model = _FakeModelWithOwnTransforms()
    transform = resolve_train_transform(model, config={"dataset": {"image_size": 128}})
    assert isinstance(transform, TrainTransform)
    assert transform.image_size == 224


# -- offline checkpoint loading (no pretrained download) ---------------------------


def test_load_model_checkpoint_never_mutates_saved_config(tmp_path):
    """_construct_model_offline must deep-copy before forcing
    pretrained=False -- the returned (saved) config must be untouched."""
    from src.inference.artifacts import _construct_model_offline

    saved_config = {"model": {"family": "core", "architecture": "shared_adapters", "backbone": {"name": "simple_cnn"},
                               "adapters": {"enabled": True, "bottleneck_dim": 8},
                               "age_head": {"hidden_dim": 8, "age_min": 0, "age_max": 120},
                               "gender_head": {"hidden_dim": 8, "num_classes": 2},
                               "loss_balancing": {"mode": "fixed"}}}
    import copy as _copy

    before = _copy.deepcopy(saved_config)
    _construct_model_offline("core", saved_config)
    assert saved_config == before  # untouched


def test_construct_model_offline_rejects_unknown_family():
    from src.inference.artifacts import _construct_model_offline

    with pytest.raises(ValueError, match="Unknown model.family"):
        _construct_model_offline("not_a_real_family", {"model": {}})


def test_load_model_checkpoint_reload_matches_original_outputs(tmp_path):
    """End-to-end (core family, no timm needed): save a checkpoint, reload
    it via load_model_checkpoint, confirm outputs match and the returned
    config equals what was saved (not the reconstruction-only variant)."""
    import torch

    from src.inference.artifacts import load_model_checkpoint
    from src.models.multitask_model import build_multitask_model
    from src.training.checkpointing import save_checkpoint
    from src.utils.config import load_full_config

    config = load_full_config()
    config["dataset"]["image_size"] = 32
    config["model"]["adapters"]["bottleneck_dim"] = 8
    config["model"]["age_head"]["hidden_dim"] = 8
    config["model"]["gender_head"]["hidden_dim"] = 8

    model = build_multitask_model(config)
    checkpoint_path = tmp_path / "core_best.pt"
    save_checkpoint(checkpoint_path, model, None, epoch=1, metrics={}, config=config)

    reloaded_model, reloaded_config, _ = load_model_checkpoint(checkpoint_path, device="cpu")
    assert reloaded_config == config  # unmutated, byte-for-byte the saved config

    model.eval()
    dummy = torch.zeros(2, 3, 32, 32)
    with torch.no_grad():
        out_original = model(dummy)
        out_reloaded = reloaded_model(dummy)
    assert torch.allclose(out_original["age_output"]["q50"], out_reloaded["age_output"]["q50"], atol=1e-6)


# -- conformal provenance / rejection -----------------------------------------------


def test_preprocessing_fingerprint_differs_for_different_crop_pct():
    fp1 = compute_preprocessing_fingerprint(224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 3, crop_pct=1.0)
    fp2 = compute_preprocessing_fingerprint(224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 3, crop_pct=0.96)
    assert fp1 != fp2


def test_preprocessing_fingerprint_deterministic():
    args = (224, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 2, 0.9)
    assert compute_preprocessing_fingerprint(*args) == compute_preprocessing_fingerprint(*args)


def test_validate_calibration_artifact_rejects_model_id_mismatch(tmp_path):
    artifact = fit_and_save_calibration(
        np.array([10.0, 20.0, 30.0]), np.array([5.0, 15.0, 25.0]), np.array([15.0, 25.0, 35.0]),
        alpha=0.1, output_dir=tmp_path, model_id="resnet18_224", pretrained_source="imagenet1k",
        preprocessing_fingerprint="fp-abc",
    )
    validate_calibration_artifact(artifact, model_id="resnet18_224", preprocessing_fingerprint="fp-abc")  # OK

    with pytest.raises(CalibrationMismatchError):
        validate_calibration_artifact(artifact, model_id="resnet18")

    with pytest.raises(CalibrationMismatchError):
        validate_calibration_artifact(artifact, preprocessing_fingerprint="fp-different")


def test_validate_calibration_artifact_skips_unrecorded_fields():
    """An artifact fit before model_id/preprocessing_fingerprint provenance
    existed (both None) must not be rejected just because the caller now
    supplies those fields -- there's nothing on disk to compare against."""
    old_artifact = {"method": "split_conformal_cqr", "offset": 1.0, "model_id": None, "preprocessing_fingerprint": None}
    validate_calibration_artifact(old_artifact, model_id="resnet18_224", preprocessing_fingerprint="fp-xyz")
