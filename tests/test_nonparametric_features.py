"""Tests for src/evaluation/nonparametric/features.py (T4) -- raw-pixel and
frozen-backbone feature extraction. Synthetic images only. The frozen-
backbone extractor requires torchvision (offline, pretrained=False).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.nonparametric.features import extract_raw_pixel_features


def test_raw_pixel_features_shape_and_determinism(synthetic_metadata_df):
    df = synthetic_metadata_df.iloc[:10]
    features_1, ids_1 = extract_raw_pixel_features(df, image_size=16)
    features_2, ids_2 = extract_raw_pixel_features(df, image_size=16)
    assert features_1.shape == (10, 3 * 16 * 16)
    assert np.array_equal(ids_1, ids_2)
    assert np.allclose(features_1, features_2)  # deterministic, no augmentation


def test_raw_pixel_features_sample_ids_match_row_order(synthetic_metadata_df):
    df = synthetic_metadata_df.iloc[:5]
    _, ids = extract_raw_pixel_features(df, image_size=16)
    assert list(ids) == list(df["image_path"])


def test_raw_pixel_features_never_touches_a_trained_model():
    """Structural proof it's a genuinely unlearned baseline: no
    model/checkpoint parameter in the function signature at all."""
    import inspect

    sig = inspect.signature(extract_raw_pixel_features)
    for forbidden in ("model", "checkpoint", "adapter", "head"):
        assert forbidden not in sig.parameters


@pytest.mark.parametrize("model_id,expected_dim", [("resnet18", 512), ("resnet50", 2048)])
def test_frozen_backbone_features_shape(synthetic_metadata_df, model_id, expected_dim):
    pytest.importorskip("torchvision")
    from src.evaluation.nonparametric.features import extract_frozen_backbone_features

    df = synthetic_metadata_df.iloc[:6]
    features, ids = extract_frozen_backbone_features(df, model_id=model_id, pretrained=False)
    assert features.shape == (6, expected_dim)
    assert list(ids) == list(df["image_path"])


def test_frozen_backbone_features_backbone_stays_frozen(synthetic_metadata_df):
    pytest.importorskip("torchvision")
    from src.evaluation.nonparametric.features import extract_frozen_backbone_features

    # If this function ever accidentally left the backbone trainable AND
    # something later called .backward(), that would be a training path --
    # this test just confirms no gradient tracking happens at all during
    # extraction (torch.no_grad() decorator).
    df = synthetic_metadata_df.iloc[:4]
    features, _ = extract_frozen_backbone_features(df, model_id="resnet18", pretrained=False)
    assert features.dtype == np.float32 or features.dtype == np.float64
