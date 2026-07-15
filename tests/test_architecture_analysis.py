"""Tests for gradient interference analysis and linear CKA representation similarity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.architecture_analysis import compute_gradient_cosine_similarity, linear_cka
from src.models.multitask_model import MultiTaskFaceModel


def test_linear_cka_identical_matrices_is_one():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 16))
    assert abs(linear_cka(x, x) - 1.0) < 1e-6


def test_linear_cka_is_symmetric():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 16))
    y = rng.normal(size=(50, 16))
    assert abs(linear_cka(x, y) - linear_cka(y, x)) < 1e-6


def test_linear_cka_scale_invariant():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 16))
    assert abs(linear_cka(x, x * 5.0) - 1.0) < 1e-6


def test_linear_cka_bounded_between_zero_and_one_ish():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 16))
    y = rng.normal(size=(200, 16))
    value = linear_cka(x, y)
    assert -1e-3 <= value <= 1.0 + 1e-3


def _tiny_shared_config():
    return {
        "model": {
            "architecture": "shared_adapters",
            "backbone": {"block_layout": [1, 1, 1, 1], "embedding_dim": 32, "stem_channels": 8},
            "adapters": {"enabled": True, "bottleneck_dim": 8, "dropout": 0.0},
            "age_head": {"hidden_dim": 8, "dropout": 0.0, "age_min": 0, "age_max": 120},
            "gender_head": {"hidden_dim": 8, "dropout": 0.0, "num_classes": 2},
            "loss_balancing": {"mode": "fixed", "fixed": {"age_weight": 1.0, "gender_weight": 1.0}},
        }
    }


def test_gradient_cosine_similarity_shared_backbone():
    config = _tiny_shared_config()
    model = MultiTaskFaceModel(config)

    def fake_loader():
        for _ in range(4):
            yield {
                "image": torch.randn(4, 3, 32, 32),
                "age": torch.rand(4) * 80,
                "age_mask": torch.ones(4, dtype=torch.bool),
                "gender_label": torch.randint(0, 2, (4,)),
                "gender_mask": torch.ones(4, dtype=torch.bool),
            }

    similarities = compute_gradient_cosine_similarity(model, list(fake_loader()), device="cpu")
    assert len(similarities) == 4
    assert np.all(similarities >= -1.0 - 1e-4) and np.all(similarities <= 1.0 + 1e-4)


def test_gradient_cosine_similarity_raises_for_separate_architecture():
    config = _tiny_shared_config()
    config["model"]["architecture"] = "separate"
    model = MultiTaskFaceModel(config)
    with pytest.raises(ValueError):
        compute_gradient_cosine_similarity(model, [], device="cpu")
