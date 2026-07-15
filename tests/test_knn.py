"""Tests for the non-parametric k-NN embedding-space baseline."""

from __future__ import annotations

import numpy as np

from src.evaluation.knn_baseline import KNNEmbeddingBaseline, weighted_mean_std, weighted_quantile


def test_weighted_quantile_matches_unweighted_for_uniform_weights():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.ones_like(values)
    median = weighted_quantile(values, weights, 0.5)
    assert abs(median - 3.0) < 0.5


def test_weighted_mean_std_basic():
    values = np.array([10.0, 20.0])
    weights = np.array([1.0, 1.0])
    mean, std = weighted_mean_std(values, weights)
    assert abs(mean - 15.0) < 1e-6
    assert std > 0


def test_knn_age_prediction_close_to_neighbor_ages():
    rng = np.random.default_rng(0)
    n = 200
    # Two well-separated clusters, each with a fixed age, so predictions
    # should closely recover the true age of the nearest cluster.
    embeddings = np.concatenate([
        rng.normal(loc=0.0, scale=0.05, size=(n // 2, 8)),
        rng.normal(loc=5.0, scale=0.05, size=(n // 2, 8)),
    ])
    ages = np.concatenate([np.full(n // 2, 20.0), np.full(n // 2, 60.0)])
    genders = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    mask = np.ones(n, dtype=bool)

    knn = KNNEmbeddingBaseline(k=10).fit(embeddings, ages, mask, genders, mask)
    query = np.array([[0.0] * 8])
    result = knn.predict_age(query)
    assert abs(result.q50[0] - 20.0) < 3.0
    assert result.q10[0] <= result.q50[0] <= result.q90[0]


def test_knn_age_prediction_never_widens_past_age_bounds():
    """The interval-widening step (mean - std*widen_factor) is not bounded by the
    observed neighbor ages, so young/high-variance/far-away queries could
    previously produce a negative q10. It must be clamped to [age_min, age_max]."""
    rng = np.random.default_rng(3)
    n = 200
    embeddings = rng.normal(loc=0.0, scale=1.0, size=(n, 8))
    ages = np.clip(rng.normal(loc=4.0, scale=4.0, size=n), 0, 120)
    genders = rng.integers(0, 2, n).astype(float)
    mask = np.ones(n, dtype=bool)

    knn = KNNEmbeddingBaseline(k=15, age_min=0.0, age_max=120.0).fit(embeddings, ages, mask, genders, mask)
    query = rng.normal(size=(20, 8)) * 3  # far from training distribution -> large widen_factor
    result = knn.predict_age(query)

    assert result.q10.min() >= 0.0
    assert result.q90.max() <= 120.0


def test_knn_gender_prediction_class_probabilities_sum_to_one():
    rng = np.random.default_rng(1)
    n = 100
    embeddings = rng.normal(size=(n, 8))
    ages = rng.uniform(0, 80, n)
    genders = rng.integers(0, 2, n).astype(float)
    mask = np.ones(n, dtype=bool)

    knn = KNNEmbeddingBaseline(k=15).fit(embeddings, ages, mask, genders, mask)
    query = rng.normal(size=(5, 8))
    result = knn.predict_gender(query, confidence_threshold=0.80)
    assert result.probabilities.shape == (5, 2)
    np.testing.assert_allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert result.abstain.dtype == bool


def test_knn_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    n = 50
    embeddings = rng.normal(size=(n, 8))
    ages = rng.uniform(0, 80, n)
    genders = rng.integers(0, 2, n).astype(float)
    mask = np.ones(n, dtype=bool)
    knn = KNNEmbeddingBaseline(k=5).fit(embeddings, ages, mask, genders, mask)

    path = tmp_path / "knn.pkl"
    knn.save(path)
    loaded = KNNEmbeddingBaseline.load(path)
    query = rng.normal(size=(3, 8))
    result_original = knn.predict_age(query)
    result_loaded = loaded.predict_age(query)
    np.testing.assert_allclose(result_original.q50, result_loaded.q50)
