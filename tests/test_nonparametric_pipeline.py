"""Tests for src/evaluation/nonparametric/pipeline.py and kernels.py (T4,
final-run hardening) -- train-only scaler/PCA fitting, validation-only
grid search (saving all candidates), safe PCA dimensionality clamping, and
the underlying kernel methods' numerical safety. Synthetic arrays only.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.nonparametric.kernels import ClassConditionalKDEClassifier, NadarayaWatsonRegressor
from src.evaluation.nonparametric.pipeline import (
    fit_feature_pipeline,
    safe_n_components,
    tune_kde_gender,
    tune_kernel_regression_age,
    tune_knn_age,
    tune_knn_gender,
)


def _synthetic_features(n=100, d=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    age = X[:, 0] * 10 + 40 + rng.normal(scale=2, size=n)
    gender = (X[:, 1] > 0).astype(int)
    return X, age, gender


# -- safe_n_components ---------------------------------------------------------------


def test_safe_n_components_clamps_to_dataset_size():
    assert safe_n_components(200, n_train_samples=50, n_features=100) == 50
    assert safe_n_components(200, n_train_samples=500, n_features=30) == 30
    assert safe_n_components(10, n_train_samples=500, n_features=100) == 10


def test_safe_n_components_never_below_one():
    assert safe_n_components(0, n_train_samples=5, n_features=5) >= 1
    assert safe_n_components(-5, n_train_samples=5, n_features=5) >= 1


# -- fit_feature_pipeline: train-only fitting -----------------------------------------


def test_pipeline_pca_fit_only_on_train_not_refit_on_transform():
    X_train, _, _ = _synthetic_features(n=80, d=15, seed=1)
    X_other, _, _ = _synthetic_features(n=40, d=15, seed=2)  # different distribution

    pipeline = fit_feature_pipeline(X_train, feature_source="raw_pca", n_components=5)
    # The fitted scaler/PCA statistics must be train-derived, not recomputed
    # from X_other -- verify by checking transform is a pure function of the
    # already-fit scaler/PCA (same input always gives same output,
    # regardless of what other data has been transformed in between).
    out_1 = pipeline.transform(X_other)
    out_2 = pipeline.transform(X_other)
    assert np.allclose(out_1, out_2)
    # And the scaler's fit statistics match X_train's mean, not X_other's.
    assert np.allclose(pipeline.scaler.mean_, X_train.mean(axis=0))
    assert not np.allclose(pipeline.scaler.mean_, X_other.mean(axis=0))


def test_pipeline_l2_normalizes_when_requested():
    X_train, _, _ = _synthetic_features(n=50, d=10, seed=3)
    pipeline = fit_feature_pipeline(X_train, feature_source="raw_pca", n_components=5, l2_normalize=True)
    transformed = pipeline.transform(X_train)
    norms = np.linalg.norm(transformed, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_pipeline_without_pca_keeps_full_dimensionality():
    X_train, _, _ = _synthetic_features(n=50, d=10, seed=4)
    pipeline = fit_feature_pipeline(X_train, feature_source="frozen_backbone", n_components=None)
    transformed = pipeline.transform(X_train)
    assert transformed.shape[1] == 10
    assert pipeline.pca is None


def test_pipeline_provenance_records_actual_dimensionality_used():
    X_train, _, _ = _synthetic_features(n=20, d=50, seed=5)  # more requested components than samples
    pipeline = fit_feature_pipeline(X_train, feature_source="raw_pca", n_components=200)
    provenance = pipeline.provenance()
    assert provenance["n_components_requested"] == 200
    assert provenance["n_components_used"] <= 20  # clamped to n_train_samples


# -- validation-only grid search: k-NN --------------------------------------------------


def test_tune_knn_age_selects_validation_optimal_and_saves_all_candidates():
    X_train, y_train, _ = _synthetic_features(n=60, seed=6)
    X_val, y_val, _ = _synthetic_features(n=20, seed=7)
    best, candidates = tune_knn_age(X_train, y_train, X_val, y_val, k_values=[1, 5, 10], metrics=["euclidean"])
    assert len(candidates) == 3
    assert best["val_mae"] == min(c["val_mae"] for c in candidates)
    assert "k" in best and "metric" in best


def test_tune_knn_gender_uses_balanced_accuracy_not_raw_accuracy():
    X_train, _, y_train = _synthetic_features(n=80, seed=8)
    X_val, _, y_val = _synthetic_features(n=30, seed=9)
    best, candidates = tune_knn_gender(X_train, y_train, X_val, y_val, k_values=[1, 5], metrics=["euclidean", "cosine"])
    assert len(candidates) == 4
    assert best["val_balanced_accuracy"] == max(c["val_balanced_accuracy"] for c in candidates)
    assert "val_balanced_accuracy" in candidates[0]
    assert "val_accuracy" not in candidates[0]  # never selected on plain accuracy


def test_knn_k_adapted_to_small_dataset_size():
    """k > n_train_samples must not raise -- adapted (k_effective) rather than crashing."""
    X_train, y_train, _ = _synthetic_features(n=3, seed=10)
    X_val, y_val, _ = _synthetic_features(n=2, seed=11)
    best, candidates = tune_knn_age(X_train, y_train, X_val, y_val, k_values=[1, 50], metrics=["euclidean"])
    assert all(c["k_effective"] <= 3 for c in candidates)


# -- validation-only grid search: kernel methods ----------------------------------------


def test_tune_kernel_regression_age_selects_validation_optimal():
    X_train, y_train, _ = _synthetic_features(n=60, d=5, seed=12)
    X_val, y_val, _ = _synthetic_features(n=20, d=5, seed=13)
    best, candidates = tune_kernel_regression_age(X_train, y_train, X_val, y_val, bandwidth_scales=[0.5, 1.0, 2.0])
    assert len(candidates) == 3
    assert best["val_mae"] == min(c["val_mae"] for c in candidates)
    assert best["bandwidth"] > 0


def test_tune_kde_gender_selects_validation_optimal():
    X_train, _, y_train = _synthetic_features(n=80, d=5, seed=14)
    X_val, _, y_val = _synthetic_features(n=30, d=5, seed=15)
    best, candidates = tune_kde_gender(X_train, y_train, X_val, y_val, bw_methods=[0.5, 1.0, "scott"])
    assert len(candidates) == 3
    assert best["val_balanced_accuracy"] == max(c["val_balanced_accuracy"] for c in candidates)


# -- kernel methods: reduced-dimensional safety ------------------------------------------


def test_nadaraya_watson_never_full_dimensional_by_construction():
    """Regression guard: NadarayaWatsonRegressor doesn't reduce dimensionality
    itself -- callers (tune_kernel_regression_age via the pipeline) are
    responsible for PCA-reducing first. This test documents the contract:
    passing high-dimensional data works (no crash) but is the CALLER's
    methodological error, not this class's job to prevent -- the real
    safety net is that scripts/tune_nonparametric_baselines.py always
    PCA-reduces before calling this."""
    X_train = np.random.default_rng(0).normal(size=(20, 512))  # simulates raw ResNet-18 dim
    y_train = np.random.default_rng(1).normal(size=20)
    model = NadarayaWatsonRegressor(bandwidth=5.0).fit(X_train, y_train)
    preds = model.predict(X_train[:3])
    assert preds.shape == (3,)  # doesn't crash; still not the recommended usage


def test_kde_classifier_handles_missing_class_without_crashing():
    X = np.random.default_rng(0).normal(size=(20, 3))
    y = np.zeros(20, dtype=int)  # class 1 never appears
    clf = ClassConditionalKDEClassifier().fit(X, y)
    preds = clf.predict(X[:5])
    assert set(preds.tolist()) <= {0}


def test_kde_classifier_handles_tiny_class_count_without_crashing():
    X = np.random.default_rng(0).normal(size=(6, 10))  # n <= d for at least one class
    y = np.array([0, 0, 0, 1, 1, 1])
    clf = ClassConditionalKDEClassifier().fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == 6
