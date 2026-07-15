"""Feature-pipeline fitting (StandardScaler -> PCA -> optional L2-norm, all
TRAIN-ONLY) and validation-only hyperparameter grid search for the T4
non-parametric baselines (final-run hardening).

Protocol enforced throughout this module (see docs/transfer_learning.md /
the mission's "train/validation/calibration/test roles for non-parametric
methods"):

* ``fit_feature_pipeline`` is called on **train** features only -- its
  ``StandardScaler``/``PCA`` are never re-fit on validation/calibration/test.
* ``tune_knn_*`` / ``tune_kernel_*`` select hyperparameters using
  **validation** predictions only.
* Nothing here touches calibration or test data at all -- those are the
  caller's job (conformal calibration and final evaluation respectively,
  see ``scripts/evaluate_nonparametric_baselines.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import gender_balanced_accuracy
from src.evaluation.nonparametric.kernels import ClassConditionalKDEClassifier, NadarayaWatsonRegressor

# Documented hyperparameter grids (mission-specified defaults).
K_VALUES = [1, 3, 5, 10, 20, 50]
DISTANCE_METRICS = ["euclidean", "cosine"]
PCA_COMPONENTS_RAW = [50, 100, 200]
PCA_COMPONENTS_BACKBONE = [10, 25, 50, 100]
L2_NORMALIZATION = [False, True]
KDE_BANDWIDTH_SCALES = [0.25, 0.5, 1.0, 2.0, 4.0]  # multiplied by the median pairwise train distance (a Silverman-like heuristic scale)


def safe_n_components(requested: int, n_train_samples: int, n_features: int) -> int:
    """Clamp a requested PCA ``n_components`` to a value ``sklearn.decomposition.PCA``
    can actually satisfy: ``1 <= n_components <= min(n_train_samples, n_features)``.
    Adapts an out-of-range grid value to the dataset size rather than raising."""
    max_allowed = max(1, min(n_train_samples, n_features))
    return max(1, min(int(requested), max_allowed))


@dataclass
class FittedFeaturePipeline:
    """A StandardScaler (+ optional PCA, + optional L2-normalization),
    fit once on TRAIN features. ``transform`` applies the identical
    train-fit statistics to any other split -- this is what makes "PCA
    fitted on train only" enforceable: there is no ``fit`` method exposed
    on non-train data, only ``transform``."""

    scaler: StandardScaler
    pca: PCA | None
    l2_normalize: bool
    feature_source: str  # "raw_pca" | "frozen_backbone"
    n_components_requested: int | None
    n_components_used: int | None

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = self.scaler.transform(X)
        if self.pca is not None:
            X = self.pca.transform(X)
        if self.l2_normalize:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / np.clip(norms, 1e-8, None)
        return X

    def provenance(self) -> dict:
        return {
            "feature_source": self.feature_source,
            "n_components_requested": self.n_components_requested,
            "n_components_used": self.n_components_used,
            "l2_normalize": self.l2_normalize,
            "scaler_mean_shape": list(self.scaler.mean_.shape),
        }


def fit_feature_pipeline(
    X_train: np.ndarray, feature_source: str, n_components: int | None = None, l2_normalize: bool = False,
) -> FittedFeaturePipeline:
    """Fit StandardScaler (+ optional PCA, + optional L2-norm) on TRAIN
    features only. Callers must never pass validation/calibration/test
    features here."""
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)
    pca = None
    n_used = None
    if n_components is not None:
        n_used = safe_n_components(n_components, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n_used, random_state=0).fit(X_scaled)
    return FittedFeaturePipeline(
        scaler=scaler, pca=pca, l2_normalize=l2_normalize, feature_source=feature_source,
        n_components_requested=n_components, n_components_used=n_used,
    )


# -- k-NN grid search (validation-only selection) -----------------------------------


def tune_knn_age(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    k_values: list[int] = K_VALUES, metrics: list[str] = DISTANCE_METRICS,
) -> tuple[dict, list[dict]]:
    """Grid search k-NN age regression hyperparameters on VALIDATION ONLY,
    selecting by minimum validation MAE. Returns ``(best, all_candidates)``
    -- every candidate is saved, not just the winner."""
    candidates = []
    for k in k_values:
        k_eff = min(k, len(X_train))
        for metric in metrics:
            model = KNeighborsRegressor(n_neighbors=k_eff, metric=metric, weights="distance")
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            mae = float(np.mean(np.abs(preds - y_val)))
            candidates.append({"k": k, "k_effective": k_eff, "metric": metric, "val_mae": mae})
    best = min(candidates, key=lambda c: c["val_mae"])
    return best, candidates


def tune_knn_gender(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    k_values: list[int] = K_VALUES, metrics: list[str] = DISTANCE_METRICS,
) -> tuple[dict, list[dict]]:
    """Grid search k-NN gender classification hyperparameters on VALIDATION
    ONLY, selecting by maximum validation BALANCED accuracy (never plain
    accuracy, which can be misleading under class imbalance). Returns
    ``(best, all_candidates)``."""
    candidates = []
    for k in k_values:
        k_eff = min(k, len(X_train))
        for metric in metrics:
            model = KNeighborsClassifier(n_neighbors=k_eff, metric=metric, weights="distance")
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            balanced_acc = gender_balanced_accuracy(y_val, preds)
            candidates.append({"k": k, "k_effective": k_eff, "metric": metric, "val_balanced_accuracy": balanced_acc})
    best = max(candidates, key=lambda c: c["val_balanced_accuracy"])
    return best, candidates


# -- kernel-method grid search (validation-only selection) --------------------------


def _median_pairwise_distance(X: np.ndarray, max_samples: int = 500, seed: int = 0) -> float:
    """A cheap, deterministic bandwidth-scale reference: the median
    pairwise Euclidean distance among (up to ``max_samples``) training
    points -- the standard "median heuristic" starting point for kernel
    bandwidth selection, before the actual grid search below picks among
    scaled multiples of it on validation."""
    rng = np.random.default_rng(seed)
    n = len(X)
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X = X[idx]
    from scipy.spatial.distance import pdist

    distances = pdist(X, metric="euclidean")
    return float(np.median(distances)) if len(distances) else 1.0


def tune_kernel_regression_age(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    bandwidth_scales: list[float] = KDE_BANDWIDTH_SCALES,
) -> tuple[dict, list[dict]]:
    """Grid search Nadaraya-Watson bandwidth on VALIDATION ONLY (minimum
    MAE). Bandwidths are ``bandwidth_scales[i] * median_pairwise_train_distance``
    -- a data-dependent, reproducible grid rather than an arbitrary
    absolute scale that wouldn't transfer across feature spaces."""
    reference_scale = max(_median_pairwise_distance(X_train), 1e-6)
    candidates = []
    for scale in bandwidth_scales:
        bandwidth = scale * reference_scale
        model = NadarayaWatsonRegressor(bandwidth=bandwidth).fit(X_train, y_train)
        preds = model.predict(X_val)
        mae = float(np.mean(np.abs(preds - y_val)))
        candidates.append({"bandwidth_scale": scale, "bandwidth": bandwidth, "val_mae": mae})
    best = min(candidates, key=lambda c: c["val_mae"])
    return best, candidates


def tune_kde_gender(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    bw_methods: list[float | str] = (0.5, 1.0, 2.0, "scott", "silverman"),
) -> tuple[dict, list[dict]]:
    """Grid search class-conditional-KDE ``bw_method`` on VALIDATION ONLY
    (maximum balanced accuracy)."""
    candidates = []
    for bw_method in bw_methods:
        model = ClassConditionalKDEClassifier(bw_method=bw_method).fit(X_train, y_train)
        preds = model.predict(X_val)
        balanced_acc = gender_balanced_accuracy(y_val, preds)
        candidates.append({"bw_method": bw_method, "val_balanced_accuracy": balanced_acc})
    best = max(candidates, key=lambda c: c["val_balanced_accuracy"])
    return best, candidates
