"""Kernel-based (Nadaraya-Watson / class-conditional KDE) non-parametric
methods for the T4 baselines (final-run hardening).

**Dimensionality contract**: every function here is meant to run on
REDUCED-DIMENSIONAL features (post train-only PCA, see
``src/evaluation/nonparametric/pipeline.py``) -- never directly on raw
512/2048/384-d backbone features or raw flattened pixels, which is
statistically unstable (curse of dimensionality: kernel density estimates
degrade badly past a few tens of dimensions) and computationally
impractical (``scipy.stats.gaussian_kde``'s covariance is
``O(d^2)``/``O(d^3)`` to build/invert). Callers are responsible for PCA-
reducing first; nothing here silently reduces dimensionality on the
caller's behalf.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.sum(a**2, axis=1)[:, None]
    bb = np.sum(b**2, axis=1)[None, :]
    return np.maximum(aa + bb - 2 * a @ b.T, 0.0)


def gaussian_kernel_weights(query: np.ndarray, reference: np.ndarray, bandwidth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unnormalized Gaussian kernel weights ``K_h(d(query_i, reference_j))``,
    computed in a numerically stable way (subtracting each row's max
    log-weight before exponentiating, matching the standard "log-sum-exp"
    stability trick). Returns ``(weights, row_sums, underflow_mask)`` --
    ``underflow_mask`` flags query rows whose weights ALL underflowed to
    (numerically) zero, meaning the row sum can't be used as a denominator.
    """
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}.")
    d2 = _pairwise_sq_dists(query, reference)
    log_w = -d2 / (2.0 * bandwidth**2)
    log_w_max = log_w.max(axis=1, keepdims=True)
    log_w_max = np.where(np.isfinite(log_w_max), log_w_max, 0.0)
    w = np.exp(log_w - log_w_max)
    row_sums = w.sum(axis=1, keepdims=True)
    underflow = (row_sums < _EPS).flatten()
    return w, row_sums, underflow


class NadarayaWatsonRegressor:
    """Nadaraya-Watson kernel regression:
    ``yhat(x) = sum_i K_h(d(x, x_i)) y_i / sum_i K_h(d(x, x_i))``,
    Gaussian kernel on Euclidean distance in the (already reduced-
    dimensional) feature space. When every reference weight underflows for
    a query point (it is far from all training points relative to
    ``bandwidth``), falls back to its single nearest neighbor's target
    value rather than returning NaN or dividing by (numerically) zero.
    """

    def __init__(self, bandwidth: float = 1.0):
        if bandwidth <= 0:
            raise ValueError(f"bandwidth must be positive, got {bandwidth}.")
        self.bandwidth = bandwidth
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NadarayaWatsonRegressor":
        self._X = np.asarray(X, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.float64)
        if len(self._X) == 0:
            raise ValueError("Cannot fit NadarayaWatsonRegressor on zero training samples.")
        return self

    def predict(self, X_query: np.ndarray) -> np.ndarray:
        X_query = np.asarray(X_query, dtype=np.float64)
        weights, row_sums, underflow = gaussian_kernel_weights(X_query, self._X, self.bandwidth)
        preds = np.full(len(X_query), np.nan)
        valid = ~underflow
        if valid.any():
            preds[valid] = (weights[valid] @ self._y) / row_sums[valid, 0]
        if underflow.any():
            d2 = _pairwise_sq_dists(X_query[underflow], self._X)
            nearest = np.argmin(d2, axis=1)
            preds[underflow] = self._y[nearest]
        return preds


class ClassConditionalKDEClassifier:
    """Class-conditional KDE Bayes classifier: fits one Gaussian KDE per
    class (via ``scipy.stats.gaussian_kde``) on reduced-dimensional
    features, combines with class priors in log-space, and classifies by
    maximum a posteriori log-density.

    Handles, explicitly (never crashes, never silently mis-scores):

    * **missing class** (zero training samples for a class) -- that
      class's log-density is ``-inf`` everywhere, so it is simply never
      predicted (but does not raise);
    * **tiny class counts** (``n_samples <= n_features``, too few for a
      full-rank KDE covariance) -- falls back to a diagonal-covariance
      Gaussian (regularized with a small variance floor) instead of
      raising or producing a singular/degenerate density;
    * **singular covariance** even with enough samples (e.g. a
      near-duplicate cluster) -- ``scipy``'s ``LinAlgError`` is caught,
      same diagonal-Gaussian fallback;
    * **non-finite likelihoods** -- any ``NaN``/``+-inf`` log-density is
      clamped to ``-inf`` (never silently propagated as a score);
    * **a query row where every class's density is ``-inf``** -- predicts
      the majority-prior class rather than an arbitrary default index.
    """

    def __init__(self, bw_method: float | str | None = None, variance_floor: float = 1e-6):
        self.bw_method = bw_method
        self.variance_floor = variance_floor
        self._kdes: dict = {}
        self._priors: dict = {}
        self._classes: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassConditionalKDEClassifier":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self._classes = sorted(np.unique(y).tolist())
        n_total = len(y)
        for c in self._classes:
            X_c = X[y == c]
            self._priors[c] = len(X_c) / n_total if n_total else 0.0
            self._kdes[c] = self._fit_one_class(X_c)
        return self

    def _fit_one_class(self, X_c: np.ndarray):
        from scipy.stats import gaussian_kde

        n, d = X_c.shape if X_c.ndim == 2 else (0, 0)
        if n == 0:
            return None  # missing class
        if n <= d:
            return ("gaussian_fallback", X_c.mean(axis=0), np.var(X_c, axis=0) + self.variance_floor)
        try:
            return ("kde", gaussian_kde(X_c.T, bw_method=self.bw_method))
        except np.linalg.LinAlgError:
            return ("gaussian_fallback", X_c.mean(axis=0), np.var(X_c, axis=0) + self.variance_floor)

    def _log_density(self, kde_obj, X_query: np.ndarray) -> np.ndarray:
        if kde_obj is None:
            return np.full(len(X_query), -np.inf)
        if kde_obj[0] == "kde":
            kde = kde_obj[1]
            with np.errstate(divide="ignore", invalid="ignore"):
                log_d = kde.logpdf(X_query.T)
            return np.where(np.isfinite(log_d), log_d, -np.inf)
        _, mean, var = kde_obj
        with np.errstate(divide="ignore", invalid="ignore"):
            log_d = -0.5 * np.sum(((X_query - mean) ** 2) / var + np.log(2 * np.pi * var), axis=1)
        return np.where(np.isfinite(log_d), log_d, -np.inf)

    def predict_log_joint(self, X_query: np.ndarray) -> tuple[np.ndarray, list]:
        """Returns ``(log_joint, classes)`` where ``log_joint[i, k]`` is the
        log of ``P(class=classes[k]) * density_k(X_query[i])``."""
        X_query = np.asarray(X_query, dtype=np.float64)
        log_joint = np.full((len(X_query), len(self._classes)), -np.inf)
        for i, c in enumerate(self._classes):
            prior = self._priors[c]
            log_prior = np.log(prior) if prior > 0 else -np.inf
            log_joint[:, i] = self._log_density(self._kdes[c], X_query) + log_prior
        return log_joint, self._classes

    def predict(self, X_query: np.ndarray) -> np.ndarray:
        log_joint, classes = self.predict_log_joint(X_query)
        idx = np.argmax(log_joint, axis=1)
        all_non_finite = ~np.any(np.isfinite(log_joint), axis=1)
        if all_non_finite.any():
            majority_idx = int(np.argmax([self._priors[c] for c in classes]))
            idx[all_non_finite] = majority_idx
        return np.array([classes[i] for i in idx])

    def predict_proba(self, X_query: np.ndarray) -> np.ndarray:
        """Normalized posterior probabilities per class (softmax over the
        log-joint, numerically stable, robust to a row that is all -inf --
        such a row falls back to the class priors)."""
        log_joint, classes = self.predict_log_joint(X_query)
        all_non_finite = ~np.any(np.isfinite(log_joint), axis=1)
        row_max = np.where(all_non_finite, 0.0, np.nanmax(np.where(np.isfinite(log_joint), log_joint, -np.inf), axis=1))
        with np.errstate(over="ignore"):
            unnormalized = np.exp(log_joint - row_max[:, None])
        unnormalized = np.where(np.isfinite(unnormalized), unnormalized, 0.0)
        row_sums = unnormalized.sum(axis=1, keepdims=True)
        probs = np.divide(unnormalized, row_sums, out=np.zeros_like(unnormalized), where=row_sums > 0)
        if all_non_finite.any():
            priors = np.array([self._priors[c] for c in classes])
            priors = priors / priors.sum() if priors.sum() > 0 else np.full(len(classes), 1.0 / len(classes))
            probs[all_non_finite] = priors
        return probs
