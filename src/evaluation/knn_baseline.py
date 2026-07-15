"""Non-parametric k-nearest-neighbor baseline in learned embedding space.

Uses ``sklearn.neighbors.NearestNeighbors`` over L2-normalized embeddings
extracted from a trained checkpoint. Age predictions use distance-weighted
neighbor statistics (weighted mean and weighted q10/q50/q90), with
intervals additionally widened when neighbors are far away or
inconsistent with each other. Gender-label predictions use
distance-weighted voting with the same confidence threshold / "Not sure"
behavior as the parametric model.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

_EPS = 1e-8


def _normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, _EPS, None)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cum_weights = np.cumsum(weights) - 0.5 * weights
    cum_weights /= np.sum(weights)
    return float(np.interp(quantile, cum_weights, values))


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mean = float(np.sum(values * weights) / np.sum(weights))
    variance = float(np.sum(weights * (values - mean) ** 2) / np.sum(weights))
    return mean, float(np.sqrt(max(variance, 0.0)))


@dataclass
class KNNAgeResult:
    q10: np.ndarray
    q50: np.ndarray
    q90: np.ndarray
    weighted_mean: np.ndarray
    neighbor_std: np.ndarray
    mean_distance: np.ndarray


@dataclass
class KNNGenderResult:
    probabilities: np.ndarray  # (n, num_classes)
    predicted_class: np.ndarray
    confidence: np.ndarray
    abstain: np.ndarray


class KNNEmbeddingBaseline:
    """A fitted k-NN index plus the labels needed for age and gender prediction."""

    def __init__(
        self,
        k: int = 15,
        distance_weighted: bool = True,
        metric: str = "euclidean",
        age_min: float = 0.0,
        age_max: float = 120.0,
    ) -> None:
        self.k = k
        self.distance_weighted = distance_weighted
        self.metric = metric
        self.age_min = age_min
        self.age_max = age_max
        self.age_index: NearestNeighbors | None = None
        self.gender_index: NearestNeighbors | None = None
        self.age_values: np.ndarray | None = None
        self.gender_values: np.ndarray | None = None
        self.age_distance_scale: float = 1.0
        self.num_classes: int = 2

    def fit(
        self,
        embeddings: np.ndarray,
        ages: np.ndarray,
        age_mask: np.ndarray,
        gender_labels: np.ndarray,
        gender_mask: np.ndarray,
        num_classes: int = 2,
    ) -> "KNNEmbeddingBaseline":
        embeddings = _normalize(embeddings)
        self.num_classes = num_classes

        if age_mask.any():
            age_embeddings = embeddings[age_mask]
            self.age_values = ages[age_mask]
            k_age = min(self.k, len(age_embeddings))
            self.age_index = NearestNeighbors(n_neighbors=k_age, metric=self.metric).fit(age_embeddings)
            sample = age_embeddings[: min(500, len(age_embeddings))]
            pairwise = self.age_index.kneighbors(sample, n_neighbors=min(k_age, 5))[0]
            self.age_distance_scale = float(np.median(pairwise[:, 1:])) if pairwise.shape[1] > 1 else 1.0
            self.age_distance_scale = max(self.age_distance_scale, _EPS)

        if gender_mask.any():
            gender_embeddings = embeddings[gender_mask]
            self.gender_values = gender_labels[gender_mask]
            k_gender = min(self.k, len(gender_embeddings))
            self.gender_index = NearestNeighbors(n_neighbors=k_gender, metric=self.metric).fit(gender_embeddings)

        return self

    def _weights(self, distances: np.ndarray) -> np.ndarray:
        if not self.distance_weighted:
            return np.ones_like(distances)
        weights = 1.0 / (distances + _EPS)
        return weights

    def predict_age(self, query_embeddings: np.ndarray) -> KNNAgeResult:
        if self.age_index is None:
            raise RuntimeError("k-NN age index was not fit (no age-labeled samples)")
        query_embeddings = _normalize(query_embeddings)
        distances, indices = self.age_index.kneighbors(query_embeddings)
        weights = self._weights(distances)

        q10s, q50s, q90s, means, stds = [], [], [], [], []
        for row_dist, row_idx, row_w in zip(distances, indices, weights):
            neighbor_ages = self.age_values[row_idx]
            norm_w = row_w / row_w.sum()
            mean, std = weighted_mean_std(neighbor_ages, norm_w)
            q10 = weighted_quantile(neighbor_ages, norm_w, 0.10)
            q50 = weighted_quantile(neighbor_ages, norm_w, 0.50)
            q90 = weighted_quantile(neighbor_ages, norm_w, 0.90)

            mean_distance = float(row_dist.mean())
            widen_factor = 1.0 + mean_distance / self.age_distance_scale
            age_min = getattr(self, "age_min", 0.0)
            age_max = getattr(self, "age_max", 120.0)
            widened_q10 = np.clip(min(q10, mean - std * widen_factor), age_min, age_max)
            widened_q90 = np.clip(max(q90, mean + std * widen_factor), age_min, age_max)

            q10s.append(widened_q10)
            q50s.append(q50)
            q90s.append(widened_q90)
            means.append(mean)
            stds.append(std)

        return KNNAgeResult(
            q10=np.array(q10s), q50=np.array(q50s), q90=np.array(q90s),
            weighted_mean=np.array(means), neighbor_std=np.array(stds),
            mean_distance=distances.mean(axis=1),
        )

    def predict_gender(self, query_embeddings: np.ndarray, confidence_threshold: float = 0.80) -> KNNGenderResult:
        if self.gender_index is None:
            raise RuntimeError("k-NN gender index was not fit (no gender-labeled samples)")
        query_embeddings = _normalize(query_embeddings)
        distances, indices = self.gender_index.kneighbors(query_embeddings)
        weights = self._weights(distances)

        probs = np.zeros((len(query_embeddings), self.num_classes))
        for row, (row_idx, row_w) in enumerate(zip(indices, weights)):
            neighbor_labels = self.gender_values[row_idx].astype(int)
            norm_w = row_w / row_w.sum()
            for c in range(self.num_classes):
                probs[row, c] = norm_w[neighbor_labels == c].sum()

        predicted = probs.argmax(axis=1)
        confidence = probs.max(axis=1)
        abstain = confidence < confidence_threshold
        return KNNGenderResult(probabilities=probs, predicted_class=predicted, confidence=confidence, abstain=abstain)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> "KNNEmbeddingBaseline":
        with open(path, "rb") as fh:
            return pickle.load(fh)
