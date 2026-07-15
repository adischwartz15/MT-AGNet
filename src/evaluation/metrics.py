"""Core age and dataset gender-label evaluation metrics."""

from __future__ import annotations

import numpy as np


def age_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def age_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def age_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def interval_coverage(y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray) -> float:
    """Fraction of samples where ``q_low <= y_true <= q_high``."""
    return float(np.mean((y_true >= q_low) & (y_true <= q_high)))


def mean_interval_width(q_low: np.ndarray, q_high: np.ndarray) -> float:
    return float(np.mean(q_high - q_low))


def median_interval_width(q_low: np.ndarray, q_high: np.ndarray) -> float:
    return float(np.median(q_high - q_low))


def expected_calibration_error_intervals(y_true: np.ndarray, q_low: np.ndarray, q_high: np.ndarray, target_coverage: float = 0.80) -> float:
    """|empirical coverage - target coverage| for the q10-q90 interval."""
    empirical = interval_coverage(y_true, q_low, q_high)
    return float(abs(empirical - target_coverage))


def age_absolute_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.abs(y_true - y_pred)


def age_median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median absolute age error (years) -- robust to the heavy tail age_mae is sensitive to."""
    return float(np.median(age_absolute_errors(y_true, y_pred)))


def age_cumulative_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 5.0) -> float:
    """CS@threshold: fraction of samples with absolute age error <= ``threshold`` years.

    Standard age-estimation literature metric (e.g. CS@5), complementing
    MAE/RMSE with "how often are we close enough" rather than an average
    magnitude of error.
    """
    errors = age_absolute_errors(y_true, y_pred)
    return float(np.mean(errors <= threshold))


def age_error_percentiles(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Median / 90th / 95th percentile of absolute age error.

    Complements age_mae/age_rmse (both mean-based, sensitive to a handful
    of large errors) with distributional detail: two models can share a
    similar MAE while one has a much heavier error tail.
    """
    errors = age_absolute_errors(y_true, y_pred)
    return {
        "median": age_median_absolute_error(y_true, y_pred),
        "p90": float(np.percentile(errors, 90)),
        "p95": float(np.percentile(errors, 95)),
    }


def age_tail_error_rates(
    y_true: np.ndarray, y_pred: np.ndarray, thresholds: tuple[int, ...] = (5, 10, 15, 20),
) -> dict[str, float]:
    """Fraction of samples with absolute age error exceeding each threshold (years)."""
    errors = age_absolute_errors(y_true, y_pred)
    return {f">{t}": float(np.mean(errors > t)) for t in thresholds}


def gender_coverage(abstain_mask: np.ndarray) -> float:
    """Fraction of samples the model actually answers (1 - abstention_rate)."""
    return float(1.0 - np.mean(abstain_mask))


def gender_effective_accuracy(y_true: np.ndarray, y_pred: np.ndarray, abstain_mask: np.ndarray) -> float:
    """Correct *accepted* predictions / all samples (denominator includes abstentions).

    Distinct from gender_accuracy (selective accuracy: correct / accepted
    only, denominator excludes abstentions). A model that abstains on
    every hard case can have high selective accuracy but low effective
    accuracy -- this metric is what actually gets a usable answer right,
    out of everything it was asked.
    """
    if len(y_true) == 0:
        return float("nan")
    accepted_correct = (y_true == y_pred) & (~abstain_mask)
    return float(np.sum(accepted_correct) / len(y_true))


def age_uncertainty_by_bucket(
    y_true: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray,
    bucket_edges: list[int] | None = None,
) -> dict[str, dict]:
    """Per-age-bucket MAE, q10-q90 coverage, and interval width.

    Used for the uncertainty evaluation plots (coverage by bucket, width
    by bucket): a model can look well-calibrated on average while
    over/under-covering specific age ranges, which a single global
    coverage number hides. Buckets with zero samples report ``None`` for
    every metric rather than a fabricated value.
    """
    if bucket_edges is None:
        bucket_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 200]
    result = {}
    for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:]):
        mask = (y_true >= lo) & (y_true < hi)
        label = f"{lo}-{hi if hi < 200 else '120+'}"
        if mask.sum() == 0:
            result[label] = {"count": 0, "mae": None, "coverage": None, "mean_width": None, "median_width": None}
        else:
            result[label] = {
                "count": int(mask.sum()),
                "mae": age_mae(y_true[mask], q50[mask]),
                "coverage": interval_coverage(y_true[mask], q10[mask], q90[mask]),
                "mean_width": mean_interval_width(q10[mask], q90[mask]),
                "median_width": median_interval_width(q10[mask], q90[mask]),
            }
    return result


def select_interval_examples(
    image_paths: np.ndarray, y_true: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray, k: int = 5,
) -> dict[str, list[dict]]:
    """Select the ``k`` narrowest and ``k`` widest q10-q90 intervals in a test set.

    Returns real per-sample records (image path, true age, q10/q50/q90,
    width) for the report's "examples of narrow and wide prediction
    intervals" section -- never synthesized examples. Actual image files
    are not embedded in the report (they may be private/licensed
    dataset images); only their paths and the model's own outputs are
    listed.
    """
    width = q90 - q10
    order = np.argsort(width)

    def _record(i: int) -> dict:
        return {
            "image_path": str(image_paths[i]),
            "true_age": float(y_true[i]),
            "q10": float(q10[i]), "q50": float(q50[i]), "q90": float(q90[i]),
            "width": float(width[i]),
        }

    n = len(width)
    k = min(k, n)
    narrowest = [_record(i) for i in order[:k]]
    widest = [_record(i) for i in order[::-1][:k]]
    return {"narrowest": narrowest, "widest": widest}


def gender_accuracy(y_true: np.ndarray, y_pred: np.ndarray, abstain_mask: np.ndarray | None = None) -> float:
    """Accuracy computed only over non-abstained predictions."""
    if abstain_mask is not None:
        keep = ~abstain_mask
        if keep.sum() == 0:
            return float("nan")
        y_true, y_pred = y_true[keep], y_pred[keep]
    return float(np.mean(y_true == y_pred))


def abstention_rate(abstain_mask: np.ndarray) -> float:
    return float(np.mean(abstain_mask))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 2) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[int(t), int(p)] += 1
    return matrix


def gender_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean of per-class recall -- unlike raw accuracy, insensitive to class imbalance."""
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(y_true, y_pred))


def gender_precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary", pos_label: int = 1,
) -> dict[str, float]:
    """Precision/recall/F1 for the gender-label head.

    ``average="binary"`` (default) scores the ``pos_label`` class only,
    matching how a 2-class dataset gender-label head is normally read.
    ``zero_division=0`` avoids a warning/exception when a class is entirely
    absent or entirely unpredicted (e.g. a tiny smoke-test split).
    """
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, pos_label=pos_label, zero_division=0,
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def gender_roc_auc(y_true: np.ndarray, positive_class_probs: np.ndarray) -> float | None:
    """ROC-AUC for the positive-class probability.

    Returns ``None`` (never a fabricated value, and never a swallowed
    exception) when ``y_true`` contains only one class -- AUC is
    mathematically undefined there, which is an expected condition on tiny
    smoke-test splits, not a bug to hide behind a broad except.
    """
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, positive_class_probs))


def confidence_statistics(confidences: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(confidences)),
        "std": float(np.std(confidences)),
        "min": float(np.min(confidences)),
        "max": float(np.max(confidences)),
        "median": float(np.median(confidences)),
    }
