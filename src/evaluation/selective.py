"""Generic selective-prediction analysis: risk-coverage curves, AURC, and paired bootstrap CIs.

Used identically for both tasks in this project:

* Gender-label selective prediction -- confidence score = max class
  probability, per-sample loss = 0/1 error indicator, so "risk" at a given
  coverage is the error rate (1 - selective accuracy) among the most
  confident fraction of samples.
* Age selective prediction -- confidence score = -(q90 - q10) (narrower
  raw interval = higher confidence), per-sample loss = absolute age
  error, so "risk" at a given coverage is the mean absolute error among
  the most confident fraction of samples.

Nothing here is task-specific; :func:`selective_risk_coverage_curve` only
needs a confidence score (higher = more confident) and a per-sample loss,
both already computed by the caller. This keeps age and gender selective
analyses from duplicating the same sort-and-sweep logic twice.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    gender_accuracy,
    gender_balanced_accuracy,
    gender_coverage,
    gender_effective_accuracy,
    gender_precision_recall_f1,
    gender_roc_auc,
)


def selective_risk_coverage_curve(
    confidence_scores: np.ndarray, per_sample_loss: np.ndarray, n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep coverage from low to full, keeping the most-confident fraction at each step.

    Returns ``(coverages, risks)``, both ascending in coverage, where
    ``risks[i]`` is the mean of ``per_sample_loss`` over the
    ``coverages[i]``-fraction of samples with the highest
    ``confidence_scores``. ``coverages`` always ends at 1.0 (full
    coverage, i.e. every sample accepted).
    """
    n = len(confidence_scores)
    if n == 0:
        raise ValueError("Cannot compute a risk-coverage curve on zero samples")
    order = np.argsort(-confidence_scores)  # most confident first
    sorted_loss = per_sample_loss[order]
    cumulative_loss = np.cumsum(sorted_loss)

    n_points = min(n_points, n)
    counts = np.unique(np.linspace(1, n, n_points).astype(int))
    coverages = counts / n
    risks = cumulative_loss[counts - 1] / counts
    return coverages, risks


def compute_aurc(coverages: np.ndarray, risks: np.ndarray) -> float:
    """Area under the risk-coverage curve (lower is better). Trapezoidal integration over coverage in [coverages[0], 1]."""
    order = np.argsort(coverages)
    return float(np.trapz(risks[order], coverages[order]))


def risk_at_coverage(coverages: np.ndarray, risks: np.ndarray, target_coverage: float) -> float:
    """Linearly interpolate the risk at an arbitrary target coverage level."""
    order = np.argsort(coverages)
    return float(np.interp(target_coverage, coverages[order], risks[order]))


def gender_selective_prediction_report(
    y_true: np.ndarray, probs: np.ndarray, confidence_threshold: float = 0.80, n_risk_coverage_points: int = 20,
) -> dict:
    """The full evaluation-time selective-prediction report for gender-label
    predictions, from the SAME probabilities (i.e. the same already-trained
    checkpoint's already-computed softmax outputs) at a chosen confidence
    threshold -- this function never trains or re-runs inference, it only
    varies how those existing probabilities are thresholded/summarized.

    Abstention here is exactly what it says: an evaluation-time policy
    applied to a fixed model's confidence, not a separate model. The
    ``confidence_threshold=0.0`` case (every prediction accepted, i.e. "no
    abstention") is simply the full-coverage point of this same report --
    ``coverage == 1.0``, ``selective_accuracy == raw_argmax_accuracy ==
    effective_accuracy`` -- never a second training run with abstention
    "disabled". See :func:`full_coverage_gender_report` for that point
    computed directly.

    Reuses the project's existing metric primitives
    (``src/evaluation/metrics.py``) rather than re-deriving any of them;
    this function only assembles them into one report and adds the
    risk-coverage curve / AURC summary via :func:`selective_risk_coverage_curve`
    / :func:`compute_aurc`.
    """
    predicted = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    abstain_mask = confidence < confidence_threshold
    prf = gender_precision_recall_f1(y_true, predicted)

    per_sample_error = (predicted != y_true).astype(float)
    coverages, risks = selective_risk_coverage_curve(confidence, per_sample_error, n_points=n_risk_coverage_points)
    aurc = compute_aurc(coverages, risks)

    return {
        "confidence_threshold": confidence_threshold,
        "raw_argmax_accuracy": gender_accuracy(y_true, predicted),  # no abstain_mask -- every sample scored
        "balanced_accuracy": gender_balanced_accuracy(y_true, predicted),
        "precision": prf["precision"], "recall": prf["recall"], "f1": prf["f1"],
        "roc_auc": gender_roc_auc(y_true, probs[:, 1]) if probs.shape[1] > 1 else None,
        "selective_accuracy": gender_accuracy(y_true, predicted, abstain_mask),
        "coverage": gender_coverage(abstain_mask),
        "abstention_rate": float(np.mean(abstain_mask)),
        "effective_accuracy": gender_effective_accuracy(y_true, predicted, abstain_mask),
        "aurc": aurc,
        "risk_coverage_curve": {"coverages": coverages.tolist(), "risks": risks.tolist()},
    }


def full_coverage_gender_report(y_true: np.ndarray, probs: np.ndarray, n_risk_coverage_points: int = 20) -> dict:
    """The full-coverage ("no abstention") point of :func:`gender_selective_prediction_report`
    -- confidence_threshold=0.0 accepts every prediction, so
    ``selective_accuracy == raw_argmax_accuracy == effective_accuracy`` and
    ``coverage == 1.0``, all from the identical checkpoint/probabilities as
    any other threshold's report, never a retrained model."""
    return gender_selective_prediction_report(y_true, probs, confidence_threshold=0.0, n_risk_coverage_points=n_risk_coverage_points)


def paired_bootstrap_risk_diff_ci(
    confidence_a: np.ndarray, loss_a: np.ndarray,
    confidence_b: np.ndarray, loss_b: np.ndarray,
    target_coverage: float, n_bootstrap: int = 1000, seed: int = 42, alpha: float = 0.05,
) -> dict:
    """Paired bootstrap confidence interval for (risk_b - risk_a) at a fixed coverage level.

    "Paired" means both models are resampled using the *same* bootstrap
    sample indices in each iteration -- valid because both models were
    evaluated on the identical, index-aligned test set. Models must be
    compared at the same coverage, not at each model's own arbitrary
    confidence threshold, since a model can trade risk for coverage (or
    vice versa) and an unpaired single-threshold comparison would confound
    that trade-off with genuine accuracy differences.

    Returns a dict with the point estimate, the ``(1 - alpha)`` confidence
    interval bounds, and whether that interval excludes zero (a common,
    simple significance check).
    """
    if len(confidence_a) != len(confidence_b):
        raise ValueError("Paired bootstrap requires both models to share the same (index-aligned) test samples")
    n = len(confidence_a)
    rng = np.random.default_rng(seed)

    risk_a_point = risk_at_coverage(*selective_risk_coverage_curve(confidence_a, loss_a), target_coverage)
    risk_b_point = risk_at_coverage(*selective_risk_coverage_curve(confidence_b, loss_b), target_coverage)

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        risk_a = risk_at_coverage(*selective_risk_coverage_curve(confidence_a[idx], loss_a[idx]), target_coverage)
        risk_b = risk_at_coverage(*selective_risk_coverage_curve(confidence_b[idx], loss_b[idx]), target_coverage)
        diffs[i] = risk_b - risk_a

    lower = float(np.percentile(diffs, 100 * (alpha / 2)))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return {
        "coverage": target_coverage,
        "risk_a": risk_a_point,
        "risk_b": risk_b_point,
        "risk_diff_b_minus_a": risk_b_point - risk_a_point,
        "ci_lower": lower,
        "ci_upper": upper,
        "excludes_zero": (lower > 0) or (upper < 0),
        "n_bootstrap": n_bootstrap,
    }


def paired_bootstrap_aurc_diff_ci(
    confidence_a: np.ndarray, loss_a: np.ndarray,
    confidence_b: np.ndarray, loss_b: np.ndarray,
    n_bootstrap: int = 1000, seed: int = 42, alpha: float = 0.05,
) -> dict:
    """Paired bootstrap confidence interval for (AURC_b - AURC_a), the scalar summary statistic.

    :func:`paired_bootstrap_risk_diff_ci` only supports a claim about risk
    *at one fixed coverage level*; it is not evidence about the AURC
    summary statistic itself (the area under the *whole* risk-coverage
    curve). A claim like "model B has a statistically lower AURC than
    model A" must be backed by a CI computed on AURC directly, which is
    what this function provides -- same paired-resampling logic (both
    models resampled with identical bootstrap indices each iteration,
    valid only because both were evaluated on the identical,
    index-aligned test set).
    """
    if len(confidence_a) != len(confidence_b):
        raise ValueError("Paired bootstrap requires both models to share the same (index-aligned) test samples")
    n = len(confidence_a)
    rng = np.random.default_rng(seed)

    aurc_a_point = compute_aurc(*selective_risk_coverage_curve(confidence_a, loss_a))
    aurc_b_point = compute_aurc(*selective_risk_coverage_curve(confidence_b, loss_b))

    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        aurc_a = compute_aurc(*selective_risk_coverage_curve(confidence_a[idx], loss_a[idx]))
        aurc_b = compute_aurc(*selective_risk_coverage_curve(confidence_b[idx], loss_b[idx]))
        diffs[i] = aurc_b - aurc_a

    lower = float(np.percentile(diffs, 100 * (alpha / 2)))
    upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return {
        "aurc_a": aurc_a_point,
        "aurc_b": aurc_b_point,
        "aurc_diff_b_minus_a": aurc_b_point - aurc_a_point,
        "ci_lower": lower,
        "ci_upper": upper,
        "excludes_zero": (lower > 0) or (upper < 0),
        "n_bootstrap": n_bootstrap,
    }
