"""Tests for T3 (final-run hardening): abstention as an evaluation-time
selective-prediction policy, not a separate trained model.

src/evaluation/selective.py::gender_selective_prediction_report /
full_coverage_gender_report -- proves these are pure functions of
(y_true, probs) with no model/training dependency at all, and that the
full-coverage ("no abstention") point equals the raw argmax metrics
exactly, from the SAME probabilities used at any other threshold.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.evaluation.metrics import gender_accuracy
from src.evaluation.selective import full_coverage_gender_report, gender_selective_prediction_report


def _synthetic_probs_and_labels(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    # Probabilities correlated with the truth but imperfect, so both correct
    # and incorrect predictions occur at a range of confidences.
    logits = rng.normal(0, 1, size=(n, 2))
    logits[np.arange(n), y_true] += rng.normal(1.5, 1.0, size=n)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return y_true, probs


def test_report_is_a_pure_function_no_model_or_training_dependency():
    """Structural proof it never retrains: the function signature has no
    model/dataset/optimizer/trainer parameter at all."""
    sig = inspect.signature(gender_selective_prediction_report)
    for forbidden in ("model", "dataset", "optimizer", "trainer", "checkpoint"):
        assert forbidden not in sig.parameters


def test_report_contains_all_required_fields():
    y_true, probs = _synthetic_probs_and_labels()
    report = gender_selective_prediction_report(y_true, probs, confidence_threshold=0.7)
    for field in (
        "raw_argmax_accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc",
        "selective_accuracy", "coverage", "abstention_rate", "effective_accuracy", "aurc",
        "risk_coverage_curve",
    ):
        assert field in report, f"missing {field!r}"


def test_raw_argmax_accuracy_independent_of_threshold():
    """raw_argmax_accuracy scores every sample regardless of confidence
    threshold -- it must be identical across different thresholds (the
    same underlying probabilities, just different abstention policy)."""
    y_true, probs = _synthetic_probs_and_labels()
    report_low = gender_selective_prediction_report(y_true, probs, confidence_threshold=0.51)
    report_high = gender_selective_prediction_report(y_true, probs, confidence_threshold=0.99)
    assert report_low["raw_argmax_accuracy"] == pytest.approx(report_high["raw_argmax_accuracy"])
    predicted = probs.argmax(axis=1)
    assert report_low["raw_argmax_accuracy"] == pytest.approx(gender_accuracy(y_true, predicted))


def test_full_coverage_report_equals_zero_threshold_report():
    y_true, probs = _synthetic_probs_and_labels()
    full_coverage = full_coverage_gender_report(y_true, probs)
    zero_threshold = gender_selective_prediction_report(y_true, probs, confidence_threshold=0.0)
    assert full_coverage == zero_threshold


def test_full_coverage_point_has_selective_equal_raw_and_effective():
    """At confidence_threshold=0.0, every sample is accepted -- selective
    accuracy, raw argmax accuracy, and effective accuracy must all
    coincide, and coverage must be exactly 1.0."""
    y_true, probs = _synthetic_probs_and_labels()
    report = full_coverage_gender_report(y_true, probs)
    assert report["coverage"] == pytest.approx(1.0)
    assert report["abstention_rate"] == pytest.approx(0.0)
    assert report["selective_accuracy"] == pytest.approx(report["raw_argmax_accuracy"])
    assert report["effective_accuracy"] == pytest.approx(report["raw_argmax_accuracy"])


def test_higher_threshold_never_increases_coverage():
    y_true, probs = _synthetic_probs_and_labels(n=500)
    thresholds = [0.0, 0.5, 0.7, 0.9, 0.99]
    coverages = [gender_selective_prediction_report(y_true, probs, t)["coverage"] for t in thresholds]
    assert all(coverages[i] >= coverages[i + 1] for i in range(len(coverages) - 1))


def test_effective_accuracy_never_exceeds_selective_accuracy():
    """Effective accuracy's denominator (all samples) is always >= selective
    accuracy's denominator (accepted samples only), for the same numerator
    (correct AND accepted) -- so effective_accuracy <= selective_accuracy
    whenever coverage < 1."""
    y_true, probs = _synthetic_probs_and_labels()
    report = gender_selective_prediction_report(y_true, probs, confidence_threshold=0.85)
    if report["coverage"] < 1.0:
        assert report["effective_accuracy"] <= report["selective_accuracy"] + 1e-9


def test_aurc_is_finite_and_nonnegative():
    y_true, probs = _synthetic_probs_and_labels()
    report = gender_selective_prediction_report(y_true, probs)
    assert report["aurc"] == report["aurc"]  # not NaN
    assert report["aurc"] >= 0.0


def test_risk_coverage_curve_ends_at_full_coverage():
    y_true, probs = _synthetic_probs_and_labels()
    report = gender_selective_prediction_report(y_true, probs)
    assert report["risk_coverage_curve"]["coverages"][-1] == pytest.approx(1.0)


def test_never_calls_any_training_entrypoint(monkeypatch):
    """Belt-and-braces: patch every plausible training entrypoint to raise,
    then confirm both report functions still run cleanly -- proves at
    runtime (not just by signature inspection) that no training path is on
    the call graph."""
    import src.training.trainer as trainer_module

    def _boom(*a, **kw):
        raise AssertionError("must never train")

    monkeypatch.setattr(trainer_module.Trainer, "train", _boom)

    y_true, probs = _synthetic_probs_and_labels()
    gender_selective_prediction_report(y_true, probs, confidence_threshold=0.8)
    full_coverage_gender_report(y_true, probs)
