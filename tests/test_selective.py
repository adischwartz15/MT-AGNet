"""Tests for the generic selective-prediction analysis (risk-coverage curve, AURC, paired bootstrap CI).

Used identically for gender selective-accuracy and age selective-MAE
analyses in the backbone comparison report -- these tests exercise the
generic mechanics only (monotonic coverage, correct AURC/interpolation
arithmetic, and that models are compared at the same coverage level, not
at independent arbitrary thresholds).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.selective import (
    compute_aurc, paired_bootstrap_aurc_diff_ci, paired_bootstrap_risk_diff_ci, risk_at_coverage,
    selective_risk_coverage_curve,
)


def test_coverage_axis_is_monotonic_and_ends_at_full_coverage():
    rng = np.random.default_rng(0)
    confidence = rng.uniform(0, 1, size=200)
    loss = rng.integers(0, 2, size=200).astype(float)

    coverages, risks = selective_risk_coverage_curve(confidence, loss, n_points=50)

    assert np.all(np.diff(coverages) > 0), "coverage axis must be strictly increasing"
    assert coverages[-1] == pytest.approx(1.0)
    assert coverages[0] > 0
    assert len(coverages) == len(risks)


def test_perfect_confidence_ordering_gives_near_zero_aurc():
    """If the most confident predictions are exactly the correct ones, risk
    should be ~0 at low coverage and only rise as low-confidence (wrong)
    samples are forced in -- giving a low AURC."""
    n = 100
    loss = np.array([0.0] * 80 + [1.0] * 20)  # first 80 correct, last 20 wrong
    confidence = np.linspace(1.0, 0.0, n)  # perfectly ordered: most confident = correct

    coverages, risks = selective_risk_coverage_curve(confidence, loss, n_points=n)
    aurc = compute_aurc(coverages, risks)

    # At 80% coverage, risk should be exactly 0 (only correct samples accepted).
    risk_at_80 = risk_at_coverage(coverages, risks, 0.80)
    assert risk_at_80 == pytest.approx(0.0, abs=1e-6)
    # Overall AURC must be low since risk stays at 0 for most of the curve.
    assert aurc < 0.05


def test_worst_case_confidence_ordering_gives_higher_aurc_than_perfect_ordering():
    n = 100
    loss = np.array([0.0] * 80 + [1.0] * 20)
    good_confidence = np.linspace(1.0, 0.0, n)  # most confident = correct
    bad_confidence = np.linspace(0.0, 1.0, n)  # most confident = wrong (adversarial ordering)

    good_coverages, good_risks = selective_risk_coverage_curve(good_confidence, loss, n_points=n)
    bad_coverages, bad_risks = selective_risk_coverage_curve(bad_confidence, loss, n_points=n)

    assert compute_aurc(bad_coverages, bad_risks) > compute_aurc(good_coverages, good_risks)


def test_risk_at_coverage_interpolates_between_points():
    coverages = np.array([0.5, 1.0])
    risks = np.array([0.0, 1.0])
    # Linear interpolation halfway between the two points.
    assert risk_at_coverage(coverages, risks, 0.75) == pytest.approx(0.5)


def test_paired_bootstrap_ci_detects_a_clear_difference():
    """Model A is always correct (loss=0); model B is always wrong (loss=1).
    At any coverage, risk_b - risk_a should be clearly positive and the CI
    should exclude zero -- this is the "obviously different" sanity case."""
    n = 200
    rng = np.random.default_rng(1)
    confidence_a = rng.uniform(0, 1, size=n)
    confidence_b = rng.uniform(0, 1, size=n)
    loss_a = np.zeros(n)
    loss_b = np.ones(n)

    result = paired_bootstrap_risk_diff_ci(
        confidence_a, loss_a, confidence_b, loss_b, target_coverage=0.90, n_bootstrap=200, seed=0,
    )
    assert result["risk_diff_b_minus_a"] == pytest.approx(1.0)
    assert result["excludes_zero"] is True
    assert result["ci_lower"] > 0


def test_paired_bootstrap_ci_does_not_exclude_zero_for_identical_models():
    """Same confidence/loss for both 'models' -- the difference must be exactly
    zero and the CI must not spuriously exclude it."""
    n = 200
    rng = np.random.default_rng(2)
    confidence = rng.uniform(0, 1, size=n)
    loss = rng.integers(0, 2, size=n).astype(float)

    result = paired_bootstrap_risk_diff_ci(
        confidence, loss, confidence, loss, target_coverage=0.90, n_bootstrap=200, seed=0,
    )
    assert result["risk_diff_b_minus_a"] == pytest.approx(0.0)
    assert result["excludes_zero"] is False


def test_paired_bootstrap_requires_equal_length_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap_risk_diff_ci(
            np.array([0.9, 0.8]), np.array([0.0, 1.0]),
            np.array([0.9]), np.array([0.0]),
            target_coverage=0.9,
        )


def test_paired_bootstrap_aurc_ci_detects_a_clear_difference():
    """Same 'obviously different' sanity case as the fixed-coverage version,
    but for the AURC summary statistic itself (not just one coverage level)
    -- this is what a claim like "model X has lower AURC" must be backed by."""
    n = 200
    rng = np.random.default_rng(1)
    confidence_a = rng.uniform(0, 1, size=n)
    confidence_b = rng.uniform(0, 1, size=n)
    loss_a = np.zeros(n)
    loss_b = np.ones(n)

    result = paired_bootstrap_aurc_diff_ci(confidence_a, loss_a, confidence_b, loss_b, n_bootstrap=200, seed=0)
    # AURC integrates from the curve's first (near-zero, not exactly zero)
    # coverage point to 1.0, so a constant loss=1 model's AURC is just
    # under 1.0 -- not exactly 1.0.
    assert result["aurc_diff_b_minus_a"] == pytest.approx(1.0, abs=0.01)
    assert result["excludes_zero"] is True
    assert result["ci_lower"] > 0


def test_paired_bootstrap_aurc_ci_does_not_exclude_zero_for_identical_models():
    n = 200
    rng = np.random.default_rng(2)
    confidence = rng.uniform(0, 1, size=n)
    loss = rng.integers(0, 2, size=n).astype(float)

    result = paired_bootstrap_aurc_diff_ci(confidence, loss, confidence, loss, n_bootstrap=200, seed=0)
    assert result["aurc_diff_b_minus_a"] == pytest.approx(0.0)
    assert result["excludes_zero"] is False


def test_paired_bootstrap_aurc_requires_equal_length_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap_aurc_diff_ci(
            np.array([0.9, 0.8]), np.array([0.0, 1.0]), np.array([0.9]), np.array([0.0]),
        )


def test_models_compared_at_same_coverage_not_independent_thresholds():
    """Two models with very different confidence distributions must still be
    compared at the *same* requested coverage level, not each model's own
    natural threshold -- risk_at_coverage must respect the exact
    target_coverage argument regardless of each model's score scale."""
    n = 100
    loss = np.array([0.0] * 50 + [1.0] * 50)
    # Model A's confidence scores are all tiny (e.g. near 0); model B's are all huge.
    # Both perfectly ordered relative to loss, just on different scales.
    confidence_a = np.linspace(0.001, 0.0001, n)
    confidence_b = np.linspace(1000.0, 100.0, n)

    curve_a = selective_risk_coverage_curve(confidence_a, loss, n_points=n)
    curve_b = selective_risk_coverage_curve(confidence_b, loss, n_points=n)

    # At the same 50% coverage, both should show ~0 risk (perfectly ordered).
    assert risk_at_coverage(*curve_a, 0.50) == pytest.approx(0.0, abs=1e-6)
    assert risk_at_coverage(*curve_b, 0.50) == pytest.approx(0.0, abs=1e-6)
