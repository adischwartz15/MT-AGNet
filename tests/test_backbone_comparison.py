"""Tests for the cross-model backbone comparison analysis suite.

Covers: clean-test summary assembly, gender/age selective-risk analysis
(risk-coverage curves, AURC, at-coverage table, paired bootstrap CIs), the
tail-error analysis, and -- most importantly -- that the final
interpretation is honest in both directions: it must say the compact
model is preferred when no measured advantage exists, and must never
claim a ResNet advantage that isn't backed by a statistically significant
(bootstrap CI excludes zero) AURC difference.
"""

from __future__ import annotations

import numpy as np

import pytest

from src.evaluation.backbone_comparison import (
    build_age_selective_analysis, build_clean_test_summary, build_clean_test_table,
    build_final_interpretation, build_gender_risk_coverage_analysis, build_tail_error_analysis,
)


def _synthetic_preds(n=200, age_mae_scale=1.0, gender_error_rate=0.1, seed=0, latency=1.5):
    rng = np.random.default_rng(seed)
    age = rng.uniform(0, 80, size=n)
    q50 = age + rng.normal(0, age_mae_scale, size=n)
    q10 = q50 - rng.uniform(5, 10, size=n)
    q90 = q50 + rng.uniform(5, 10, size=n)

    y_true_gender = rng.integers(0, 2, size=n)
    flip = rng.random(n) < gender_error_rate
    predicted_gender = np.where(flip, 1 - y_true_gender, y_true_gender)
    # Build softmax-like probabilities consistent with predicted_gender and a
    # confidence roughly anti-correlated with whether it's actually wrong.
    confidence = np.where(flip, rng.uniform(0.5, 0.75, n), rng.uniform(0.8, 0.99, n))
    probs = np.zeros((n, 2))
    for i in range(n):
        probs[i, predicted_gender[i]] = confidence[i]
        probs[i, 1 - predicted_gender[i]] = 1 - confidence[i]

    return {
        "q10": q10, "q50": q50, "q90": q90, "probs": probs,
        "age": age, "age_mask": np.ones(n, dtype=bool),
        "gender": y_true_gender, "gender_mask": np.ones(n, dtype=bool),
        "latency_ms_per_image": latency,
        # Two models "evaluated on the same test set" share this ordered id
        # array -- required by build_*_analysis's paired-alignment check.
        "sample_id": np.arange(n),
    }


def test_build_clean_test_summary_has_expected_fields():
    preds = _synthetic_preds()
    summary = build_clean_test_summary(
        "custom_resnet18", preds, confidence_threshold=0.80,
        parameter_breakdown={"total_parameters": 11_500_000, "backbone_parameters": 11_000_000},
        mean_epoch_time_seconds=30.0,
    )
    for key in (
        "age_mae", "age_rmse", "age_error_median", "age_error_p90", "age_error_p95",
        "age_error_frac_>5", "age_error_frac_>10", "raw_interval_coverage", "raw_interval_width",
        "gender_selective_accuracy", "gender_coverage", "gender_abstention_rate",
        "gender_effective_accuracy", "total_parameters", "backbone_parameters", "latency_ms_per_image",
    ):
        assert key in summary, f"missing key: {key}"


def test_build_clean_test_summary_includes_calibrated_fields_when_calibration_present():
    preds = _synthetic_preds()
    summary = build_clean_test_summary("m", preds, confidence_threshold=0.80, calibration={"offset": 2.0})
    assert "calibrated_interval_coverage" in summary
    assert "calibrated_interval_width" in summary
    assert summary["calibrated_interval_width"] > summary["raw_interval_width"]


def test_build_clean_test_table_has_one_row_per_model():
    summaries = {
        "simple_cnn": build_clean_test_summary("simple_cnn", _synthetic_preds(seed=1), 0.80),
        "custom_resnet18": build_clean_test_summary("custom_resnet18", _synthetic_preds(seed=2), 0.80),
    }
    table = build_clean_test_table(summaries)
    assert len(table) == 2
    assert set(table["model"]) == {"simple_cnn", "custom_resnet18"}
    assert "age_mae" in table.columns


def test_gender_risk_coverage_analysis_structure():
    models_preds = {"a": _synthetic_preds(seed=1), "b": _synthetic_preds(seed=2)}
    result = build_gender_risk_coverage_analysis(models_preds, confidence_threshold=0.80, primary_model="a")
    assert set(result["curves"]) == {"a", "b"}
    assert set(result["aurc"]) == {"a", "b"}
    assert list(result["at_coverage"]["coverage"]) == [0.80, 0.90, 0.95, 0.98]
    assert "b" in result["pairwise_bootstrap"]
    assert "a" not in result["pairwise_bootstrap"]  # never compares the primary against itself
    for ci in result["pairwise_bootstrap"]["b"].values():
        assert "risk_diff_b_minus_a" in ci
        assert "excludes_zero" in ci


def test_age_selective_analysis_structure():
    models_preds = {"a": _synthetic_preds(seed=1), "b": _synthetic_preds(seed=2)}
    result = build_age_selective_analysis(models_preds, primary_model="a")
    assert set(result["mae_curves"]) == {"a", "b"}
    assert set(result["rmse_curves"]) == {"a", "b"}
    assert list(result["at_coverage"]["coverage"]) == [0.80, 0.90, 0.95, 0.98]
    assert "b" in result["pairwise_bootstrap"]


def test_gender_risk_coverage_analysis_includes_aurc_bootstrap_ci():
    models_preds = {"a": _synthetic_preds(seed=1), "b": _synthetic_preds(seed=2)}
    result = build_gender_risk_coverage_analysis(models_preds, confidence_threshold=0.80, primary_model="a")
    assert "b" in result["pairwise_bootstrap_aurc"]
    ci = result["pairwise_bootstrap_aurc"]["b"]
    for key in ("aurc_a", "aurc_b", "aurc_diff_b_minus_a", "ci_lower", "ci_upper", "excludes_zero"):
        assert key in ci


def test_age_selective_analysis_includes_aurc_bootstrap_ci():
    models_preds = {"a": _synthetic_preds(seed=1), "b": _synthetic_preds(seed=2)}
    result = build_age_selective_analysis(models_preds, primary_model="a")
    assert "b" in result["pairwise_bootstrap_aurc"]
    assert "aurc_diff_b_minus_a" in result["pairwise_bootstrap_aurc"]["b"]


def test_gender_risk_coverage_analysis_rejects_mismatched_sample_ids():
    """Regression test: equal sample count must not be treated as sufficient
    evidence of alignment -- two models with the same-length but differently
    ordered/identified samples must raise, not silently compute a bootstrap
    CI over mismatched pairs."""
    preds_a = _synthetic_preds(seed=1)
    preds_b = _synthetic_preds(seed=2)
    preds_b["sample_id"] = preds_b["sample_id"][::-1]  # same count, reordered identifiers
    with pytest.raises(ValueError):
        build_gender_risk_coverage_analysis({"a": preds_a, "b": preds_b}, confidence_threshold=0.80, primary_model="a")


def test_age_selective_analysis_rejects_missing_sample_ids():
    preds_a = _synthetic_preds(seed=1)
    preds_b = _synthetic_preds(seed=2)
    del preds_b["sample_id"]
    with pytest.raises(ValueError):
        build_age_selective_analysis({"a": preds_a, "b": preds_b}, primary_model="a")


def test_tail_error_analysis_structure_and_buckets():
    models_preds = {"a": _synthetic_preds(seed=1), "b": _synthetic_preds(seed=2)}
    result = build_tail_error_analysis(models_preds)
    assert set(result["errors_by_model"]) == {"a", "b"}
    assert set(result["tail_rates_by_model"]) == {"a", "b"}
    bucket_labels = list(result["bucket_table"]["age_bucket"])
    assert bucket_labels == ["0-12", "13-19", "20-34", "35-49", "50-64", "65+"]
    assert "a_mae" in result["bucket_table"].columns
    assert "b_count" in result["bucket_table"].columns


def test_final_interpretation_prefers_compact_model_when_no_measured_advantage():
    """The core fairness requirement: when ResNet and the alternative are
    statistically indistinguishable, the interpretation must say the
    compact/plain model is preferred -- never fabricate a ResNet win."""
    # Identical distributions for both models -> no real difference at all.
    resnet_preds = _synthetic_preds(seed=1, age_mae_scale=2.0, gender_error_rate=0.15)
    cnn_preds = _synthetic_preds(seed=1, age_mae_scale=2.0, gender_error_rate=0.15)  # same seed = same data
    models_preds = {"exp_0_simple_cnn": cnn_preds, "exp_d_resnet": resnet_preds}

    clean_table = build_clean_test_table({
        "exp_0_simple_cnn": build_clean_test_summary(
            "exp_0_simple_cnn", cnn_preds, 0.80,
            parameter_breakdown={"total_parameters": 4_000_000}, mean_epoch_time_seconds=20.0,
        ),
        "exp_d_resnet": build_clean_test_summary(
            "exp_d_resnet", resnet_preds, 0.80,
            parameter_breakdown={"total_parameters": 11_500_000}, mean_epoch_time_seconds=35.0,
        ),
    })
    gender_analysis = build_gender_risk_coverage_analysis(models_preds, 0.80, primary_model="exp_d_resnet")
    age_analysis = build_age_selective_analysis(models_preds, primary_model="exp_d_resnet")

    interpretation = build_final_interpretation(
        clean_table, gender_analysis, age_analysis, resnet_name="exp_d_resnet",
        comparison_names=["exp_0_simple_cnn"],
    )

    assert "preferred model" in interpretation
    assert "no measured evidence supports" in interpretation


def test_final_interpretation_reports_resnet_advantage_only_when_statistically_significant():
    """Construct a scenario where ResNet is genuinely, substantially better
    at gender selective prediction (near-zero error rate vs. a much
    higher one) -- only then should the interpretation credit ResNet."""
    resnet_preds = _synthetic_preds(seed=1, gender_error_rate=0.01, age_mae_scale=1.0)
    cnn_preds = _synthetic_preds(seed=2, gender_error_rate=0.35, age_mae_scale=1.0)
    models_preds = {"exp_0_simple_cnn": cnn_preds, "exp_d_resnet": resnet_preds}

    clean_table = build_clean_test_table({
        "exp_0_simple_cnn": build_clean_test_summary(
            "exp_0_simple_cnn", cnn_preds, 0.80,
            parameter_breakdown={"total_parameters": 4_000_000}, mean_epoch_time_seconds=20.0,
        ),
        "exp_d_resnet": build_clean_test_summary(
            "exp_d_resnet", resnet_preds, 0.80,
            parameter_breakdown={"total_parameters": 11_500_000}, mean_epoch_time_seconds=35.0,
        ),
    })
    gender_analysis = build_gender_risk_coverage_analysis(models_preds, 0.80, primary_model="exp_d_resnet")
    age_analysis = build_age_selective_analysis(models_preds, primary_model="exp_d_resnet")

    interpretation = build_final_interpretation(
        clean_table, gender_analysis, age_analysis, resnet_name="exp_d_resnet",
        comparison_names=["exp_0_simple_cnn"],
    )

    assert "statistically supported" in interpretation
    assert "deployment scenario" in interpretation
