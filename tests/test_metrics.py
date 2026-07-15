"""Tests for the per-bucket uncertainty metrics and interval-example selection
used by the uncertainty evaluation report section."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    age_cumulative_score, age_error_percentiles, age_median_absolute_error, age_tail_error_rates,
    age_uncertainty_by_bucket, gender_balanced_accuracy, gender_coverage, gender_effective_accuracy,
    gender_precision_recall_f1, gender_roc_auc, select_interval_examples,
)


def test_age_uncertainty_by_bucket_computes_coverage_and_width_per_bucket():
    y_true = np.array([5.0, 5.0, 25.0, 25.0])
    q10 = np.array([0.0, 0.0, 20.0, 30.0])   # bucket 0-10: both covered; bucket 20-30: one covered, one not
    q50 = np.array([5.0, 5.0, 25.0, 25.0])
    q90 = np.array([10.0, 10.0, 30.0, 35.0])

    result = age_uncertainty_by_bucket(y_true, q10, q50, q90, bucket_edges=[0, 10, 20, 30, 200])
    assert result["0-10"]["count"] == 2
    assert result["0-10"]["coverage"] == 1.0
    assert result["20-30"]["count"] == 2
    assert result["20-30"]["coverage"] == 0.5  # second sample: q10=30 > y_true=25 -> not covered
    assert result["10-20"]["count"] == 0
    assert result["10-20"]["mae"] is None
    assert result["10-20"]["coverage"] is None


def test_age_uncertainty_by_bucket_mean_width_matches_manual_computation():
    y_true = np.array([5.0, 6.0])
    q10 = np.array([2.0, 3.0])
    q50 = np.array([5.0, 6.0])
    q90 = np.array([8.0, 10.0])
    result = age_uncertainty_by_bucket(y_true, q10, q50, q90, bucket_edges=[0, 10, 200])
    expected_mean_width = np.mean([8.0 - 2.0, 10.0 - 3.0])
    assert abs(result["0-10"]["mean_width"] - expected_mean_width) < 1e-9


def test_select_interval_examples_picks_narrowest_and_widest():
    image_paths = np.array([f"img_{i}.jpg" for i in range(5)])
    y_true = np.array([20.0, 25.0, 30.0, 35.0, 40.0])
    q10 = np.array([19.0, 20.0, 10.0, 34.0, 5.0])
    q50 = y_true.copy()
    q90 = np.array([21.0, 30.0, 50.0, 36.0, 75.0])
    # widths: [2, 10, 40, 2, 70] -- two ties at width=2 (img_0, img_3), widest is img_4 (70)

    result = select_interval_examples(image_paths, y_true, q10, q50, q90, k=2)
    narrow_paths = {r["image_path"] for r in result["narrowest"]}
    wide_paths = {r["image_path"] for r in result["widest"]}
    assert narrow_paths == {"img_0.jpg", "img_3.jpg"}
    assert "img_4.jpg" in wide_paths
    assert result["widest"][0]["image_path"] == "img_4.jpg"
    assert result["widest"][0]["width"] == 70.0


def test_select_interval_examples_handles_fewer_samples_than_k():
    image_paths = np.array(["a.jpg", "b.jpg"])
    y_true = np.array([10.0, 20.0])
    q10 = np.array([8.0, 15.0])
    q50 = y_true.copy()
    q90 = np.array([12.0, 25.0])
    result = select_interval_examples(image_paths, y_true, q10, q50, q90, k=5)
    assert len(result["narrowest"]) == 2
    assert len(result["widest"]) == 2


def test_select_interval_examples_record_fields():
    image_paths = np.array(["only.jpg"])
    y_true = np.array([42.0])
    q10 = np.array([40.0])
    q50 = np.array([42.0])
    q90 = np.array([45.0])
    result = select_interval_examples(image_paths, y_true, q10, q50, q90, k=1)
    record = result["narrowest"][0]
    assert record == {
        "image_path": "only.jpg", "true_age": 42.0, "q10": 40.0, "q50": 42.0, "q90": 45.0, "width": 5.0,
    }


def test_age_error_percentiles_matches_numpy_percentile():
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([12.0, 18.0, 35.0, 30.0, 90.0])  # errors: 2, 2, 5, 10, 40
    result = age_error_percentiles(y_true, y_pred)
    errors = np.array([2.0, 2.0, 5.0, 10.0, 40.0])
    assert abs(result["median"] - float(np.median(errors))) < 1e-9
    assert abs(result["p90"] - float(np.percentile(errors, 90))) < 1e-9
    assert abs(result["p95"] - float(np.percentile(errors, 95))) < 1e-9


def test_age_tail_error_rates_counts_fractions_above_each_threshold():
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([3.0, 7.0, 12.0, 22.0])  # errors: 3, 7, 12, 22
    result = age_tail_error_rates(y_true, y_pred, thresholds=(5, 10, 15, 20))
    assert result[">5"] == 0.75   # 7, 12, 22 exceed 5
    assert result[">10"] == 0.5   # 12, 22 exceed 10
    assert result[">15"] == 0.25  # only 22 exceeds 15
    assert result[">20"] == 0.25  # only 22 exceeds 20


def test_gender_coverage_is_one_minus_abstention_rate():
    abstain = np.array([True, False, False, False])
    assert gender_coverage(abstain) == 0.75


def test_gender_effective_accuracy_denominator_includes_abstentions():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])  # first two correct, last two wrong
    abstain = np.array([False, False, False, True])  # last sample abstained (also wrong if answered)
    # Effective accuracy = correct-and-accepted / all = 2 / 4 = 0.5
    assert gender_effective_accuracy(y_true, y_pred, abstain) == 0.5


def test_age_median_absolute_error_matches_manual_computation():
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([12.0, 18.0, 35.0, 30.0, 90.0])  # errors: 2, 2, 5, 10, 40
    assert abs(age_median_absolute_error(y_true, y_pred) - 5.0) < 1e-9


def test_age_median_absolute_error_matches_age_error_percentiles():
    """age_error_percentiles reuses age_median_absolute_error internally --
    they must always agree exactly, not just approximately."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([12.0, 18.0, 35.0, 30.0, 90.0])
    assert age_median_absolute_error(y_true, y_pred) == age_error_percentiles(y_true, y_pred)["median"]


def test_age_cumulative_score_at_5_years():
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([3.0, 5.0, 7.0, 20.0])  # errors: 3, 5, 7, 20
    # <=5 years: errors 3 and 5 qualify (2 of 4)
    assert age_cumulative_score(y_true, y_pred, threshold=5.0) == 0.5


def test_age_cumulative_score_is_1_when_all_errors_within_threshold():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([11.0, 19.0])
    assert age_cumulative_score(y_true, y_pred, threshold=5.0) == 1.0


def test_gender_balanced_accuracy_handles_class_imbalance():
    # 8 of class 0 (all correct), 2 of class 1 (both wrong) -> raw accuracy
    # 0.8, but balanced accuracy averages per-class recall: (1.0 + 0.0) / 2.
    y_true = np.array([0] * 8 + [1] * 2)
    y_pred = np.array([0] * 8 + [0] * 2)
    assert abs(gender_balanced_accuracy(y_true, y_pred) - 0.5) < 1e-9


def test_gender_precision_recall_f1_binary_positive_class_1():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])  # class 1: TP=2, FP=1, FN=1
    result = gender_precision_recall_f1(y_true, y_pred)
    assert abs(result["precision"] - 2 / 3) < 1e-9
    assert abs(result["recall"] - 2 / 3) < 1e-9
    assert abs(result["f1"] - 2 / 3) < 1e-9


def test_gender_precision_recall_f1_zero_division_returns_zero_not_a_crash():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])  # no positive predictions or labels at all
    result = gender_precision_recall_f1(y_true, y_pred)
    assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_gender_roc_auc_returns_none_for_single_class_without_raising():
    """Undefined on a tiny/degenerate smoke split (only one class present) --
    must return None explicitly, never raise or silently fabricate a value."""
    y_true = np.array([1, 1, 1, 1])
    probs = np.array([0.9, 0.6, 0.7, 0.55])
    assert gender_roc_auc(y_true, probs) is None


def test_gender_roc_auc_perfect_separation_is_1():
    y_true = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    assert gender_roc_auc(y_true, probs) == 1.0


def test_gender_effective_accuracy_lower_than_selective_accuracy_when_abstaining_on_hard_cases():
    from src.evaluation.metrics import gender_accuracy

    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 1])  # only first two are actually correct
    abstain = np.array([False, False, True, True, True])  # abstains on all three wrong cases
    selective_acc = gender_accuracy(y_true, y_pred, abstain)  # accuracy among accepted only: 2/2 = 1.0
    effective_acc = gender_effective_accuracy(y_true, y_pred, abstain)  # 2/5 = 0.4
    assert selective_acc == 1.0
    assert effective_acc == 0.4
    assert effective_acc < selective_acc
