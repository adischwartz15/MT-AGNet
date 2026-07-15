"""Tests for deterministic robustness corruption functions.

Covers the full required set: blur, brightness, contrast, Gaussian
noise, JPEG compression, partial occlusion, resize degradation
(low_resolution), and grayscale conversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.evaluation.robustness import (
    CORRUPTION_NAMES, apply_corruption, build_robustness_diff_table, compute_degradation, corruption_summary,
    gaussian_blur, gaussian_noise, grayscale, high_brightness, high_contrast, iter_corruption_configs,
    jpeg_compression, low_brightness, low_contrast, low_resolution, partial_crop, partial_occlusion,
    stratified_sample,
)


def _sample_image(size=(64, 64)) -> Image.Image:
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(array)


def test_all_required_corruption_types_are_registered():
    required = {
        "gaussian_blur", "gaussian_noise", "low_resolution", "jpeg_compression",
        "low_brightness", "high_brightness", "low_contrast", "high_contrast",
        "grayscale", "partial_occlusion",
    }
    assert required <= set(CORRUPTION_NAMES)


def test_each_corruption_preserves_image_size():
    image = _sample_image((80, 60))
    corruptions = [
        (gaussian_blur, 1.5), (gaussian_noise, 0.1), (low_resolution, 0.3), (jpeg_compression, 20),
        (low_brightness, 0.5), (high_brightness, 1.6), (low_contrast, 0.5), (high_contrast, 1.8),
        (grayscale, 0.7), (partial_occlusion, 0.2), (partial_crop, 0.2),
    ]
    for fn, param in corruptions:
        result = fn(image, param, seed=1)
        assert result.size == (80, 60), f"{fn.__name__} changed image size"


def test_grayscale_blend_factor_one_removes_all_color_variation():
    image = _sample_image()
    result = grayscale(image, blend_factor=1.0)
    array = np.asarray(result)
    # Fully desaturated: R, G, B channels should be identical per pixel.
    assert np.allclose(array[..., 0], array[..., 1])
    assert np.allclose(array[..., 1], array[..., 2])


def test_grayscale_blend_factor_zero_is_original_image():
    image = _sample_image()
    result = grayscale(image, blend_factor=0.0)
    assert np.array_equal(np.asarray(result), np.asarray(image.convert("RGB")))


def test_grayscale_clamps_out_of_range_blend_factor():
    image = _sample_image()
    over = grayscale(image, blend_factor=1.5)
    under = grayscale(image, blend_factor=-0.5)
    fully_gray = grayscale(image, blend_factor=1.0)
    original = np.asarray(image.convert("RGB"))
    assert np.array_equal(np.asarray(over), np.asarray(fully_gray))
    assert np.array_equal(np.asarray(under), original)


def test_low_contrast_and_high_contrast_move_in_opposite_directions():
    image = _sample_image()
    baseline_std = np.asarray(image.convert("L"), dtype=np.float64).std()
    low = np.asarray(low_contrast(image, 0.3).convert("L"), dtype=np.float64).std()
    high = np.asarray(high_contrast(image, 2.0).convert("L"), dtype=np.float64).std()
    assert low < baseline_std
    assert high > baseline_std


def test_apply_corruption_dispatches_new_corruption_types():
    image = _sample_image()
    for name, param in (("low_contrast", 0.5), ("high_contrast", 1.5), ("grayscale", 0.5)):
        result = apply_corruption(image, name, param, seed=0)
        assert result.size == image.size


def test_apply_corruption_rejects_unknown_name():
    import pytest

    with pytest.raises(ValueError):
        apply_corruption(_sample_image(), "not_a_real_corruption", 1.0)


def test_iter_corruption_configs_yields_new_corruption_types():
    robustness_cfg = {
        "corruptions": {
            "low_contrast": {"severities": [1, 2], "params": [0.7, 0.5]},
            "grayscale": {"severities": [1], "params": [0.4]},
        }
    }
    configs = list(iter_corruption_configs(robustness_cfg))
    names = {name for name, _, _ in configs}
    assert names == {"low_contrast", "grayscale"}
    assert len(configs) == 3


def test_corruption_summary_is_computed_not_hardcoded():
    robustness_cfg = {
        "corruptions": {
            "low_contrast": {"severities": [1, 2], "params": [0.7, 0.5]},
            "grayscale": {"severities": [1], "params": [0.4]},
        }
    }
    summary = corruption_summary(robustness_cfg)
    assert summary["n_corruption_types"] == 2
    assert summary["corruption_type_names"] == ["grayscale", "low_contrast"]
    assert summary["n_total_conditions"] == 3  # 2 severities + 1 severity
    assert summary["severities_per_type"] == {"low_contrast": 2, "grayscale": 1}


def test_corruption_summary_matches_real_config():
    """The doc claim of '11 types x 3 severities' must match the ACTUAL
    configs/robustness.yaml -- this test fails loudly the moment they
    diverge, instead of letting docs/robustness.md silently go stale."""
    from src.utils.config import CONFIG_DIR, load_config

    robustness_cfg = load_config(CONFIG_DIR / "robustness.yaml")["robustness"]
    summary = corruption_summary(robustness_cfg)
    assert summary["n_corruption_types"] == len(CORRUPTION_NAMES)
    assert summary["n_total_conditions"] == sum(summary["severities_per_type"].values())


@pytest.mark.parametrize(
    "name,param",
    [
        ("gaussian_blur", 1.5), ("gaussian_noise", 0.1), ("low_resolution", 0.3), ("jpeg_compression", 20),
        ("low_brightness", 0.5), ("high_brightness", 1.6), ("low_contrast", 0.5), ("high_contrast", 1.8),
        ("grayscale", 0.7), ("partial_occlusion", 0.2), ("partial_crop", 0.2),
    ],
)
def test_corruption_is_deterministic_for_a_fixed_seed(name, param):
    """Every corruption must produce byte-identical output given the same
    seed -- required for a fair, reproducible robustness comparison across
    models (the same corrupted image must be shown to every model)."""
    image = _sample_image((48, 48))
    result_1 = apply_corruption(image, name, param, seed=7)
    result_2 = apply_corruption(image, name, param, seed=7)
    assert np.array_equal(np.asarray(result_1), np.asarray(result_2))


def test_corruption_with_randomness_differs_across_seeds():
    """Sanity check for the determinism test above: corruptions that use
    randomness (noise/occlusion/crop) must actually depend on the seed,
    otherwise the "same seed -> same output" test would be vacuous."""
    image = _sample_image((48, 48))
    result_a = apply_corruption(image, "gaussian_noise", 0.1, seed=1)
    result_b = apply_corruption(image, "gaussian_noise", 0.1, seed=2)
    assert not np.array_equal(np.asarray(result_a), np.asarray(result_b))


def _robustness_results_df():
    return pd.DataFrame([
        {"corruption": "clean", "severity": 0, "param": None, "age_mae": 5.0, "gender_accuracy": 0.95, "abstention_rate": 0.05},
        {"corruption": "gaussian_blur", "severity": 1, "param": 0.8, "age_mae": 6.0, "gender_accuracy": 0.90, "abstention_rate": 0.10},
        {"corruption": "gaussian_blur", "severity": 2, "param": 1.6, "age_mae": 8.0, "gender_accuracy": 0.80, "abstention_rate": 0.20},
    ])


def test_compute_degradation_adds_delta_and_pct_change_columns():
    df = compute_degradation(_robustness_results_df())
    clean_row = df[df["corruption"] == "clean"].iloc[0]
    assert clean_row["age_mae_delta"] == 0.0
    assert clean_row["age_mae_pct_change"] == 0.0

    blur_severity_2 = df[(df["corruption"] == "gaussian_blur") & (df["severity"] == 2)].iloc[0]
    assert blur_severity_2["age_mae_delta"] == pytest.approx(3.0)  # 8.0 - 5.0
    assert blur_severity_2["age_mae_pct_change"] == pytest.approx(60.0)  # 3.0 / 5.0 * 100
    assert blur_severity_2["gender_accuracy_delta"] == pytest.approx(-0.15)


def test_compute_degradation_raises_without_clean_baseline():
    df = _robustness_results_df()
    df = df[df["corruption"] != "clean"]
    with pytest.raises(ValueError):
        compute_degradation(df)


def test_build_robustness_diff_table_computes_direct_model_vs_model_difference():
    df_cnn = _robustness_results_df()
    df_resnet = _robustness_results_df().copy()
    df_resnet["age_mae"] = df_resnet["age_mae"] - 1.0  # ResNet uniformly 1 year better

    diff_table = build_robustness_diff_table({"simple_cnn": df_cnn, "custom_resnet18": df_resnet})
    assert len(diff_table) == 3
    row = diff_table[(diff_table["corruption"] == "gaussian_blur") & (diff_table["severity"] == 2)].iloc[0]
    assert row["simple_cnn_age_mae"] == pytest.approx(8.0)
    assert row["custom_resnet18_age_mae"] == pytest.approx(7.0)
    assert row["diff_age_mae_(custom_resnet18_minus_simple_cnn)"] == pytest.approx(-1.0)


def test_build_robustness_diff_table_requires_at_least_two_models():
    with pytest.raises(ValueError):
        build_robustness_diff_table({"only_one": _robustness_results_df()})


def test_build_robustness_diff_table_produces_all_pairwise_comparisons_for_three_models():
    """Regression test: with three models, every pairwise comparison must
    be present -- not just the first two by dict insertion order -- and in
    particular SimpleCNN vs ResNet, PlainDeep18NoSkip vs ResNet, and
    SimpleCNN vs PlainDeep18NoSkip must all appear."""
    df_cnn = _robustness_results_df()
    df_plain = _robustness_results_df().copy()
    df_plain["age_mae"] = df_plain["age_mae"] - 0.5
    df_resnet = _robustness_results_df().copy()
    df_resnet["age_mae"] = df_resnet["age_mae"] - 1.0

    diff_table = build_robustness_diff_table({
        "simple_cnn": df_cnn, "plain_deep18_no_skip": df_plain, "custom_resnet18": df_resnet,
    })

    assert set(diff_table["comparison"].unique()) == {
        "plain_deep18_no_skip_vs_simple_cnn", "custom_resnet18_vs_simple_cnn", "custom_resnet18_vs_plain_deep18_no_skip",
    }
    # 3 pairs x 3 (corruption, severity) rows each.
    assert len(diff_table) == 9

    resnet_vs_cnn = diff_table[diff_table["comparison"] == "custom_resnet18_vs_simple_cnn"]
    row = resnet_vs_cnn[(resnet_vs_cnn["corruption"] == "gaussian_blur") & (resnet_vs_cnn["severity"] == 2)].iloc[0]
    assert row["diff_age_mae_(custom_resnet18_minus_simple_cnn)"] == pytest.approx(-1.0)


def _synthetic_split_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "image_path": [f"img_{i}.jpg" for i in range(n)],
        "age": rng.uniform(0, 90, n),
        "gender_label": rng.integers(0, 2, n),
    })


def test_stratified_sample_returns_full_df_when_max_samples_covers_it():
    df = _synthetic_split_df(50)
    sampled = stratified_sample(df, max_samples=None)
    assert len(sampled) == 50
    sampled = stratified_sample(df, max_samples=1000)
    assert len(sampled) == 50


def test_stratified_sample_respects_approximate_max_samples_and_is_deterministic():
    df = _synthetic_split_df(400)
    sampled_1 = stratified_sample(df, max_samples=100, seed=42)
    sampled_2 = stratified_sample(df, max_samples=100, seed=42)
    # Rounding per stratum means this is approximate, not exact.
    assert 80 <= len(sampled_1) <= 120
    assert list(sampled_1["image_path"]) == list(sampled_2["image_path"])


def test_stratified_sample_covers_every_gender_label_present():
    """A naive head(max_samples) could silently drop an entire subgroup if
    the split CSV happens to be sorted/grouped -- stratified sampling must
    not do that as long as the subgroup has enough rows to be represented."""
    df = _synthetic_split_df(400)
    sampled = stratified_sample(df, max_samples=200, seed=1)
    assert set(sampled["gender_label"].unique()) == set(df["gender_label"].unique())


def test_stratified_sample_different_seeds_pick_different_rows():
    df = _synthetic_split_df(400)
    sampled_a = stratified_sample(df, max_samples=100, seed=1)
    sampled_b = stratified_sample(df, max_samples=100, seed=2)
    assert list(sampled_a["image_path"]) != list(sampled_b["image_path"])


class _FakeQuantileModel:
    """Minimal stand-in for MultiTaskFaceModel: fixed q10/q50/q90 and gender
    logits regardless of input, so evaluate_condition's calibration-offset
    wiring can be tested without a real trained checkpoint."""

    def eval(self):
        return self

    def __call__(self, images):
        n = images.shape[0]
        return {
            "age_output": {
                "q10": torch.full((n,), 20.0), "q50": torch.full((n,), 25.0), "q90": torch.full((n,), 30.0),
            },
            "gender_logits": torch.zeros((n, 2)),
        }


def test_evaluate_condition_reports_calibrated_coverage_and_width_alongside_raw(tmp_path):
    from src.data.transforms import EvalTransform
    from src.evaluation.robustness import evaluate_condition

    rows = []
    for i in range(6):
        path = tmp_path / f"img_{i}.png"
        Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(path)
        rows.append({"image_path": str(path), "age": 25.0, "gender_label": 0})
    df = pd.DataFrame(rows)

    metrics = evaluate_condition(
        _FakeQuantileModel(), df, EvalTransform(32), device="cpu", gender_confidence_threshold=0.80,
        corruption_name=None, severity=0, param=None, seed=0, calibration_offset=5.0,
    )

    assert metrics["mean_interval_width"] == pytest.approx(10.0)  # raw: 30 - 20
    assert metrics["mean_interval_width_calibrated"] == pytest.approx(20.0)  # (30+5) - (20-5)
    assert metrics["interval_coverage"] == pytest.approx(1.0)  # age=25 is inside [20, 30]
    assert metrics["interval_coverage_calibrated"] == pytest.approx(1.0)


def test_evaluate_condition_omits_calibrated_keys_without_an_offset(tmp_path):
    from src.data.transforms import EvalTransform
    from src.evaluation.robustness import evaluate_condition

    path = tmp_path / "img_0.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(path)
    df = pd.DataFrame([{"image_path": str(path), "age": 25.0, "gender_label": 0}])

    metrics = evaluate_condition(
        _FakeQuantileModel(), df, EvalTransform(32), device="cpu", gender_confidence_threshold=0.80,
        corruption_name=None, severity=0, param=None, seed=0,
    )
    assert "interval_coverage_calibrated" not in metrics
    assert "mean_interval_width_calibrated" not in metrics
