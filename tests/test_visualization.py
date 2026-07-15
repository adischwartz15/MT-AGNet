"""Tests for the new uncertainty/comparison plotting helpers.

These only check that each function runs without error and produces a
real, non-empty image file -- exact pixel content isn't asserted (that's
what would make these tests brittle for no real benefit).
"""

from __future__ import annotations

import numpy as np

from src.utils.visualization import (
    plot_age_error_cdf, plot_coverage_width_tradeoff, plot_interval_width_by_bucket, plot_mean_std_bar,
    plot_parameter_latency_comparison, plot_pareto, plot_risk_coverage_curves, plot_tail_error_bars,
)


def test_plot_interval_width_by_bucket_creates_file(tmp_path):
    out = plot_interval_width_by_bucket(["0-10", "10-20"], np.array([5.0, 8.0]), tmp_path / "width.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_coverage_width_tradeoff_creates_file(tmp_path):
    out = plot_coverage_width_tradeoff(
        coverage_before=0.72, width_before=12.0, coverage_after=0.90, width_after=18.0,
        target_coverage=0.90, out_path=tmp_path / "tradeoff.png",
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_parameter_latency_comparison_creates_file(tmp_path):
    out = plot_parameter_latency_comparison(
        ["simple_cnn", "custom_resnet18"], [4_000_000, 11_500_000], [1.5, 1.8], tmp_path / "param_latency.png",
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_mean_std_bar_creates_file(tmp_path):
    out = plot_mean_std_bar(
        ["exp_c", "exp_d"], np.array([5.7, 5.5]), np.array([0.2, 0.15]), "Age MAE", tmp_path / "mean_std.png",
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_pareto_creates_file(tmp_path):
    out = plot_pareto(
        ["simple_cnn", "plain_deep18_no_skip", "custom_resnet18"], [4_000_000, 11_000_000, 11_200_000],
        [6.5, 5.9, 5.7], "Total parameters", "Age MAE", "Age MAE vs. parameter count", tmp_path / "pareto.png",
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_risk_coverage_curves_creates_file(tmp_path):
    curves = {
        "simple_cnn": (np.linspace(0.1, 1.0, 10), np.linspace(0.3, 0.1, 10)),
        "custom_resnet18": (np.linspace(0.1, 1.0, 10), np.linspace(0.2, 0.05, 10)),
    }
    out = plot_risk_coverage_curves(curves, "Selective risk (1 - accuracy)", "Gender risk-coverage", tmp_path / "risk_coverage.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_age_error_cdf_creates_file(tmp_path):
    rng = np.random.default_rng(0)
    errors_by_model = {"simple_cnn": rng.uniform(0, 20, 50), "custom_resnet18": rng.uniform(0, 15, 50)}
    out = plot_age_error_cdf(errors_by_model, tmp_path / "cdf.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_tail_error_bars_creates_file(tmp_path):
    tail_rates = {
        "simple_cnn": {">5": 0.4, ">10": 0.2, ">15": 0.1, ">20": 0.05},
        "custom_resnet18": {">5": 0.3, ">10": 0.15, ">15": 0.05, ">20": 0.02},
    }
    out = plot_tail_error_bars(tail_rates, tmp_path / "tail_bars.png")
    assert out.exists()
    assert out.stat().st_size > 0
