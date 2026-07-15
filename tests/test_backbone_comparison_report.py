"""Tests for the "Backbone Comparison" report section.

Covers the pure table-building function, the corrected comparison
framing (SimpleCNN vs ResNet is an efficiency/accuracy trade-off, not a
residual-connection ablation; PlainDeep18NoSkip vs ResNet is the actual
residual-connection ablation), the optional zero-init-residual ablation
(exp_0c), and the full report assembly's honest "not yet run" fallback
when any of the three required experiments' artifacts don't exist on disk.
"""

from __future__ import annotations

from src.evaluation.comparison import build_backbone_comparison_table, build_backbone_comparison_table_multi
from src.evaluation.reports import (
    _CNN_EXPERIMENT, _PLAIN_DEEP18_EXPERIMENT, _RESNET_EXPERIMENT, _RESNET_NO_ZERO_INIT_EXPERIMENT,
    _backbone_comparison_interpretation, build_backbone_comparison_section, generate_markdown_report,
)
from src.utils.io import save_json


def _cnn_metrics():
    return {
        "backbone_name": "simple_cnn", "total_parameters": 4_000_000, "backbone_parameters": 3_700_000,
        "mean_epoch_time_seconds": 30.0, "latency_ms_per_image": 1.5,
        "age_mae": 6.50, "age_rmse": 9.0, "gender_accuracy": 0.94, "abstention_rate": 0.25,
        "interval_coverage": 0.75, "mean_interval_width": 20.0,
    }


def _plain_deep18_metrics():
    return {
        "backbone_name": "plain_deep18_no_skip", "total_parameters": 11_500_000, "backbone_parameters": 11_100_000,
        "mean_epoch_time_seconds": 40.0, "latency_ms_per_image": 1.75,
        "age_mae": 6.10, "age_rmse": 8.7, "gender_accuracy": 0.95, "abstention_rate": 0.22,
        "interval_coverage": 0.77, "mean_interval_width": 18.5,
    }


def _resnet_metrics():
    return {
        "backbone_name": "custom_resnet18", "total_parameters": 11_571_909, "backbone_parameters": 11_176_512,
        "mean_epoch_time_seconds": 42.0, "latency_ms_per_image": 1.8,
        "age_mae": 5.71, "age_rmse": 8.32, "gender_accuracy": 0.97, "abstention_rate": 0.19,
        "interval_coverage": 0.79, "mean_interval_width": 16.79,
    }


def test_build_backbone_comparison_table_has_expected_rows_and_values():
    table = build_backbone_comparison_table(_cnn_metrics(), _resnet_metrics())
    row_by_metric = {row["metric"]: row for _, row in table.iterrows()}
    assert row_by_metric["Backbone"]["simple_cnn"] == "simple_cnn"
    assert row_by_metric["Backbone"]["custom_resnet18"] == "custom_resnet18"
    assert row_by_metric["Age MAE"]["simple_cnn"] == 6.50
    assert row_by_metric["Age MAE"]["custom_resnet18"] == 5.71
    # Calibrated coverage absent from both fixtures -> None, not fabricated.
    assert row_by_metric["Calibrated interval coverage"]["simple_cnn"] is None


def test_build_backbone_comparison_table_multi_has_one_column_per_model():
    table = build_backbone_comparison_table_multi({
        "simple_cnn": _cnn_metrics(), "plain_deep18_no_skip": _plain_deep18_metrics(), "custom_resnet18": _resnet_metrics(),
    })
    row_by_metric = {row["metric"]: row for _, row in table.iterrows()}
    assert row_by_metric["Age MAE"]["simple_cnn"] == 6.50
    assert row_by_metric["Age MAE"]["plain_deep18_no_skip"] == 6.10
    assert row_by_metric["Age MAE"]["custom_resnet18"] == 5.71


def test_backbone_comparison_interpretation_credits_lower_mae_side():
    sentence = _backbone_comparison_interpretation(_cnn_metrics(), _resnet_metrics())
    assert "ResNet" in sentence
    assert "0.79" in sentence or "0.8" in sentence  # mae_diff = 6.50 - 5.71 = 0.79
    assert "does not, by itself, establish" in sentence  # no unwarranted causal claim


def test_backbone_comparison_interpretation_handles_reversed_direction():
    cnn = _cnn_metrics()
    resnet = _resnet_metrics()
    cnn["age_mae"] = 4.0  # now the plain CNN has the lower MAE
    sentence = _backbone_comparison_interpretation(cnn, resnet)
    assert "plain CNN" in sentence.split(",")[0]  # credited in the leading clause


def test_backbone_comparison_interpretation_missing_data_is_honest():
    sentence = _backbone_comparison_interpretation({}, {})
    assert "Not enough metrics available" in sentence


def test_backbone_comparison_section_reports_unavailable_when_no_artifacts(tmp_path):
    section = build_backbone_comparison_section(tmp_path)
    assert "## Backbone Comparison" in section
    assert "Results unavailable" in section
    assert _CNN_EXPERIMENT in section
    assert _PLAIN_DEEP18_EXPERIMENT in section
    assert _RESNET_EXPERIMENT in section


def test_backbone_comparison_section_reports_unavailable_when_only_two_of_three_present(tmp_path):
    """Regression test: the section requires all three models (SimpleCNN,
    PlainDeep18NoSkip, Custom ResNet-18), not just the historical two --
    PlainDeep18NoSkip missing must not silently render a table anyway."""
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True)
    save_json(_cnn_metrics(), metrics_dir / f"{_CNN_EXPERIMENT}_parameter_breakdown.json")
    save_json(_resnet_metrics(), metrics_dir / f"{_RESNET_EXPERIMENT}_parameter_breakdown.json")

    section = build_backbone_comparison_section(tmp_path)
    assert "Results unavailable" in section
    assert _PLAIN_DEEP18_EXPERIMENT in section


def test_backbone_comparison_section_renders_all_three_models_and_correct_framing(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True)
    save_json(_cnn_metrics(), metrics_dir / f"{_CNN_EXPERIMENT}_parameter_breakdown.json")
    save_json(_plain_deep18_metrics(), metrics_dir / f"{_PLAIN_DEEP18_EXPERIMENT}_parameter_breakdown.json")
    save_json(_resnet_metrics(), metrics_dir / f"{_RESNET_EXPERIMENT}_parameter_breakdown.json")

    section = build_backbone_comparison_section(tmp_path)
    assert "Results unavailable" not in section
    assert "simple_cnn" in section
    assert "plain_deep18_no_skip" in section
    assert "custom_resnet18" in section
    # The corrected framing: SimpleCNN-vs-ResNet must not be called a
    # residual-connection ablation, and PlainDeep18NoSkip-vs-ResNet must be.
    assert "efficiency/accuracy trade-off" in section
    assert "*not* a residual-connection ablation" in section
    assert "the residual-connection ablation" in section
    # exp_0c not present on disk -- honest "not yet generated" message.
    assert _RESNET_NO_ZERO_INIT_EXPERIMENT in section


def test_backbone_comparison_section_includes_zero_init_ablation_when_present(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True)
    save_json(_cnn_metrics(), metrics_dir / f"{_CNN_EXPERIMENT}_parameter_breakdown.json")
    save_json(_plain_deep18_metrics(), metrics_dir / f"{_PLAIN_DEEP18_EXPERIMENT}_parameter_breakdown.json")
    save_json(_resnet_metrics(), metrics_dir / f"{_RESNET_EXPERIMENT}_parameter_breakdown.json")
    no_zero_init = dict(_resnet_metrics())
    no_zero_init["backbone_name"] = "custom_resnet18_no_zero_init"
    no_zero_init["age_mae"] = 5.9
    save_json(no_zero_init, metrics_dir / f"{_RESNET_NO_ZERO_INIT_EXPERIMENT}_parameter_breakdown.json")

    section = build_backbone_comparison_section(tmp_path)
    assert "zero-init ablation" in section
    assert "_Not yet generated" not in section.split("zero-init ablation")[1][:500]


def test_generate_markdown_report_includes_backbone_comparison_heading(tmp_path):
    report = generate_markdown_report(tmp_path)
    assert "## Backbone Comparison" in report
