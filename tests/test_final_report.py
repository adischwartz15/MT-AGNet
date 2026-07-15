"""Tests for the final cross-cutting results report generator.

Two things matter most here: (1) with no artifacts on disk, every section
renders an honest "not yet generated" message instead of fabricating
numbers; (2) once real (synthetic-but-saved) artifacts exist, the report
picks them up correctly -- ablation table, seed mean+/-std, per-bucket
uncertainty metrics, robustness summary, and parameter/latency comparison.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.final_report import generate_final_results_report, save_final_results_report


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_report_is_honest_when_no_artifacts_exist(tmp_path):
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()
    repo_root = tmp_path

    report = generate_final_results_report(outputs_dir, repo_root)

    assert "Final Results Report" in report
    assert "Not yet generated" in report
    assert "marginal coverage" in report.lower() or "conditional coverage" in report.lower()
    assert "No findings are stated yet" in report


def test_report_picks_up_seed_runs_and_bucket_metrics(tmp_path):
    outputs_dir = tmp_path / "outputs"
    metrics_dir = outputs_dir / "metrics"
    repo_root = tmp_path

    for seed in (42, 43):
        _write_json(
            metrics_dir / f"exp_c_shared_adapters_seed{seed}_test_metrics.json",
            {"age_mae": 5.0 + seed * 0.01, "gender_accuracy": 0.9},
        )

    _write_json(
        metrics_dir / "exp_d_shared_adapters_learned_balance_test_metrics.json",
        {
            "age_mae": 5.2,
            "interval_coverage": 0.79,
            "mean_interval_width": 12.0,
            "age_metrics_by_bucket": {
                "0-10": {"count": 5, "mae": 2.1, "coverage": 0.8, "mean_width": 8.0, "median_width": 7.5},
                "10-20": {"count": 0, "mae": None, "coverage": None, "mean_width": None, "median_width": None},
            },
            "interval_examples": {
                "narrowest": [{"image_path": "a.jpg", "true_age": 20.0, "q10": 15.0, "q50": 20.0, "q90": 25.0, "width": 10.0}],
                "widest": [{"image_path": "b.jpg", "true_age": 40.0, "q10": 20.0, "q50": 40.0, "q90": 60.0, "width": 40.0}],
            },
        },
    )

    report = generate_final_results_report(outputs_dir, repo_root)

    assert "exp_c_shared_adapters" in report
    assert "exp_d_shared_adapters_learned_balance" in report
    assert "0-10" in report
    assert "a.jpg" in report and "b.jpg" in report
    # exp_c has 2 real seed runs on disk -- a real mean +/- std should be rendered.
    assert "+/-" in report


def test_report_does_not_crash_when_plots_live_outside_report_dir(tmp_path):
    """Regression test: outputs_dir/plots must not need to be a subpath of
    report_dir (e.g. a notebook's RUN_DIR is entirely outside the repo
    checkout) -- Path.relative_to would raise ValueError here; the fix
    uses os.path.relpath, which handles unrelated absolute paths."""
    # outputs_dir and report_dir are siblings, neither nested in the other --
    # exactly the RUN_DIR-vs-REPO_DIR relationship in the real notebooks.
    outputs_dir = tmp_path / "run_dir"
    report_dir = tmp_path / "somewhere_else" / "reports"
    report_dir.mkdir(parents=True)

    metrics_dir = outputs_dir / "metrics"
    plots_dir = outputs_dir / "plots"
    _write_json(
        metrics_dir / "exp_d_shared_adapters_learned_balance_test_metrics.json",
        {
            "age_mae": 5.2,
            "age_metrics_by_bucket": {"0-10": {"count": 5, "mae": 2.1, "coverage": 0.8, "mean_width": 8.0, "median_width": 7.5}},
        },
    )
    plots_dir.mkdir(parents=True, exist_ok=True)
    (plots_dir / "exp_d_shared_adapters_learned_balance_test_metrics_interval_coverage.png").write_bytes(b"fake-png-bytes")

    report = generate_final_results_report(outputs_dir, report_dir)

    assert "interval_coverage.png" in report
    assert "![" in report


def test_save_final_results_report_writes_working_relative_links(tmp_path):
    outputs_dir = tmp_path / "outputs"
    docs_dir = tmp_path / "docs"
    metrics_dir = outputs_dir / "metrics"
    plots_dir = outputs_dir / "plots"
    _write_json(
        metrics_dir / "exp_d_shared_adapters_learned_balance_test_metrics.json",
        {"age_mae": 5.2, "age_metrics_by_bucket": {}},
    )
    plots_dir.mkdir(parents=True, exist_ok=True)
    (plots_dir / "exp_d_shared_adapters_learned_balance_test_metrics_interval_coverage.png").write_bytes(b"fake-png-bytes")

    out_path = save_final_results_report(outputs_dir, docs_dir)
    assert out_path.exists()

    report_text = out_path.read_text(encoding="utf-8")
    for line in report_text.splitlines():
        if line.startswith("!["):
            rel_path = line.split("(", 1)[1].rstrip(")")
            resolved = (out_path.parent / rel_path).resolve()
            assert resolved.exists(), f"broken image link: {rel_path} from {out_path.parent}"
