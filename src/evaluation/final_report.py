"""Builds the final, cross-cutting results report for the whole project.

Unlike ``src/evaluation/reports.py`` (which focuses on the architecture
ablation study), this module assembles *all* of the course-facing results
in one document: the ablation table, the plain-CNN-vs-ResNet comparison,
a mean +/- std table across seeds, per-age-bucket uncertainty metrics
(raw and calibrated), robustness degradation, and parameter/latency
comparison plots. Every section reads only real artifacts already saved
under ``outputs/`` by other scripts (``run_experiments.py``,
``run_seeds.py``, ``evaluate.py``, ``run_robustness.py``) -- any section
whose backing artifact is missing renders an explicit "not yet generated"
message with the command that would produce it, never a fabricated number.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.comparison import (
    aggregate_seed_metrics, build_architecture_ablation_table, build_seed_aggregate_table,
)
from src.evaluation.reports import (
    _CNN_EXPERIMENT, _RESNET_EXPERIMENT, _MISSING, _backbone_comparison_interpretation, _df_to_md_table,
    _load_merged_experiment_metrics, _read_csv, _read_json, build_backbone_comparison_section,
    discover_experiment_results,
)
from src.utils.visualization import plot_mean_std_bar, plot_parameter_latency_comparison

_SEED_GROUP_RE = re.compile(r"^(?P<experiment>.+)_seed\d+_test_metrics\.json$")

# Preferred order for picking the "primary" model shown in the uncertainty
# section: the main research backbone (learned-balance shared adapters)
# first, falling back to earlier ablation stages, and finally to whatever
# checkpoint was evaluated under the generic "multitask" name.
_PRIMARY_EXPERIMENT_CANDIDATES = (
    "exp_d_shared_adapters_learned_balance", "exp_c_shared_adapters", "multitask",
)


def _md_image(path: Path, report_dir: Path, label: str) -> str:
    """Markdown image link, relative to ``report_dir`` (where the .md file itself lives).

    Uses ``os.path.relpath`` rather than ``Path.relative_to`` deliberately:
    ``relative_to`` raises ``ValueError`` unless ``path`` is actually a
    subpath of ``report_dir``'s ancestor, which fails whenever the report
    and its plots live in unrelated directory trees (e.g. a notebook's
    ``RUN_DIR`` is entirely outside the repository checkout).
    ``os.path.relpath`` computes a correct ``../``-chain between any two
    absolute paths (same drive on Windows) regardless of ancestry.
    """
    rel = os.path.relpath(path, start=report_dir)
    return f"![{label}]({Path(rel).as_posix()})\n"


def _discover_seed_metrics(outputs_dir: Path) -> dict[str, list[dict]]:
    """Group every ``{experiment}_seed{N}_test_metrics.json`` found by experiment name.

    Searches both the flat legacy ``outputs/metrics`` directory and every
    isolated ``experiments/<experiment>/seed_<seed>/metrics`` directory
    produced by ``scripts/run_seeds.py`` (see
    ``src/evaluation/reports.py:_experiment_metrics_dirs`` and
    ``src/utils/experiment_paths.py``) -- a seed's metrics file keeps this
    same filename regardless of which directory it physically lives under.
    """
    from src.evaluation.reports import _experiment_metrics_dirs

    groups: dict[str, list[dict]] = {}
    seen_files: set[str] = set()
    for metrics_dir in _experiment_metrics_dirs(outputs_dir):
        if not metrics_dir.exists():
            continue
        for f in sorted(metrics_dir.glob("*_seed*_test_metrics.json")):
            match = _SEED_GROUP_RE.match(f.name)
            if not match or f.name in seen_files:
                continue
            data = _read_json(f)
            if data is not None:
                groups.setdefault(match.group("experiment"), []).append(data)
                seen_files.add(f.name)
    return groups


def _find_primary_test_metrics(outputs_dir: Path) -> tuple[str | None, dict | None]:
    from src.evaluation.reports import _experiment_metrics_dirs, _read_json_from_any

    dirs = _experiment_metrics_dirs(outputs_dir)
    for name in _PRIMARY_EXPERIMENT_CANDIDATES:
        data = _read_json_from_any(dirs, f"{name}_test_metrics.json")
        if data is not None:
            return name, data
    return None, None


def _build_ablation_section(outputs_dir: Path) -> str:
    lines = ["## Architecture Ablation Table\n"]
    experiment_results = discover_experiment_results(outputs_dir / "metrics")
    if not experiment_results:
        lines.append(_MISSING.format(cmd="python scripts/run_experiments.py") + "\n")
        return "\n".join(lines)
    table = build_architecture_ablation_table(experiment_results)
    lines.append(_df_to_md_table(table) + "\n")
    return "\n".join(lines)


def _build_seed_aggregate_section(outputs_dir: Path) -> str:
    lines = ["## Mean +/- Std Across Seeds\n"]
    groups = _discover_seed_metrics(outputs_dir)
    if not groups:
        lines.append(
            _MISSING.format(cmd="python scripts/run_seeds.py --experiment <name> --seeds 42,123,2026") + "\n"
        )
        return "\n".join(lines)

    aggregates = {name: aggregate_seed_metrics(seed_metrics) for name, seed_metrics in groups.items()}
    table = build_seed_aggregate_table(aggregates)
    lines.append(_df_to_md_table(table) + "\n")

    single_seed = [name for name, agg in aggregates.items() if agg.get("_n_seed_runs", 0) < 2]
    if single_seed:
        lines.append(
            f"_Note: {', '.join(single_seed)} has fewer than 2 seed runs on disk; std is reported "
            "as unavailable rather than a misleadingly precise 0.000._\n"
        )
    return "\n".join(lines)


def _build_seed_plots(outputs_dir: Path, report_dir: Path) -> str:
    groups = _discover_seed_metrics(outputs_dir)
    if not groups:
        return ""
    aggregates = {name: aggregate_seed_metrics(seed_metrics) for name, seed_metrics in groups.items()}
    plots_dir = outputs_dir / "plots" / "final_report"
    plots_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for metric in ("age_mae", "gender_accuracy"):
        names, means, stds = [], [], []
        for name, agg in aggregates.items():
            stats = agg.get(metric)
            if stats is None:
                continue
            names.append(name)
            means.append(stats["mean"])
            stds.append(stats["std"] or 0.0)
        if names:
            out = plot_mean_std_bar(names, np.array(means), np.array(stds), metric, plots_dir / f"seed_mean_std_{metric}.png")
            lines.append(_md_image(out, report_dir, f"{metric} mean +/- std across seeds"))
    return "\n".join(lines)


def _build_uncertainty_section(outputs_dir: Path, report_dir: Path) -> str:
    lines = ["## Uncertainty Evaluation\n"]
    lines.append(
        "**Important caveat: marginal coverage is not conditional coverage.** "
        "Conformal calibration (when used) targets *marginal* coverage -- "
        "averaged across the entire test set -- not coverage conditioned on "
        "age bucket, gender-label subgroup, or any other subpopulation. A "
        "bucket can be systematically under- or over-covered even while the "
        "overall test-set coverage exactly matches the target. The per-bucket "
        "tables and plots below exist specifically so this can be checked, "
        "not assumed away.\n"
    )

    primary_name, primary_metrics = _find_primary_test_metrics(outputs_dir)
    if primary_metrics is None:
        lines.append(_MISSING.format(cmd="python scripts/evaluate.py --checkpoint <primary checkpoint>") + "\n")
        return "\n".join(lines)
    lines.append(f"Primary model shown below: `{primary_name}`.\n")

    bucket_report = primary_metrics.get("age_metrics_by_bucket")
    lines.append("### Age MAE / Coverage / Width by Age Bucket (raw)\n")
    if bucket_report:
        table = pd.DataFrame([{"age_bucket": label, **stats} for label, stats in bucket_report.items()])
        lines.append(_df_to_md_table(table) + "\n")
    else:
        lines.append(_MISSING.format(cmd="python scripts/evaluate.py --checkpoint <primary checkpoint>") + "\n")

    calibrated_report = primary_metrics.get("age_metrics_by_bucket_calibrated")
    lines.append("### Age MAE / Coverage / Width by Age Bucket (after conformal calibration)\n")
    if calibrated_report:
        table = pd.DataFrame([{"age_bucket": label, **stats} for label, stats in calibrated_report.items()])
        lines.append(_df_to_md_table(table) + "\n")
    else:
        lines.append(
            "_Calibrated per-bucket metrics unavailable -- run `python scripts/calibrate.py` "
            "then re-run `python scripts/evaluate.py` against the same checkpoint._\n"
        )

    from src.evaluation.reports import _experiment_search_dirs

    plots_dirs = _experiment_search_dirs(outputs_dir, "plots")
    plot_specs = (
        (f"{primary_name}_test_metrics_interval_coverage.png", "Empirical interval coverage by age bucket"),
        (f"{primary_name}_test_metrics_interval_width_by_bucket.png", "Interval width by age bucket"),
        (f"{primary_name}_test_metrics_coverage_width_tradeoff.png", "Coverage-width trade-off before/after conformal calibration"),
    )
    for filename, label in plot_specs:
        path = next((d / filename for d in plots_dirs if (d / filename).exists()), plots_dirs[0] / filename)
        if path.exists():
            lines.append(_md_image(path, report_dir, label))
        else:
            lines.append(f"_{label} plot not yet generated (`{filename}` not found)._\n")

    lines.append("### Narrowest and Widest Prediction Intervals\n")
    examples = primary_metrics.get("interval_examples")
    if examples:
        for kind in ("narrowest", "widest"):
            lines.append(f"**{kind.capitalize()}**\n")
            lines.append(_df_to_md_table(pd.DataFrame(examples[kind])) + "\n")
    else:
        lines.append(_MISSING.format(cmd="python scripts/evaluate.py --checkpoint <primary checkpoint>") + "\n")

    return "\n".join(lines)


def _build_robustness_section(outputs_dir: Path, report_dir: Path) -> str:
    lines = ["## Robustness Degradation\n"]

    diff_table = _read_csv(outputs_dir / "backbone_comparison" / "robustness_diff_table.csv")
    if diff_table is not None:
        lines.append(
            "**All pairwise model robustness comparisons** (one row per corruption, "
            "severity, and model pair -- see `scripts/compare_backbones.py --robustness-csv`):\n\n"
            + _df_to_md_table(diff_table) + "\n"
        )
    else:
        lines.append(
            "_Pairwise robustness comparison not yet generated. Run "
            "`python scripts/run_robustness.py --checkpoint <checkpoint>` for each model, then "
            "`python scripts/compare_backbones.py --robustness-csv NAME=path/to/robustness_results.csv ...` "
            "for all of them together._\n"
        )

    from src.evaluation.reports import _experiment_search_dirs

    robustness_dirs = _experiment_search_dirs(outputs_dir, "robustness")
    found_any = False
    for robustness_dir in robustness_dirs:
        df = _read_csv(robustness_dir / "robustness_results.csv")
        if df is None:
            continue
        found_any = True
        label = robustness_dir.parent.parent.name if robustness_dir.parent.name != "outputs" else "outputs/robustness"
        lines.append(f"### {label}\n")
        clean = df[df["corruption"] == "clean"]
        if not clean.empty:
            lines.append("**Clean baseline**\n\n" + _df_to_md_table(clean) + "\n")
        corrupted = df[df["corruption"] != "clean"]
        summary_cols = [
            c for c in (
                "age_mae", "gender_accuracy", "abstention_rate", "mean_confidence",
                "mean_interval_width", "interval_coverage_calibrated", "mean_interval_width_calibrated",
            )
            if c in df.columns
        ]
        if not corrupted.empty and summary_cols:
            summary = corrupted.groupby("corruption")[summary_cols].mean().reset_index()
            lines.append("**Mean metrics by corruption type (across severities)**\n\n" + _df_to_md_table(summary) + "\n")
        for metric in ("age_mae", "gender_accuracy", "abstention_rate"):
            plot_path = robustness_dir / f"robustness_{metric}.png"
            if plot_path.exists():
                lines.append(_md_image(plot_path, report_dir, f"{label}: robustness curve ({metric})"))
            degradation_plot_path = robustness_dir / f"degradation_{metric}_pct_change.png"
            if degradation_plot_path.exists():
                lines.append(_md_image(degradation_plot_path, report_dir, f"{label}: degradation vs. severity ({metric} % change)"))

    if not found_any and diff_table is None:
        lines.append(_MISSING.format(cmd="python scripts/run_robustness.py --checkpoint <checkpoint>") + "\n")
    return "\n".join(lines)


def _build_parameter_latency_section(outputs_dir: Path, report_dir: Path) -> str:
    lines = ["## Parameter Count and Inference Latency Comparison\n"]
    experiment_results = discover_experiment_results(outputs_dir / "metrics")
    labels, params, latencies = [], [], []
    for name, result in experiment_results.items():
        total_params = result.get("parameter_breakdown", {}).get("total_parameters")
        latency = result.get("test_metrics", {}).get("latency_ms_per_image")
        if total_params is not None and latency is not None:
            labels.append(name)
            params.append(total_params)
            latencies.append(latency)

    if not labels:
        lines.append(
            _MISSING.format(cmd="python scripts/run_experiments.py (trains + evaluates each experiment)") + "\n"
        )
        return "\n".join(lines)

    plots_dir = outputs_dir / "plots" / "final_report"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_parameter_latency_comparison(labels, params, latencies, plots_dir / "parameter_latency_comparison.png")
    lines.append(_md_image(out_path, report_dir, "Parameter count vs inference latency per experiment"))
    table = pd.DataFrame({"experiment": labels, "total_parameters": params, "latency_ms_per_image": latencies})
    lines.append(_df_to_md_table(table) + "\n")
    return "\n".join(lines)


def _build_findings_section(outputs_dir: Path) -> str:
    from src.evaluation.reports import _PLAIN_DEEP18_EXPERIMENT, _experiment_search_dirs

    findings = []

    cnn_metrics = _load_merged_experiment_metrics(outputs_dir, _CNN_EXPERIMENT)
    plain_metrics = _load_merged_experiment_metrics(outputs_dir, _PLAIN_DEEP18_EXPERIMENT)
    resnet_metrics = _load_merged_experiment_metrics(outputs_dir, _RESNET_EXPERIMENT)

    if cnn_metrics and resnet_metrics and cnn_metrics.get("age_mae") is not None and resnet_metrics.get("age_mae") is not None:
        findings.append(
            "**Efficiency/accuracy trade-off (SimpleCNN vs. Custom ResNet-18, *not* a residual-connection "
            "ablation -- depth and width differ too):** " + _backbone_comparison_interpretation(cnn_metrics, resnet_metrics).strip()
        )
    if plain_metrics and resnet_metrics and plain_metrics.get("age_mae") is not None and resnet_metrics.get("age_mae") is not None:
        findings.append(
            "**Residual-connection ablation (PlainDeep18NoSkip vs. Custom ResNet-18, depth/width held fixed):** "
            + _backbone_comparison_interpretation(plain_metrics, resnet_metrics).strip()
        )

    interpretation_path = outputs_dir / "backbone_comparison" / "final_interpretation.md"
    if interpretation_path.exists():
        findings.append(
            "**Selective-risk (AURC) comparison:** see the \"Selective-Risk (AURC) Comparison and Final "
            "Interpretation\" section below for the statistically-gated verdict on whether ResNet's added "
            "complexity is justified."
        )

    for robustness_dir in _experiment_search_dirs(outputs_dir, "robustness"):
        df = _read_csv(robustness_dir / "robustness_results.csv")
        if df is None or "age_mae" not in df.columns:
            continue
        clean_row = df[df["corruption"] == "clean"]
        corrupted = df[df["corruption"] != "clean"]
        if clean_row.empty or corrupted.empty:
            continue
        clean_mae = float(clean_row.iloc[0]["age_mae"])
        worst = corrupted.loc[corrupted["age_mae"].idxmax()]
        findings.append(
            f"Under the measured corruptions, age MAE degraded from {clean_mae:.2f} years (clean) to "
            f"as much as {float(worst['age_mae']):.2f} years under '{worst['corruption']}' at severity "
            f"{int(worst['severity'])}, an increase of {float(worst['age_mae']) - clean_mae:.2f} years."
        )
        break  # one representative robustness finding is enough here; see the full section above for all models

    lines = ["## Evidence-Based Findings\n"]
    if not findings:
        lines.append(
            "_No findings are stated yet because the underlying experiments/evaluations have not been run "
            "in this environment. Run `python scripts/run_experiments.py`, `python scripts/run_seeds.py`, "
            "and `python scripts/run_robustness.py`, then re-run this report to populate this section with "
            "real, measured results._\n"
        )
    else:
        for finding in findings:
            lines.append(f"- {finding}\n")
    return "\n".join(lines)


def _build_aurc_comparison_section(outputs_dir: Path) -> str:
    """Selective-risk (AURC) paired-bootstrap comparison and the final,
    explicitly conditional interpretation from scripts/compare_backbones.py.

    Only ``pairwise_bootstrap_aurc`` (the CI computed on the AURC summary
    statistic itself) is rendered as evidence for an AURC claim -- see
    src/evaluation/backbone_comparison.py:build_final_interpretation.
    """
    lines = ["## Selective-Risk (AURC) Comparison and Final Interpretation\n"]
    comparison_dir = outputs_dir / "backbone_comparison"
    gender_aurc = _read_json(comparison_dir / "gender_aurc.json")
    age_aurc = _read_json(comparison_dir / "age_selective_aurc.json")
    gender_ci = _read_json(comparison_dir / "gender_aurc_bootstrap.json")
    age_ci = _read_json(comparison_dir / "age_aurc_bootstrap.json")

    if gender_aurc is None and age_aurc is None:
        lines.append(
            _MISSING.format(
                cmd="python scripts/compare_backbones.py --checkpoint NAME=path ... --resnet-name <resnet>"
            ) + "\n"
        )
        return "\n".join(lines)

    if gender_aurc is not None:
        lines.append("**Gender selective-risk AURC (lower is better)**\n\n" + _dict_to_md_table(gender_aurc["aurc"]) + "\n")
    if age_aurc is not None:
        lines.append("**Age selective-MAE AURC (lower is better)**\n\n" + _dict_to_md_table(age_aurc["aurc"]) + "\n")

    for label, ci_data in (("gender", gender_ci), ("age", age_ci)):
        if not ci_data:
            continue
        rows = [
            {"comparison": f"{other}_vs_primary", **ci}
            for other, ci in ci_data.items()
        ]
        if rows:
            lines.append(
                f"**{label.capitalize()} AURC paired bootstrap CI (primary = ResNet)**\n\n"
                + _df_to_md_table(pd.DataFrame(rows)) + "\n"
            )

    interpretation_path = comparison_dir / "final_interpretation.md"
    if interpretation_path.exists():
        lines.append(interpretation_path.read_text(encoding="utf-8"))
    else:
        lines.append(_MISSING.format(cmd="python scripts/compare_backbones.py ...") + "\n")
    return "\n".join(lines)


def generate_final_results_report(outputs_dir: str | Path, report_dir: str | Path) -> str:
    """Build the report's Markdown text.

    ``report_dir`` is the directory the resulting Markdown will actually be
    written into (not necessarily related to ``outputs_dir`` or any
    repository root) -- every embedded image link is computed relative to
    it via ``os.path.relpath``, so this must be the real destination
    directory for links to resolve correctly once written to disk.
    """
    outputs_dir, report_dir = Path(outputs_dir), Path(report_dir)
    sections = [
        "# Final Results Report\n",
        (
            "Auto-generated from real saved artifacts under `outputs/` only. Any "
            "section whose backing artifact does not exist yet renders an explicit "
            "\"not yet generated\" message with the command that would produce it, "
            "rather than a fabricated number. Regenerate with "
            "`python scripts/generate_final_report.py` after (re-)running the "
            "relevant experiment/evaluation/robustness scripts.\n"
        ),
        (
            "**Scope note.** This is a research/education artifact. Dataset "
            "gender-label predictions reflect labels defined by the source "
            "dataset's documentation, not a determination of gender identity, and "
            "this system must not be used for employment, policing, surveillance, "
            "identity verification, medical diagnosis, admissions, insurance, or "
            "other high-impact decisions.\n"
        ),
        _build_ablation_section(outputs_dir),
        build_backbone_comparison_section(outputs_dir),
        _build_aurc_comparison_section(outputs_dir),
        _build_seed_aggregate_section(outputs_dir),
        _build_seed_plots(outputs_dir, report_dir),
        _build_uncertainty_section(outputs_dir, report_dir),
        _build_robustness_section(outputs_dir, report_dir),
        _build_parameter_latency_section(outputs_dir, report_dir),
        _build_findings_section(outputs_dir),
    ]
    return "\n".join(s for s in sections if s)


def save_final_results_report(outputs_dir: str | Path, docs_dir: str | Path) -> Path:
    out_path = Path(docs_dir) / "final_results_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = generate_final_results_report(outputs_dir, report_dir=out_path.parent)
    out_path.write_text(report, encoding="utf-8")
    return out_path
