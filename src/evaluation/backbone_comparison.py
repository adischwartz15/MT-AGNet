"""Cross-model backbone comparison analyses (clean-test summary, gender and
age selective-prediction / risk-coverage analysis, tail-error analysis, and
an honest conditional interpretation), used by ``scripts/compare_backbones.py``.

Every function here operates on already-computed per-sample prediction
arrays (the same ``preds`` dict shape ``scripts/evaluate.py:run_inference``
returns) -- nothing here re-runs a model or duplicates training/inference
logic. All numeric outputs are either directly measured or standard,
documented aggregate statistics (percentiles, AURC, bootstrap CIs); nothing
is fabricated, and the "is the added complexity justified" interpretation
is explicitly conditional on the measured numbers (see
``build_final_interpretation``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    abstention_rate, age_error_percentiles, age_mae, age_r2, age_rmse, age_tail_error_rates,
    age_uncertainty_by_bucket, confidence_statistics, gender_accuracy, gender_coverage,
    gender_effective_accuracy, interval_coverage, mean_interval_width,
)
from src.evaluation.selective import (
    compute_aurc, paired_bootstrap_aurc_diff_ci, paired_bootstrap_risk_diff_ci, risk_at_coverage,
    selective_risk_coverage_curve,
)

COMMON_COVERAGE_LEVELS = (0.80, 0.90, 0.95, 0.98)
AGE_ERROR_TAIL_THRESHOLDS = (5, 10, 15, 20)
DEVELOPMENTAL_AGE_BUCKETS = (
    (0, 13, "0-12"), (13, 20, "13-19"), (20, 35, "20-34"),
    (35, 50, "35-49"), (50, 65, "50-64"), (65, 200, "65+"),
)


def _gender_arrays(preds: dict, confidence_threshold: float) -> dict:
    mask = preds["gender_mask"].astype(bool)
    probs = preds["probs"][mask]
    y_true = preds["gender"][mask].astype(int)
    predicted = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    abstain = confidence < confidence_threshold
    sample_id = preds.get("sample_id")
    return {
        "y_true": y_true, "predicted": predicted, "confidence": confidence, "abstain": abstain,
        "sample_id": sample_id[mask] if sample_id is not None else None,
    }


def _assert_paired_alignment(sample_ids_a, sample_ids_b, name_a: str, name_b: str) -> None:
    """Raise unless two models' per-sample arrays are provably index-aligned.

    Equal array *length* is not sufficient evidence of alignment -- a
    different row order, different upstream filtering, or even a
    different split file entirely could still produce arrays of the same
    length that silently pair up the wrong samples. This checks the
    actual ordered sample identifiers (``scripts/evaluate.py:run_inference``'s
    ``"sample_id"``, the test-split image path in row order) match exactly
    before any paired statistic (paired bootstrap CI, direct per-sample
    diff) is computed from them.
    """
    if sample_ids_a is None or sample_ids_b is None:
        raise ValueError(
            f"Cannot verify '{name_a}' and '{name_b}' were evaluated on the identical, "
            "index-aligned test set: sample IDs were not recorded for one or both models "
            "(predictions must come from scripts/evaluate.py:run_inference, which returns them)."
        )
    if len(sample_ids_a) != len(sample_ids_b) or not np.array_equal(sample_ids_a, sample_ids_b):
        raise ValueError(
            f"'{name_a}' and '{name_b}' are not index-aligned: their ordered test-sample IDs "
            "differ (different order, different filtering, or a different split file). Equal "
            "array length alone is not sufficient evidence of alignment for a paired comparison."
        )


def build_clean_test_summary(
    model_name: str, preds: dict, confidence_threshold: float, calibration: dict | None = None,
    parameter_breakdown: dict | None = None, mean_epoch_time_seconds: float | None = None,
) -> dict:
    """One model's full clean-test-set summary row (Part B.1 of the backbone comparison spec)."""
    summary: dict = {"model": model_name}
    age_mask = preds["age_mask"].astype(bool)
    if age_mask.any():
        y_true = preds["age"][age_mask]
        q10, q50, q90 = preds["q10"][age_mask], preds["q50"][age_mask], preds["q90"][age_mask]
        summary.update({
            "age_mae": age_mae(y_true, q50),
            "age_rmse": age_rmse(y_true, q50),
            "age_r2": age_r2(y_true, q50),
            **{f"age_error_{k}": v for k, v in age_error_percentiles(y_true, q50).items()},
            **{f"age_error_frac_{k}": v for k, v in age_tail_error_rates(y_true, q50, AGE_ERROR_TAIL_THRESHOLDS).items()},
            "raw_interval_coverage": interval_coverage(y_true, q10, q90),
            "raw_interval_width": mean_interval_width(q10, q90),
        })
        if calibration is not None:
            from src.evaluation.calibration import apply_conformal_offset

            q10_cal, q90_cal = apply_conformal_offset(q10, q90, calibration["offset"])
            summary["calibrated_interval_coverage"] = interval_coverage(y_true, q10_cal, q90_cal)
            summary["calibrated_interval_width"] = mean_interval_width(q10_cal, q90_cal)

    gender_mask = preds["gender_mask"].astype(bool)
    if gender_mask.any():
        g = _gender_arrays(preds, confidence_threshold)
        summary.update({
            "gender_selective_accuracy": gender_accuracy(g["y_true"], g["predicted"], g["abstain"]),
            "gender_coverage": gender_coverage(g["abstain"]),
            "gender_abstention_rate": abstention_rate(g["abstain"]),
            "gender_effective_accuracy": gender_effective_accuracy(g["y_true"], g["predicted"], g["abstain"]),
            "gender_confidence_stats": confidence_statistics(g["confidence"]),
        })

    summary["latency_ms_per_image"] = preds.get("latency_ms_per_image")
    if parameter_breakdown:
        summary["total_parameters"] = parameter_breakdown.get("total_parameters")
        summary["backbone_parameters"] = parameter_breakdown.get("backbone_parameters")
    summary["mean_epoch_time_seconds"] = mean_epoch_time_seconds
    return summary


def build_clean_test_table(summaries: dict[str, dict]) -> pd.DataFrame:
    """One row per model; flattens nested confidence-stats dicts into top-level columns."""
    rows = []
    for name, summary in summaries.items():
        row = {k: v for k, v in summary.items() if not isinstance(v, dict)}
        for prefix, nested in summary.items():
            if isinstance(nested, dict):
                for k, v in nested.items():
                    row[f"{prefix}_{k}"] = v
        row["model"] = name
        rows.append(row)
    return pd.DataFrame(rows)


def build_gender_risk_coverage_analysis(
    models_preds: dict[str, dict], confidence_threshold: float, primary_model: str | None = None,
) -> dict:
    """Gender selective-risk-vs-coverage analysis across models (Part B.2).

    Returns ``{"curves": {model: (coverages, risks)}, "aurc": {model: float},
    "at_coverage": DataFrame, "pairwise_bootstrap": {model: ci_dict},
    "pairwise_bootstrap_aurc": {model: ci_dict}}``.

    ``pairwise_bootstrap`` compares every non-primary model against
    ``primary_model`` (default: the first model) at each common coverage
    level; ``pairwise_bootstrap_aurc`` compares the scalar AURC statistic
    itself. Both use the paired bootstrap, valid only when both models
    share the identical, index-aligned test samples -- this is verified
    via each model's recorded ``sample_id`` (see
    ``scripts/evaluate.py:run_inference``), not merely equal array length,
    before any paired statistic is computed; a genuine mismatch raises
    rather than silently skipping or mispairing samples.
    """
    curves, aurc, confidences, losses, sample_ids = {}, {}, {}, {}, {}
    for name, preds in models_preds.items():
        g = _gender_arrays(preds, confidence_threshold)
        loss = (g["predicted"] != g["y_true"]).astype(float)
        coverages, risks = selective_risk_coverage_curve(g["confidence"], loss)
        curves[name] = (coverages, risks)
        aurc[name] = compute_aurc(coverages, risks)
        confidences[name], losses[name] = g["confidence"], loss
        sample_ids[name] = g["sample_id"]

    at_coverage_rows = []
    for level in COMMON_COVERAGE_LEVELS:
        row = {"coverage": level}
        for name in models_preds:
            row[f"{name}_risk"] = risk_at_coverage(*curves[name], level)
        at_coverage_rows.append(row)
    at_coverage_table = pd.DataFrame(at_coverage_rows)

    pairwise_bootstrap, pairwise_bootstrap_aurc = {}, {}
    primary = primary_model or next(iter(models_preds))
    if primary in confidences:
        for name in models_preds:
            if name == primary:
                continue
            _assert_paired_alignment(sample_ids[primary], sample_ids[name], primary, name)
            pairwise_bootstrap[name] = {
                level: paired_bootstrap_risk_diff_ci(
                    confidences[primary], losses[primary], confidences[name], losses[name], target_coverage=level,
                )
                for level in COMMON_COVERAGE_LEVELS
            }
            pairwise_bootstrap_aurc[name] = paired_bootstrap_aurc_diff_ci(
                confidences[primary], losses[primary], confidences[name], losses[name],
            )

    return {
        "curves": curves, "aurc": aurc, "at_coverage": at_coverage_table,
        "pairwise_bootstrap": pairwise_bootstrap, "pairwise_bootstrap_aurc": pairwise_bootstrap_aurc,
    }


def build_age_selective_analysis(models_preds: dict[str, dict], primary_model: str | None = None) -> dict:
    """Age selective-prediction analysis using interval width as the confidence score (Part B.3).

    Same paired-alignment and AURC-bootstrap treatment as
    :func:`build_gender_risk_coverage_analysis` (see its docstring):
    ``pairwise_bootstrap`` is per fixed coverage level, ``pairwise_bootstrap_aurc``
    is for the scalar AURC statistic, and alignment is checked via each
    model's recorded ``sample_id`` rather than array length alone.
    """
    mae_curves, rmse_curves, aurc, confidences, abs_errors, sample_ids = {}, {}, {}, {}, {}, {}
    for name, preds in models_preds.items():
        mask = preds["age_mask"].astype(bool)
        y_true, q10, q50, q90 = preds["age"][mask], preds["q10"][mask], preds["q50"][mask], preds["q90"][mask]
        confidence = -(q90 - q10)  # narrower interval = higher confidence
        errors = np.abs(y_true - q50)
        mae_coverages, mae_risks = selective_risk_coverage_curve(confidence, errors)
        _, rmse_risks = selective_risk_coverage_curve(confidence, errors ** 2)
        rmse_risks = np.sqrt(rmse_risks)

        mae_curves[name] = (mae_coverages, mae_risks)
        rmse_curves[name] = (mae_coverages, rmse_risks)
        aurc[name] = compute_aurc(mae_coverages, mae_risks)
        confidences[name], abs_errors[name] = confidence, errors
        sample_id = preds.get("sample_id")
        sample_ids[name] = sample_id[mask] if sample_id is not None else None

    at_coverage_rows = []
    for level in COMMON_COVERAGE_LEVELS:
        row = {"coverage": level}
        for name in models_preds:
            row[f"{name}_mae"] = risk_at_coverage(*mae_curves[name], level)
        at_coverage_rows.append(row)
    at_coverage_table = pd.DataFrame(at_coverage_rows)

    pairwise_bootstrap, pairwise_bootstrap_aurc = {}, {}
    primary = primary_model or next(iter(models_preds))
    if primary in confidences:
        for name in models_preds:
            if name == primary:
                continue
            _assert_paired_alignment(sample_ids[primary], sample_ids[name], primary, name)
            pairwise_bootstrap[name] = {
                level: paired_bootstrap_risk_diff_ci(
                    confidences[primary], abs_errors[primary], confidences[name], abs_errors[name],
                    target_coverage=level,
                )
                for level in COMMON_COVERAGE_LEVELS
            }
            pairwise_bootstrap_aurc[name] = paired_bootstrap_aurc_diff_ci(
                confidences[primary], abs_errors[primary], confidences[name], abs_errors[name],
            )

    return {
        "mae_curves": mae_curves, "rmse_curves": rmse_curves, "aurc": aurc,
        "at_coverage": at_coverage_table, "pairwise_bootstrap": pairwise_bootstrap,
        "pairwise_bootstrap_aurc": pairwise_bootstrap_aurc,
    }


def build_tail_error_analysis(models_preds: dict[str, dict]) -> dict:
    """CDF data, tail-error-rate bars, and per-age-bucket MAE table across models (Part B.4)."""
    errors_by_model, tail_rates_by_model, bucket_tables = {}, {}, {}
    for name, preds in models_preds.items():
        mask = preds["age_mask"].astype(bool)
        y_true, q50, q10, q90 = preds["age"][mask], preds["q50"][mask], preds["q10"][mask], preds["q90"][mask]
        errors = np.abs(y_true - q50)
        errors_by_model[name] = errors
        tail_rates_by_model[name] = age_tail_error_rates(y_true, q50, AGE_ERROR_TAIL_THRESHOLDS)

        bucket_edges = [lo for lo, _, _ in DEVELOPMENTAL_AGE_BUCKETS] + [DEVELOPMENTAL_AGE_BUCKETS[-1][1]]
        raw_buckets = age_uncertainty_by_bucket(y_true, q10, q50, q90, bucket_edges=bucket_edges)
        relabeled = {label: raw_buckets[key] for (_, _, label), key in zip(DEVELOPMENTAL_AGE_BUCKETS, raw_buckets)}
        bucket_tables[name] = relabeled

    error_percentiles = {name: age_error_percentiles(preds["age"][preds["age_mask"].astype(bool)], preds["q50"][preds["age_mask"].astype(bool)]) for name, preds in models_preds.items()}

    bucket_rows = []
    for _, _, label in DEVELOPMENTAL_AGE_BUCKETS:
        row = {"age_bucket": label}
        for name in models_preds:
            stats = bucket_tables[name][label]
            row[f"{name}_count"] = stats["count"]
            row[f"{name}_mae"] = stats["mae"]
        bucket_rows.append(row)

    return {
        "errors_by_model": errors_by_model,
        "tail_rates_by_model": tail_rates_by_model,
        "error_percentiles": error_percentiles,
        "bucket_table": pd.DataFrame(bucket_rows),
    }


def build_final_interpretation(
    clean_summary_table: pd.DataFrame, gender_risk_analysis: dict, age_selective_analysis: dict,
    resnet_name: str, comparison_names: list[str],
) -> str:
    """Honest, conditional "is additional residual complexity justified?" narrative.

    Never asserts an advantage that isn't backed by the measured numbers,
    never treats a single-seed difference as decisive, and explicitly
    states the compact/plain alternative is preferred when results are
    tied or favor it -- the whole point of this analysis is to be capable
    of concluding *against* the residual architecture.

    ``gender_risk_analysis`` / ``age_selective_analysis`` must have been
    built with ``primary_model=resnet_name`` (see
    ``build_gender_risk_coverage_analysis`` / ``build_age_selective_analysis``),
    since their ``pairwise_bootstrap_aurc[other]`` entries are defined as
    ResNet-vs-``other`` (``aurc_diff_b_minus_a = AURC(other) - AURC(resnet)``;
    positive means ResNet has *lower* AURC, i.e. an advantage).

    A claim of "statistically supported AURC advantage" is gated strictly
    on ``pairwise_bootstrap_aurc`` (the bootstrap CI computed on the AURC
    summary statistic itself), never on ``pairwise_bootstrap`` (which is
    only a CI at fixed coverage levels and is not evidence about AURC).
    """
    lines = ["## Is Additional Residual Complexity Justified?\n"]

    if resnet_name not in clean_summary_table["model"].values:
        return "\n".join(lines) + (
            "_Not available -- the ResNet checkpoint has not been evaluated in this run._\n"
        )

    resnet_row = clean_summary_table[clean_summary_table["model"] == resnet_name].iloc[0]
    findings = []
    decisive_advantage_found = False

    for other in comparison_names:
        if other == resnet_name or other not in clean_summary_table["model"].values:
            continue
        other_row = clean_summary_table[clean_summary_table["model"] == other].iloc[0]

        param_diff = resnet_row.get("total_parameters", 0) - other_row.get("total_parameters", 0)
        latency_diff = (resnet_row.get("latency_ms_per_image") or 0) - (other_row.get("latency_ms_per_image") or 0)

        # pairwise_bootstrap_aurc[other] compares (a=resnet, b=other): a
        # positive aurc_diff_b_minus_a means "other" has higher AURC (worse)
        # than ResNet, i.e. a ResNet advantage on that AURC statistic.
        gender_aurc_ci = gender_risk_analysis.get("pairwise_bootstrap_aurc", {}).get(other)
        age_aurc_ci = age_selective_analysis.get("pairwise_bootstrap_aurc", {}).get(other)
        gender_significant = bool(gender_aurc_ci and gender_aurc_ci.get("excludes_zero") and gender_aurc_ci.get("aurc_diff_b_minus_a", 0) > 0)
        age_significant = bool(age_aurc_ci and age_aurc_ci.get("excludes_zero") and age_aurc_ci.get("aurc_diff_b_minus_a", 0) > 0)

        if gender_significant or age_significant:
            decisive_advantage_found = True
            findings.append(
                f"- vs. `{other}`: Custom ResNet-18 shows a statistically supported "
                f"(paired bootstrap CI on AURC itself excludes zero) reduction in selective "
                f"risk ({'gender AURC' if gender_significant else 'age AURC'}), "
                f"at the cost of {int(param_diff):+,} parameters and {latency_diff:+.2f} ms/image. "
                "This is a plausible deployment scenario where the added residual "
                "complexity pays for itself -- e.g. when tail-risk/selective-prediction "
                "quality at high coverage matters more than raw parameter/latency cost.\n"
            )
        else:
            findings.append(
                f"- vs. `{other}`: no statistically supported ResNet advantage was found "
                "in gender or age selective risk -- specifically, no paired bootstrap CI on "
                "the AURC summary statistic itself excludes zero in ResNet's favor. Given "
                f"ResNet costs {int(param_diff):+,} more parameters and {latency_diff:+.2f} ms/image "
                f"more latency, `{other}` is the preferred model for this dataset and training "
                "setup unless a specific downstream requirement (not evaluated here) favors ResNet.\n"
            )

    lines.extend(findings)
    lines.append(
        "\n**Caveat.** These conclusions reflect the seed(s), dataset, and coverage "
        "levels evaluated in this run only. A single-seed AURC difference, even if "
        "numerically in ResNet's favor, is not treated as decisive here unless the "
        "bootstrap CI excludes zero; see the mean +/- std table across seeds for "
        "additional evidence of stability before generalizing this conclusion.\n"
    )
    if not decisive_advantage_found:
        lines.append(
            "\n**Overall:** across the comparisons run, no measured evidence supports "
            "that the additional residual-connection complexity is justified for this "
            "dataset and training setup -- the compact/plain alternative(s) are at least "
            "as good on the metrics evaluated here.\n"
        )
    return "\n".join(lines)
