#!/usr/bin/env python
"""CLI: comprehensive multi-model backbone comparison, for post-hoc analysis of
already-trained checkpoints -- never retrains.

Runs the full Part-B analysis suite across two or more checkpoints:
clean-test summary (percentiles, tail-error rates, effective accuracy,
parameter/latency), gender selective-risk-coverage (AURC, paired bootstrap
CIs), age selective-prediction risk-coverage (interval width as the
confidence score), tail-error analysis (CDF, bucket table), an optional
robustness degradation comparison (if --robustness-csv is given per model),
and a final, explicitly conditional "is the added complexity justified"
interpretation (see src/evaluation/backbone_comparison.py -- it is capable
of concluding against the residual architecture, and does so whenever the
measured numbers don't support an advantage).

Only re-runs inference (a single forward pass per test-set image) against
each checkpoint's own test split; nothing here trains or fine-tunes a model.

Usage:
    python scripts/compare_backbones.py \\
        --checkpoint simple_cnn=checkpoints/exp_0_..._best_balanced_score.pt \\
        --checkpoint plain_deep18_no_skip=checkpoints/exp_0b_..._best_balanced_score.pt \\
        --checkpoint custom_resnet18=checkpoints/exp_d_..._best_balanced_score.pt \\
        --resnet-name custom_resnet18 \\
        --calibration-dir simple_cnn=outputs/calibration/exp_0 \\
        --calibration-dir custom_resnet18=outputs/calibration/exp_d \\
        --output-dir outputs/backbone_comparison
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate import run_inference  # noqa: E402

from src.data.dataset import FaceMultiTaskDataset  # noqa: E402
from src.data.transforms import EvalTransform  # noqa: E402
from src.evaluation.backbone_comparison import (  # noqa: E402
    build_age_selective_analysis, build_clean_test_summary, build_clean_test_table,
    build_final_interpretation, build_gender_risk_coverage_analysis, build_tail_error_analysis,
)
from src.evaluation.calibration import load_calibration  # noqa: E402
from src.evaluation.robustness import build_robustness_diff_table, compute_degradation  # noqa: E402
from src.inference.artifacts import load_model_checkpoint  # noqa: E402
from src.utils.config import REPO_ROOT, resolve_device  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.visualization import (  # noqa: E402
    plot_age_error_cdf, plot_pareto, plot_risk_coverage_curves, plot_tail_error_bars,
)

import torch  # noqa: E402

logger = get_logger("scripts.compare_backbones")


def _parse_name_value_pairs(pairs: list[str] | None) -> dict[str, str]:
    result = {}
    for item in pairs or []:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Expected NAME=VALUE, got '{item}'")
        name, value = item.split("=", 1)
        result[name] = value
    return result


def load_model_and_predictions(checkpoint_path: str, calibration_dir: str | None, batch_size: int = 64):
    device = resolve_device("auto")
    model, config, _ = load_model_checkpoint(checkpoint_path, device)

    splits_path = REPO_ROOT / config["paths"]["splits_dir"] / "full_metadata_with_splits.csv"
    if not splits_path.exists():
        raise FileNotFoundError(f"No prepared split found at {splits_path} for checkpoint {checkpoint_path}")
    df = pd.read_csv(splits_path)
    test_df = df[df["split"] == "test"]
    dataset = FaceMultiTaskDataset(test_df, EvalTransform(config["dataset"]["image_size"]))

    preds = run_inference(model, dataset, device, batch_size=batch_size)
    calibration = load_calibration(calibration_dir) if calibration_dir else None
    confidence_threshold = config["model"]["gender_head"].get("confidence_threshold", 0.80)
    parameter_breakdown = model.parameter_breakdown().as_dict()
    return preds, calibration, confidence_threshold, parameter_breakdown


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", action="append", required=True, help="NAME=PATH, repeatable (>= 2 required)")
    parser.add_argument("--calibration-dir", action="append", default=[], help="NAME=PATH, optional per-model calibration dir")
    parser.add_argument("--robustness-csv", action="append", default=[], help="NAME=PATH to a robustness_results.csv, optional per-model")
    parser.add_argument("--resnet-name", required=True, help="Which --checkpoint NAME is the Custom ResNet-18 model")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "backbone_comparison"))
    parser.add_argument("--mean-epoch-time", action="append", default=[], help="NAME=SECONDS, optional")
    args = parser.parse_args()

    checkpoints = _parse_name_value_pairs(args.checkpoint)
    calibration_dirs = _parse_name_value_pairs(args.calibration_dir)
    robustness_csvs = _parse_name_value_pairs(args.robustness_csv)
    mean_epoch_times = {k: float(v) for k, v in _parse_name_value_pairs(args.mean_epoch_time).items()}

    if len(checkpoints) < 2:
        logger.error("Need at least two --checkpoint entries to compare.")
        return 1
    if args.resnet_name not in checkpoints:
        logger.error("--resnet-name '%s' must match one of the --checkpoint NAMEs: %s", args.resnet_name, list(checkpoints))
        return 1

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    models_preds, clean_summaries, confidence_thresholds = {}, {}, {}
    for name, checkpoint_path in checkpoints.items():
        logger.info("Running inference for '%s' (%s)...", name, checkpoint_path)
        preds, calibration, confidence_threshold, parameter_breakdown = load_model_and_predictions(
            checkpoint_path, calibration_dirs.get(name)
        )
        models_preds[name] = preds
        confidence_thresholds[name] = confidence_threshold
        clean_summaries[name] = build_clean_test_summary(
            name, preds, confidence_threshold, calibration=calibration,
            parameter_breakdown=parameter_breakdown, mean_epoch_time_seconds=mean_epoch_times.get(name),
        )

    # 1. Clean test-set summary.
    clean_table = build_clean_test_table(clean_summaries)
    clean_table.to_csv(output_dir / "clean_test_summary.csv", index=False)
    if "total_parameters" in clean_table.columns and "age_mae" in clean_table.columns:
        plot_pareto(
            clean_table["model"].tolist(), clean_table["total_parameters"].tolist(), clean_table["age_mae"].tolist(),
            "Total parameters", "Age MAE", "Age MAE vs. parameter count", plots_dir / "pareto_params_vs_mae.png",
        )
    if "latency_ms_per_image" in clean_table.columns and "age_mae" in clean_table.columns:
        plot_pareto(
            clean_table["model"].tolist(), clean_table["latency_ms_per_image"].tolist(), clean_table["age_mae"].tolist(),
            "Latency (ms/image)", "Age MAE", "Age MAE vs. inference latency", plots_dir / "pareto_latency_vs_mae.png",
        )
    logger.info("Saved clean-test summary to %s", output_dir / "clean_test_summary.csv")

    # 2. Gender selective-risk-coverage analysis. Each model's own confidence
    # threshold determines its abstention decisions upstream (baked into
    # clean_summaries above); the risk-coverage sweep itself needs one
    # threshold only to decide the 0/1 correctness loss per sample, so the
    # primary (ResNet) model's own threshold is used for that loss definition.
    gender_analysis = build_gender_risk_coverage_analysis(
        models_preds, confidence_thresholds[args.resnet_name], primary_model=args.resnet_name
    )
    gender_analysis["at_coverage"].to_csv(output_dir / "gender_risk_at_coverage.csv", index=False)
    save_json({"aurc": gender_analysis["aurc"]}, output_dir / "gender_aurc.json")
    save_json(
        {name: {str(level): ci for level, ci in cis.items()} for name, cis in gender_analysis["pairwise_bootstrap"].items()},
        output_dir / "gender_pairwise_bootstrap.json",
    )
    # Bootstrap CI on the AURC summary statistic itself -- distinct from the
    # fixed-coverage-level CI above, and the only one build_final_interpretation
    # is allowed to cite as evidence of a "statistically supported AURC" claim.
    save_json(gender_analysis["pairwise_bootstrap_aurc"], output_dir / "gender_aurc_bootstrap.json")
    plot_risk_coverage_curves(gender_analysis["curves"], "Selective risk (1 - accuracy)", "Gender risk-coverage", plots_dir / "gender_risk_coverage.png")

    # 3. Age selective-prediction analysis.
    age_analysis = build_age_selective_analysis(models_preds, primary_model=args.resnet_name)
    age_analysis["at_coverage"].to_csv(output_dir / "age_selective_mae_at_coverage.csv", index=False)
    save_json({"aurc": age_analysis["aurc"]}, output_dir / "age_selective_aurc.json")
    save_json(
        {name: {str(level): ci for level, ci in cis.items()} for name, cis in age_analysis["pairwise_bootstrap"].items()},
        output_dir / "age_pairwise_bootstrap.json",
    )
    save_json(age_analysis["pairwise_bootstrap_aurc"], output_dir / "age_aurc_bootstrap.json")
    plot_risk_coverage_curves(age_analysis["mae_curves"], "Selective age MAE (years)", "Age selective-prediction risk-coverage", plots_dir / "age_risk_coverage_mae.png")
    plot_risk_coverage_curves(age_analysis["rmse_curves"], "Selective age RMSE (years)", "Age selective-prediction risk-coverage (RMSE)", plots_dir / "age_risk_coverage_rmse.png")

    # 4. Tail-error analysis.
    tail_analysis = build_tail_error_analysis(models_preds)
    tail_analysis["bucket_table"].to_csv(output_dir / "age_bucket_mae.csv", index=False)
    save_json(tail_analysis["error_percentiles"], output_dir / "age_error_percentiles.json")
    plot_age_error_cdf(tail_analysis["errors_by_model"], plots_dir / "age_error_cdf.png")
    plot_tail_error_bars(tail_analysis["tail_rates_by_model"], plots_dir / "age_tail_error_rates.png")

    # 5. Robustness comparison (optional -- only if per-model CSVs were supplied).
    if robustness_csvs:
        degraded = {}
        for name, csv_path in robustness_csvs.items():
            df = pd.read_csv(csv_path)
            degraded[name] = compute_degradation(df)
            degraded[name].to_csv(output_dir / f"robustness_degradation_{name}.csv", index=False)
        if len(degraded) >= 2:
            diff_table = build_robustness_diff_table(degraded)
            diff_table.to_csv(output_dir / "robustness_diff_table.csv", index=False)
        logger.info("Saved robustness degradation/diff tables for: %s", list(degraded))
    else:
        logger.info("No --robustness-csv provided; skipping robustness comparison (run scripts/run_robustness.py per model first).")

    # 6. Final, explicitly conditional interpretation.
    comparison_names = [name for name in checkpoints if name != args.resnet_name]
    interpretation = build_final_interpretation(clean_table, gender_analysis, age_analysis, args.resnet_name, comparison_names)
    (output_dir / "final_interpretation.md").write_text(interpretation, encoding="utf-8")
    print(interpretation)

    logger.info("Backbone comparison complete. Artifacts saved under %s", output_dir)
    print(f"Saved backbone comparison artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
