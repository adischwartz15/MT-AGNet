#!/usr/bin/env python
"""CLI: deterministic robustness evaluation across corruption types and severities.

Output defaults to a checkpoint/experiment/seed-specific directory, never
a single shared ``outputs/robustness`` two different checkpoints would
silently overwrite -- see ``_default_output_dir`` below. When
``--calibration-dir`` (or its checkpoint/experiment/seed-derived default)
resolves to a real calibration artifact, its provenance is validated
against this checkpoint/split before its *fixed* conformal offset is
applied to every condition's predictions (clean and corrupted alike), so
both raw and calibrated coverage/width are reported and comparable across
severities.

Evaluates the full test split by default; pass ``--max-samples`` to
deterministically stratified-sample (by age bucket x gender label) down
to about that many rows for speed, rather than truncating to whatever
rows happen to sort first in the split CSV.

Usage:
    python scripts/run_robustness.py --checkpoint checkpoints/multitask_best_balanced_score.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.transforms import resolve_eval_transform
from src.evaluation.calibration import compute_preprocessing_fingerprint, load_calibration, validate_calibration_artifact
from src.evaluation.robustness import (
    apply_corruption, build_robustness_diff_table, compute_degradation, corruption_summary, evaluate_condition,
    iter_corruption_configs, stratified_sample,
)
from src.inference.artifacts import load_model_checkpoint
from src.utils.config import REPO_ROOT, load_config, resolve_device
from src.utils.io import checkpoint_experiment_name, save_json
from src.utils.logging import get_logger
from src.utils.visualization import plot_robustness_curves

logger = get_logger("scripts.run_robustness")


def _default_output_dir(checkpoint_path: Path) -> Path:
    """Checkpoint/experiment/seed-specific default, never the shared global outputs/robustness/.

    If the checkpoint lives in the isolated
    ``experiments/<experiment>/seed_<seed>/checkpoints/`` layout (see
    ``src/utils/experiment_paths.py``), reuse that same run's own
    ``robustness/`` subdirectory. Otherwise (e.g. a legacy flat
    ``checkpoints/`` directory) fall back to a directory named after this
    checkpoint's own experiment name -- still never shared across
    checkpoints.
    """
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "robustness"
    return REPO_ROOT / "outputs" / "robustness" / checkpoint_experiment_name(str(checkpoint_path))


def _default_calibration_dir(checkpoint_path: Path) -> Path | None:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent / "calibration"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None, help="Default: checkpoint/experiment/seed-specific (see _default_output_dir)")
    parser.add_argument("--calibration-dir", default=None, help="Default: this checkpoint's own isolated calibration dir, if any")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Deterministically stratified-sample (age bucket x gender) down to about this many test rows. Default: full test split.",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    device = resolve_device("auto")
    model, config, _ = load_model_checkpoint(checkpoint_path, device)
    robustness_cfg = load_config(REPO_ROOT / "configs" / "robustness.yaml")["robustness"]

    splits_path = REPO_ROOT / config["paths"]["splits_dir"] / "full_metadata_with_splits.csv"
    if not splits_path.exists():
        logger.error("No prepared split found at %s.", splits_path)
        return 1
    df = pd.read_csv(splits_path)
    full_test_df = df[df["split"] == "test"]
    test_df = stratified_sample(full_test_df, args.max_samples, seed=robustness_cfg.get("seed", 42))

    # Model-aware preprocessing -- the exact same deterministic clean-eval
    # transform this checkpoint's own validation/calibration/test paths use
    # (see src/data/transforms.py::resolve_eval_transform), so the "clean"
    # robustness baseline row is directly comparable to the ordinary test
    # metrics for this same checkpoint, and never silently wrong for a
    # pretrained-ResNet checkpoint.
    transform = resolve_eval_transform(model, config)
    confidence_threshold = config["model"]["gender_head"].get("confidence_threshold", 0.80)
    seed = robustness_cfg.get("seed", 42)

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "sample_corrupted_images"
    samples_dir.mkdir(parents=True, exist_ok=True)

    calibration_dir = Path(args.calibration_dir) if args.calibration_dir else _default_calibration_dir(checkpoint_path)
    calibration = load_calibration(calibration_dir) if calibration_dir else None
    if calibration is not None:
        preprocessing_fingerprint = compute_preprocessing_fingerprint(
            transform.image_size, transform.mean, transform.std,
            transform.interpolation, getattr(transform, "crop_pct", 1.0),
        )
        validate_calibration_artifact(
            calibration, checkpoint_path=checkpoint_path, split_csv_path=splits_path,
            model_id=getattr(model, "model_id", None), pretrained_source=getattr(model, "pretrained_source", None),
            preprocessing_fingerprint=preprocessing_fingerprint,
        )
        logger.info("Applying fixed conformal offset=%.4f (from %s) to every condition.", calibration["offset"], calibration_dir)
    else:
        logger.info("No calibration artifact found/given; reporting raw (uncalibrated) coverage/width only.")
    calibration_offset = calibration["offset"] if calibration is not None else None

    save_json(
        {
            "n_sampled": len(test_df), "n_full_test_split": len(full_test_df),
            "max_samples_requested": args.max_samples, "seed": seed,
            "age_bucket_counts": {
                str(k): int(v) for k, v in pd.cut(test_df["age"], bins=[0, 13, 20, 35, 50, 65, 200], right=False)
                .value_counts().sort_index().items()
            },
            "gender_label_counts": {str(k): int(v) for k, v in test_df["gender_label"].value_counts().items()},
            "sample_ids": test_df["image_path"].tolist(),
        },
        output_dir / "sampling_metadata.json",
    )

    results = [
        evaluate_condition(
            model, test_df, transform, device, confidence_threshold, None, 0, None, seed,
            calibration_offset=calibration_offset,
        )
    ]
    for corruption_name, severity, param in iter_corruption_configs(robustness_cfg):
        logger.info("Evaluating %s severity=%d (param=%s)", corruption_name, severity, param)
        metrics = evaluate_condition(
            model, test_df, transform, device, confidence_threshold, corruption_name, severity, param, seed,
            calibration_offset=calibration_offset,
        )
        results.append(metrics)

    from PIL import Image

    n_samples = robustness_cfg.get("num_samples_per_corruption_plot", 6)
    for corruption_name, severity, param in iter_corruption_configs(robustness_cfg):
        if severity != 1:
            continue
        for i, row in enumerate(test_df.head(n_samples).to_dict("records")):
            with Image.open(row["image_path"]) as img:
                corrupted = apply_corruption(img.convert("RGB"), corruption_name, param, seed=seed + i)
                corrupted.save(samples_dir / f"{corruption_name}_sample{i}.png")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "robustness_results.csv", index=False)

    performance_metrics = ("age_mae", "gender_accuracy", "abstention_rate", "interval_coverage_calibrated", "mean_interval_width_calibrated")
    for metric in performance_metrics:
        if metric in results_df.columns:
            corrupted_only = results_df[results_df["corruption"] != "clean"]
            if not corrupted_only.empty:
                plot_robustness_curves(corrupted_only, metric, output_dir / f"robustness_{metric}.png")

    # Degradation-vs-severity plots (delta/pct-change relative to the clean
    # baseline), in addition to the raw performance-vs-severity plots above
    # -- performance alone doesn't show *how much worse* a model got, which
    # is the quantity actually being compared across models/experiments.
    degraded_df = compute_degradation(results_df)
    degraded_df.to_csv(output_dir / "robustness_degradation.csv", index=False)
    degraded_corrupted_only = degraded_df[degraded_df["corruption"] != "clean"]
    for metric in ("age_mae", "gender_accuracy", "abstention_rate"):
        pct_col = f"{metric}_pct_change"
        if pct_col in degraded_corrupted_only.columns and not degraded_corrupted_only.empty:
            plot_robustness_curves(degraded_corrupted_only, pct_col, output_dir / f"degradation_{metric}_pct_change.png")

    # Programmatic corruption-type/condition count (never a hand-maintained
    # doc claim that can silently drift from the actual configs/robustness.yaml).
    corruption_stats = corruption_summary(robustness_cfg)
    save_json(corruption_stats, output_dir / "corruption_summary.json")

    summary_lines = ["# Robustness Evaluation Summary\n"]
    summary_lines.append(
        f"**Corruption coverage:** {corruption_stats['n_corruption_types']} corruption types "
        f"({', '.join(corruption_stats['corruption_type_names'])}), "
        f"{corruption_stats['n_total_conditions']} total (type x severity) conditions.\n"
    )
    clean_row = results_df[results_df["corruption"] == "clean"].iloc[0].to_dict()
    summary_lines.append(f"**Clean baseline:** {clean_row}\n")
    for corruption_name in results_df["corruption"].unique():
        if corruption_name == "clean":
            continue
        subset = results_df[results_df["corruption"] == corruption_name]
        summary_lines.append(f"## {corruption_name}\n")
        summary_lines.append(subset.to_string(index=False) + "\n")
    (output_dir / "robustness_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    logger.info("Saved robustness results to %s", output_dir)
    print(f"Saved robustness CSV/plots/summary to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
