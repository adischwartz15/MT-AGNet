#!/usr/bin/env python
"""CLI: evaluate a trained checkpoint on the held-out test split.

Computes age MAE/RMSE/R2, interval coverage/width, calibration error,
per-age-bucket uncertainty metrics (MAE/coverage/width, before and after
conformal calibration when available), narrow/wide interval examples, and
dataset gender-label accuracy/confusion matrix/abstention rate. Optionally
compares against a k-NN baseline.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/multitask_best_balanced_score.pt [--compare-knn]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import FaceMultiTaskDataset
from src.data.transforms import resolve_eval_transform
from src.evaluation.calibration import apply_conformal_offset, load_calibration, validate_calibration_artifact
from src.evaluation.comparison import build_parametric_vs_knn_table
from src.evaluation.knn_baseline import KNNEmbeddingBaseline
from src.evaluation.metrics import (
    abstention_rate, age_cumulative_score, age_mae, age_median_absolute_error, age_r2, age_rmse,
    age_uncertainty_by_bucket, confidence_statistics, confusion_matrix,
    expected_calibration_error_intervals, gender_accuracy, gender_balanced_accuracy,
    gender_precision_recall_f1, gender_roc_auc, interval_coverage, mean_interval_width,
    median_interval_width, select_interval_examples,
)
from src.evaluation.predictions import export_predictions
from src.evaluation.selective import full_coverage_gender_report, gender_selective_prediction_report
from src.inference.artifacts import load_model_checkpoint
from src.utils.config import REPO_ROOT, resolve_device
from src.utils.io import checkpoint_experiment_name, file_sha256, save_json
from src.utils.logging import get_logger
from src.utils.provenance import dependency_versions, git_commit_sha
from src.utils.visualization import (
    plot_age_scatter, plot_confusion_matrix, plot_coverage_width_tradeoff, plot_error_histogram,
    plot_interval_coverage, plot_interval_width_by_bucket,
)

logger = get_logger("scripts.evaluate")


@torch.no_grad()
def run_inference(model, dataset, device, batch_size=64):
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    q10s, q50s, q90s, probs_all = [], [], [], []
    ages, age_masks, genders, gender_masks = [], [], [], []
    start = time.time()
    n_images = 0
    for batch in loader:
        images = batch["image"].to(device)
        outputs = model(images)
        q10s.append(outputs["age_output"]["q10"].cpu().numpy())
        q50s.append(outputs["age_output"]["q50"].cpu().numpy())
        q90s.append(outputs["age_output"]["q90"].cpu().numpy())
        probs_all.append(torch.softmax(outputs["gender_logits"], dim=-1).cpu().numpy())
        ages.append(batch["age"].numpy())
        age_masks.append(batch["age_mask"].numpy())
        genders.append(batch["gender_label"].numpy())
        gender_masks.append(batch["gender_mask"].numpy())
        n_images += len(images)
    elapsed = time.time() - start
    latency_ms_per_image = (elapsed / max(1, n_images)) * 1000.0

    # DataLoader(shuffle=False) iterates the dataset in index order 0..N-1,
    # so dataset.df's own row order (whatever "sample id" a caller wants --
    # here the image path, since there is no separate id column) lines up
    # exactly with every array above. Callers (paired bootstrap comparisons,
    # calibration provenance) rely on this to verify samples are actually
    # aligned across models/runs, not just equal in count.
    sample_id = dataset.df["image_path"].to_numpy() if hasattr(dataset, "df") else None

    return {
        "q10": np.concatenate(q10s), "q50": np.concatenate(q50s), "q90": np.concatenate(q90s),
        "probs": np.concatenate(probs_all), "age": np.concatenate(ages), "age_mask": np.concatenate(age_masks),
        "gender": np.concatenate(genders), "gender_mask": np.concatenate(gender_masks),
        "latency_ms_per_image": latency_ms_per_image, "sample_id": sample_id,
    }


def compute_parametric_metrics(preds: dict, confidence_threshold: float, calibration: dict | None) -> dict:
    age_mask = preds["age_mask"].astype(bool)
    gender_mask = preds["gender_mask"].astype(bool)

    metrics = {"latency_ms_per_image": preds["latency_ms_per_image"]}

    if age_mask.any():
        y_true = preds["age"][age_mask]
        q10, q50, q90 = preds["q10"][age_mask], preds["q50"][age_mask], preds["q90"][age_mask]
        metrics.update({
            "age_mae": age_mae(y_true, q50), "age_rmse": age_rmse(y_true, q50), "age_r2": age_r2(y_true, q50),
            "age_median_ae": age_median_absolute_error(y_true, q50),
            "age_cs5": age_cumulative_score(y_true, q50, threshold=5.0),
            "interval_coverage": interval_coverage(y_true, q10, q90),
            "mean_interval_width": mean_interval_width(q10, q90),
            "median_interval_width": median_interval_width(q10, q90),
            "calibration_error": expected_calibration_error_intervals(y_true, q10, q90, target_coverage=0.80),
            # Per-bucket MAE/coverage/width -- a single global coverage number
            # can hide age ranges where the model over/under-covers.
            "age_metrics_by_bucket": age_uncertainty_by_bucket(y_true, q10, q50, q90),
        })
        if calibration is not None:
            q10_cal, q90_cal = apply_conformal_offset(q10, q90, calibration["offset"])
            metrics["interval_coverage_calibrated"] = interval_coverage(y_true, q10_cal, q90_cal)
            metrics["mean_interval_width_calibrated"] = mean_interval_width(q10_cal, q90_cal)
            metrics["age_metrics_by_bucket_calibrated"] = age_uncertainty_by_bucket(y_true, q10_cal, q50, q90_cal)

    if gender_mask.any():
        probs = preds["probs"][gender_mask]
        y_true_gender = preds["gender"][gender_mask].astype(int)
        predicted = probs.argmax(axis=1)
        confidence = probs.max(axis=1)
        abstain = confidence < confidence_threshold
        prf = gender_precision_recall_f1(y_true_gender, predicted)
        metrics.update({
            "gender_accuracy": gender_accuracy(y_true_gender, predicted, abstain),
            "abstention_rate": abstention_rate(abstain),
            "confidence_stats": confidence_statistics(confidence),
            # Standard classification metrics on the raw argmax prediction
            # (not abstention-filtered) -- distinct from the selective
            # "gender_accuracy" above, which only scores accepted predictions.
            "gender_balanced_accuracy": gender_balanced_accuracy(y_true_gender, predicted),
            "gender_precision": prf["precision"],
            "gender_recall": prf["recall"],
            "gender_f1": prf["f1"],
            "gender_roc_auc": gender_roc_auc(y_true_gender, probs[:, 1]),
        })
        # Full selective-prediction report (raw argmax accuracy, effective
        # accuracy, risk-coverage curve, AURC) at the configured confidence
        # threshold -- evaluation-only, from these exact probabilities, never
        # a retrained model. See src/evaluation/selective.py.
        metrics["gender_selective_report"] = gender_selective_prediction_report(
            y_true_gender, probs, confidence_threshold=confidence_threshold,
        )
        # The "no abstention" point: the identical checkpoint/probabilities
        # at confidence_threshold=0.0 (every prediction accepted), never a
        # separate ablation_no_abstention training run.
        metrics["gender_full_coverage_report"] = full_coverage_gender_report(y_true_gender, probs)

    return metrics


def evaluate_checkpoint(
    checkpoint_path: str,
    output_name: str = "test_evaluation",
    compare_knn: bool = False,
    knn_path: str | None = None,
    calibration_dir: str | None = None,
) -> dict | None:
    """Evaluate a checkpoint on the test split, save metrics/plots, and return the metrics dict.

    This is the callable core used both by this script's CLI and by
    ``scripts/run_experiments.py`` (which calls it right after training each
    architecture-ablation experiment so the resulting
    ``outputs/metrics/{experiment}_test_metrics.json`` can be picked up by
    ``scripts/generate_architecture_report.py``'s ablation table). Returns
    None (after logging an error) if no prepared split exists yet.
    """
    device = resolve_device("auto")
    model, config, _ = load_model_checkpoint(checkpoint_path, device)

    splits_path = REPO_ROOT / config["paths"]["splits_dir"] / "full_metadata_with_splits.csv"
    if not splits_path.exists():
        logger.error("No prepared split found at %s.", splits_path)
        return None
    df = pd.read_csv(splits_path)
    test_df = df[df["split"] == "test"]
    # A model that declares its own preprocessing (pretrained-ResNet,
    # resolved from its pretrained backbone's own config) is evaluated with
    # that transform instead of this project's 128px/IMAGENET-constant
    # default -- every core model has no such method, so this is a no-op
    # for them. See src/data/transforms.py::resolve_eval_transform, the
    # single place this resolution logic lives (also used by
    # scripts/calibrate.py, scripts/run_robustness.py,
    # scripts/build_knn_index.py, and src/inference/predictor.py) --
    # what makes the evaluation path byte-identical for a core checkpoint
    # and a transfer-learning checkpoint alike.
    eval_transform = resolve_eval_transform(model, config)
    dataset = FaceMultiTaskDataset(test_df, eval_transform)

    preds = run_inference(model, dataset, device)

    # No silent fallback to a shared global outputs/calibration directory:
    # a caller that doesn't know/pass an isolated calibration_dir gets no
    # calibration applied (raw metrics only) rather than possibly picking
    # up some other experiment/seed's leftover conformal offset.
    calibration = load_calibration(calibration_dir) if calibration_dir else None
    if calibration is not None:
        validate_calibration_artifact(
            calibration,
            checkpoint_path=checkpoint_path,
            split_csv_path=splits_path,
            test_sample_ids=test_df["image_path"].tolist(),
        )
    elif calibration_dir is None:
        logger.info("No calibration_dir given; evaluating with raw (uncalibrated) intervals only.")

    confidence_threshold = config["model"]["gender_head"].get("confidence_threshold", 0.80)
    metrics = compute_parametric_metrics(preds, confidence_threshold, calibration)

    output_dir = REPO_ROOT / config["paths"]["output_dir"]
    metrics_dir, plots_dir = output_dir / "metrics", output_dir / "plots"
    predictions_dir = output_dir / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Per-sample prediction export: the EXACT same `preds` arrays already
    # used for the aggregate metrics above -- never a second inference pass
    # -- so a later re-analysis or paired statistical comparison never needs
    # to re-run the model, and the exported file is guaranteed to reproduce
    # the reported numbers (see tests/test_predictions_export.py).
    export_predictions(
        preds, predictions_dir / f"{output_name}_predictions.csv", split="test", calibration=calibration,
        confidence_threshold=confidence_threshold,
        manifest_path=predictions_dir / f"{output_name}_predictions_manifest.json",
        provenance={
            "experiment": checkpoint_experiment_name(checkpoint_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "split_path": str(splits_path),
            "split_sha256": file_sha256(splits_path),
            "calibration_artifact_sha256": (
                file_sha256(Path(calibration_dir) / "conformal_calibration.json")
                if calibration is not None and calibration_dir else None
            ),
            "model_family": config["model"].get("family", "core"),
            "backbone_identifier": getattr(model, "model_id", config["model"].get("backbone", {}).get("name")),
            "pretrained_source": getattr(model, "pretrained_source", None),
            "input_size": getattr(model, "input_size", config["dataset"]["image_size"]),
            "preprocessing": {
                "image_size": eval_transform.image_size, "mean": list(eval_transform.mean),
                "std": list(eval_transform.std), "interpolation": eval_transform.interpolation,
                "crop_pct": getattr(eval_transform, "crop_pct", 1.0),
            },
            "git_commit_sha": git_commit_sha(),
            "dependency_versions": dependency_versions(),
        },
    )

    age_mask = preds["age_mask"].astype(bool)
    if age_mask.any():
        y_true, q10, q50, q90 = (
            preds["age"][age_mask], preds["q10"][age_mask], preds["q50"][age_mask], preds["q90"][age_mask],
        )
        plot_age_scatter(y_true, q50, plots_dir / f"{output_name}_age_scatter.png")
        plot_error_histogram(y_true - q50, plots_dir / f"{output_name}_age_error_hist.png")

        bucket_report = metrics["age_metrics_by_bucket"]
        labels = [k for k, v in bucket_report.items() if v["count"] > 0]
        if labels:
            coverage_by_bucket = np.array([bucket_report[k]["coverage"] for k in labels])
            width_by_bucket = np.array([bucket_report[k]["mean_width"] for k in labels])
            plot_interval_coverage(labels, coverage_by_bucket, 0.80, plots_dir / f"{output_name}_interval_coverage.png")
            plot_interval_width_by_bucket(labels, width_by_bucket, plots_dir / f"{output_name}_interval_width_by_bucket.png")

        if calibration is not None and "age_metrics_by_bucket_calibrated" in metrics:
            # The raw q10-q90 interval is nominally an 80% interval by
            # construction of the quantile head, independent of calibration --
            # but the *calibrated* interval's target is whatever the
            # calibration artifact was actually fit for (1 - alpha), which
            # need not be 0.80 (e.g. alpha=0.10 -> 0.90). Using a hardcoded
            # 0.80 here would silently mislabel the target line whenever
            # configs/training.yaml's calibration.alpha differs from 0.10.
            plot_coverage_width_tradeoff(
                coverage_before=metrics["interval_coverage"],
                width_before=metrics["mean_interval_width"],
                coverage_after=metrics["interval_coverage_calibrated"],
                width_after=metrics["mean_interval_width_calibrated"],
                target_coverage=calibration.get("target_coverage", 0.80),
                out_path=plots_dir / f"{output_name}_coverage_width_tradeoff.png",
            )

        if "image_path" in test_df.columns:
            image_paths = test_df["image_path"].to_numpy()[age_mask]
            metrics["interval_examples"] = select_interval_examples(image_paths, y_true, q10, q50, q90)

    gender_mask = preds["gender_mask"].astype(bool)
    if gender_mask.any():
        y_true_gender = preds["gender"][gender_mask].astype(int)
        predicted = preds["probs"][gender_mask].argmax(axis=1)
        cm = confusion_matrix(y_true_gender, predicted, num_classes=config["model"]["gender_head"]["num_classes"])
        plot_confusion_matrix(cm, config["model"]["gender_head"]["class_names"], plots_dir / f"{output_name}_confusion_matrix.png")

    if compare_knn:
        resolved_knn_path = Path(knn_path or REPO_ROOT / "outputs" / "knn" / "knn_baseline.pkl")
        if not resolved_knn_path.exists():
            logger.warning("No k-NN index at %s; run 'make build-knn' first.", resolved_knn_path)
        else:
            knn = KNNEmbeddingBaseline.load(resolved_knn_path)
            knn_metrics = _evaluate_knn(model, dataset, device, knn, confidence_threshold)
            table = build_parametric_vs_knn_table(metrics, knn_metrics)
            # Saved under this checkpoint's own output_dir (isolated per
            # experiment/seed), never a single shared global path -- two
            # experiments evaluated with --compare-knn must not overwrite
            # each other's comparison table.
            knn_dir = output_dir / "knn"
            knn_dir.mkdir(parents=True, exist_ok=True)
            knn_table_path = knn_dir / f"{output_name}_parametric_vs_knn.csv"
            table.to_csv(knn_table_path, index=False)
            metrics["knn_comparison_table_path"] = str(knn_table_path)
            logger.info("Saved parametric-vs-kNN comparison table to %s", knn_table_path)

    save_json(metrics, metrics_dir / f"{output_name}.json")
    logger.info("Evaluation metrics: %s", {k: v for k, v in metrics.items() if not isinstance(v, dict)})
    return metrics


def _default_output_name(checkpoint_path: str) -> str:
    """Derive '{experiment}_test_metrics' from a checkpoint filename.

    E.g. "exp_c_shared_adapters_best_balanced_score.pt" -> "exp_c_shared_adapters_test_metrics".
    This is what lets scripts/generate_architecture_report.py's ablation
    table pick up real performance numbers for *any* evaluated checkpoint,
    not just ones evaluated via scripts/run_experiments.py, without the
    caller having to remember to pass --output-name.
    """
    return checkpoint_experiment_name(checkpoint_path) + "_test_metrics"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--compare-knn", action="store_true")
    parser.add_argument("--knn-path", default=str(REPO_ROOT / "outputs" / "knn" / "knn_baseline.pkl"))
    parser.add_argument("--calibration-dir", default=str(REPO_ROOT / "outputs" / "calibration"))
    parser.add_argument("--output-name", default=None, help="Default: derived from the checkpoint filename")
    args = parser.parse_args()

    output_name = args.output_name or _default_output_name(args.checkpoint)
    metrics = evaluate_checkpoint(
        args.checkpoint, output_name, args.compare_knn, args.knn_path, args.calibration_dir
    )
    return 0 if metrics is not None else 1


@torch.no_grad()
def _evaluate_knn(model, dataset, device, knn: KNNEmbeddingBaseline, confidence_threshold: float) -> dict:
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    age_embeds, gender_embeds = [], []
    ages, age_masks, genders, gender_masks = [], [], [], []
    start = time.time()
    n = 0
    for batch in loader:
        images = batch["image"].to(device)
        emb = model.encode(images)
        age_embeds.append(emb["age_embedding"].cpu().numpy())
        gender_embeds.append(emb["gender_embedding"].cpu().numpy())
        ages.append(batch["age"].numpy())
        age_masks.append(batch["age_mask"].numpy())
        genders.append(batch["gender_label"].numpy())
        gender_masks.append(batch["gender_mask"].numpy())
        n += len(images)
    latency_ms_per_image = (time.time() - start) / max(1, n) * 1000.0

    age_embeds = np.concatenate(age_embeds)
    gender_embeds = np.concatenate(gender_embeds)
    ages, age_masks = np.concatenate(ages), np.concatenate(age_masks).astype(bool)
    genders, gender_masks = np.concatenate(genders), np.concatenate(gender_masks).astype(bool)

    metrics = {"latency_ms_per_image": latency_ms_per_image}
    if age_masks.any():
        result = knn.predict_age(age_embeds[age_masks])
        y_true = ages[age_masks]
        metrics.update({
            "age_mae": age_mae(y_true, result.q50), "age_rmse": age_rmse(y_true, result.q50),
            "interval_coverage": interval_coverage(y_true, result.q10, result.q90),
            "mean_interval_width": mean_interval_width(result.q10, result.q90),
        })
    if gender_masks.any():
        result = knn.predict_gender(gender_embeds[gender_masks], confidence_threshold)
        y_true_gender = genders[gender_masks].astype(int)
        metrics.update({
            "gender_accuracy": gender_accuracy(y_true_gender, result.predicted_class, result.abstain),
            "abstention_rate": abstention_rate(result.abstain),
            "mean_confidence": float(result.confidence.mean()),
        })
    return metrics


if __name__ == "__main__":
    raise SystemExit(main())
