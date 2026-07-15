#!/usr/bin/env python
"""CLI: validation-only hyperparameter grid search for the two non-parametric
baseline pipelines (final-run hardening T4):

    Pipeline 1 (raw_pca):         flattened raw pixels -> train-only
                                   StandardScaler -> train-only PCA ->
                                   optional L2-norm -> k-NN / kernel method.
    Pipeline 2 (frozen_backbone): frozen ImageNet-pretrained backbone
                                   features (adapters/heads never touched,
                                   never fine-tuned) -> the same
                                   scaler/PCA/L2-norm/k-NN/kernel protocol.

For each pipeline, tunes BOTH a k-NN baseline and a kernel-based baseline
(Nadaraya-Watson for age, class-conditional KDE for gender) across the PCA
dimensionality / L2-normalization / k-or-bandwidth grid, selecting every
hyperparameter using VALIDATION predictions only (never test, never
calibration). Saves every candidate configuration (not just the winner) plus
the fitted feature pipeline and full provenance (split hash, feature source,
selection objective) to ``outputs/nonparametric/``.

This script never touches the test or calibration splits, and never trains
or fine-tunes any neural network -- see
``src/evaluation/nonparametric/features.py`` (frozen backbone only) and
``src/evaluation/nonparametric/pipeline.py`` (train-only fitting,
validation-only selection).

Usage:
    python scripts/tune_nonparametric_baselines.py
    python scripts/tune_nonparametric_baselines.py --feature-sources raw_pca
    python scripts/tune_nonparametric_baselines.py --max-train-samples 2000  # faster grid search
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.evaluation.nonparametric.features import FEATURE_EXTRACTORS  # noqa: E402
from src.evaluation.nonparametric.pipeline import (  # noqa: E402
    K_VALUES, KDE_BANDWIDTH_SCALES, L2_NORMALIZATION, PCA_COMPONENTS_BACKBONE, PCA_COMPONENTS_RAW,
    fit_feature_pipeline, tune_kde_gender, tune_kernel_regression_age, tune_knn_age, tune_knn_gender,
)
from src.utils.config import REPO_ROOT, load_config, parse_cli_overrides  # noqa: E402
from src.utils.io import file_sha256, save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.provenance import dependency_versions, git_commit_sha  # noqa: E402

logger = get_logger("scripts.tune_nonparametric_baselines")

OUTPUT_DIR = REPO_ROOT / "outputs" / "nonparametric"
_PCA_GRIDS = {"raw_pca": PCA_COMPONENTS_RAW, "frozen_backbone": PCA_COMPONENTS_BACKBONE}


def _atomic_save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    save_json(data, tmp_path)
    os.replace(tmp_path, path)


def _atomic_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp_path, path)


def _subsample(df: pd.DataFrame, max_samples: int | None, seed: int = 42) -> pd.DataFrame:
    if max_samples is None or len(df) <= max_samples:
        return df
    return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)


def tune_one_feature_source(
    feature_source: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
    backbone_model_id: str = "resnet18", backbone_pretrained: bool = True,
    k_values: list[int] = K_VALUES, distance_metrics: list[str] = None,
    l2_options: list[bool] = L2_NORMALIZATION, kde_bandwidth_scales: list[float] = KDE_BANDWIDTH_SCALES,
) -> dict:
    """Full grid search for one feature source. Returns a dict with every
    candidate config tried (for both k-NN and kernel methods, age and
    gender) plus the selected best configuration for each."""
    distance_metrics = distance_metrics if distance_metrics is not None else ["euclidean", "cosine"]
    extractor = FEATURE_EXTRACTORS[feature_source]
    logger.info("[%s] extracting features (train=%d, val=%d)...", feature_source, len(train_df), len(val_df))
    if feature_source == "frozen_backbone":
        X_train_raw, _ = extractor(train_df, model_id=backbone_model_id, pretrained=backbone_pretrained)
        X_val_raw, _ = extractor(val_df, model_id=backbone_model_id, pretrained=backbone_pretrained)
    else:
        X_train_raw, _ = extractor(train_df)
        X_val_raw, _ = extractor(val_df)

    age_train_mask = train_df["age"].notna().to_numpy()
    age_val_mask = val_df["age"].notna().to_numpy()
    gender_train_mask = train_df["gender_label"].notna().to_numpy()
    gender_val_mask = val_df["gender_label"].notna().to_numpy()

    y_age_train, y_age_val = train_df["age"].to_numpy(dtype=float), val_df["age"].to_numpy(dtype=float)
    y_gender_train, y_gender_val = train_df["gender_label"].to_numpy(dtype=int), val_df["gender_label"].to_numpy(dtype=int)

    pca_grid = _PCA_GRIDS[feature_source]
    all_age_knn_candidates, all_gender_knn_candidates = [], []
    all_age_kernel_candidates, all_gender_kernel_candidates = [], []
    best_age_knn, best_gender_knn, best_age_kernel, best_gender_kernel = None, None, None, None

    for n_components in pca_grid:
        for l2_normalize in l2_options:
            pipeline = fit_feature_pipeline(X_train_raw, feature_source, n_components=n_components, l2_normalize=l2_normalize)
            X_train = pipeline.transform(X_train_raw)
            X_val = pipeline.transform(X_val_raw)
            provenance = pipeline.provenance()

            if age_train_mask.any() and age_val_mask.any():
                candidate, all_candidates = tune_knn_age(
                    X_train[age_train_mask], y_age_train[age_train_mask],
                    X_val[age_val_mask], y_age_val[age_val_mask], k_values=k_values, metrics=distance_metrics,
                )
                for c in all_candidates:
                    all_age_knn_candidates.append({**c, **provenance})
                candidate = {**candidate, **provenance}
                if best_age_knn is None or candidate["val_mae"] < best_age_knn["val_mae"]:
                    best_age_knn = candidate

                candidate_k, all_k_candidates = tune_kernel_regression_age(
                    X_train[age_train_mask], y_age_train[age_train_mask],
                    X_val[age_val_mask], y_age_val[age_val_mask], bandwidth_scales=kde_bandwidth_scales,
                )
                for c in all_k_candidates:
                    all_age_kernel_candidates.append({**c, **provenance})
                candidate_k = {**candidate_k, **provenance}
                if best_age_kernel is None or candidate_k["val_mae"] < best_age_kernel["val_mae"]:
                    best_age_kernel = candidate_k

            if gender_train_mask.any() and gender_val_mask.any():
                candidate, all_candidates = tune_knn_gender(
                    X_train[gender_train_mask], y_gender_train[gender_train_mask],
                    X_val[gender_val_mask], y_gender_val[gender_val_mask], k_values=k_values, metrics=distance_metrics,
                )
                for c in all_candidates:
                    all_gender_knn_candidates.append({**c, **provenance})
                candidate = {**candidate, **provenance}
                if best_gender_knn is None or candidate["val_balanced_accuracy"] > best_gender_knn["val_balanced_accuracy"]:
                    best_gender_knn = candidate

                candidate_k, all_k_candidates = tune_kde_gender(
                    X_train[gender_train_mask], y_gender_train[gender_train_mask],
                    X_val[gender_val_mask], y_gender_val[gender_val_mask],
                )
                for c in all_k_candidates:
                    all_gender_kernel_candidates.append({**c, **provenance})
                candidate_k = {**candidate_k, **provenance}
                if best_gender_kernel is None or candidate_k["val_balanced_accuracy"] > best_gender_kernel["val_balanced_accuracy"]:
                    best_gender_kernel = candidate_k

    # Persist the winning fitted feature pipeline (scaler + PCA) for each
    # method/task, so scripts/evaluate_nonparametric_baselines.py never
    # re-fits on anything -- it only ever calls .transform().
    fitted_pipelines = {}
    for label, best in (("age_knn", best_age_knn), ("gender_knn", best_gender_knn),
                         ("age_kernel", best_age_kernel), ("gender_kernel", best_gender_kernel)):
        if best is not None:
            fitted_pipelines[label] = fit_feature_pipeline(
                X_train_raw, feature_source, n_components=best["n_components_used"], l2_normalize=best["l2_normalize"],
            )

    return {
        "feature_source": feature_source,
        "best": {
            "age_knn": best_age_knn, "gender_knn": best_gender_knn,
            "age_kernel": best_age_kernel, "gender_kernel": best_gender_kernel,
        },
        "candidates": {
            "age_knn": all_age_knn_candidates, "gender_knn": all_gender_knn_candidates,
            "age_kernel": all_age_kernel_candidates, "gender_kernel": all_gender_kernel_candidates,
        },
        "fitted_pipelines": fitted_pipelines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Extra YAML config to merge on top of configs/data.yaml")
    parser.add_argument(
        "--feature-sources", nargs="+", choices=["raw_pca", "frozen_backbone"], default=["raw_pca", "frozen_backbone"],
    )
    parser.add_argument("--backbone-model-id", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--max-train-samples", type=int, default=None, help="Subsample train for a faster grid search")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument(
        "--offline-smoke", action="store_true",
        help="pretrained=False for the frozen-backbone pipeline (no network) -- non-scientific, integration-check only",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key.path=value overrides")
    args = parser.parse_args()

    config = load_config(
        REPO_ROOT / "configs" / "data.yaml", *([args.config] if args.config else []),
        overrides=parse_cli_overrides(args.overrides),
    )
    splits_path = REPO_ROOT / config["paths"]["splits_dir"] / "full_metadata_with_splits.csv"
    if not splits_path.exists():
        logger.error("No locked split found at %s. Run scripts/lock_split.py first.", splits_path)
        return 1
    split_sha256 = file_sha256(splits_path)

    df = pd.read_csv(splits_path)
    train_df = _subsample(df[df["split"] == "train"].reset_index(drop=True), args.max_train_samples)
    val_df = _subsample(df[df["split"] == "validation"].reset_index(drop=True), args.max_val_samples)
    if len(train_df) == 0 or len(val_df) == 0:
        logger.error("Train or validation split is empty -- cannot tune.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for feature_source in args.feature_sources:
        results[feature_source] = tune_one_feature_source(
            feature_source, train_df, val_df, backbone_model_id=args.backbone_model_id,
            backbone_pretrained=not args.offline_smoke,
        )
        # Save the fitted pipelines for this feature source immediately (large objects).
        for label, pipeline in results[feature_source]["fitted_pipelines"].items():
            _atomic_pickle(pipeline, OUTPUT_DIR / f"{feature_source}_{label}_pipeline.pkl")

    best_params = {
        "split_sha256": split_sha256,
        "split_path": str(splits_path),
        "n_train": len(train_df), "n_val": len(val_df),
        "offline_smoke": args.offline_smoke,
        "git_commit_sha": git_commit_sha(),
        "dependency_versions": dependency_versions(),
        "feature_sources": {
            source: {"best": r["best"], "n_candidates": {k: len(v) for k, v in r["candidates"].items()}}
            for source, r in results.items()
        },
    }
    _atomic_save_json(best_params, OUTPUT_DIR / "best_params.json")

    all_candidates_rows = []
    for feature_source, r in results.items():
        for method, candidates in r["candidates"].items():
            for c in candidates:
                all_candidates_rows.append({"feature_source": feature_source, "method": method, **c})
    if all_candidates_rows:
        import pandas as _pd

        _pd.DataFrame(all_candidates_rows).to_csv(OUTPUT_DIR / "all_candidates.csv", index=False)

    logger.info("Wrote %s (best configs) and all_candidates.csv (%d rows) to %s", "best_params.json", len(all_candidates_rows), OUTPUT_DIR)
    print(f"Non-parametric baseline tuning complete. See {OUTPUT_DIR}/best_params.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
