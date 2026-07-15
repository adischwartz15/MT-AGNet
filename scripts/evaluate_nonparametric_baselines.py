#!/usr/bin/env python
"""CLI: final TEST-set evaluation of the tuned non-parametric baselines
(final-run hardening T4), using the hyperparameters
``scripts/tune_nonparametric_baselines.py`` already selected on validation.

Never re-tunes anything -- loads ``outputs/nonparametric/best_params.json``
and the pickled, already-train-fit feature pipelines (scaler + PCA) as-is.
Rejects the loaded artifacts if their recorded split hash doesn't match the
currently locked split.

For age, additionally fits split-conformal calibration for the k-NN
interval baseline using the CALIBRATION split only (never validation or
test), then reports raw vs. calibrated coverage/width on test -- the same
protocol and the same ``src/evaluation/calibration.py`` functions the deep
model uses, so the comparison table is apples-to-apples.

Usage:
    python scripts/evaluate_nonparametric_baselines.py
    python scripts/evaluate_nonparametric_baselines.py --feature-sources raw_pca
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors  # noqa: E402

from src.evaluation.calibration import (  # noqa: E402
    compute_nonconformity_scores, evaluate_calibration_effect, fit_conformal_offset,
)
from src.evaluation.knn_baseline import weighted_quantile  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    gender_balanced_accuracy, gender_precision_recall_f1, gender_roc_auc,
)
from src.evaluation.nonparametric.features import FEATURE_EXTRACTORS  # noqa: E402
from src.evaluation.nonparametric.kernels import ClassConditionalKDEClassifier, NadarayaWatsonRegressor  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, parse_cli_overrides  # noqa: E402
from src.utils.io import file_sha256, load_json, save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("scripts.evaluate_nonparametric_baselines")

OUTPUT_DIR = REPO_ROOT / "outputs" / "nonparametric"


def _load_pipeline(feature_source: str, label: str):
    path = OUTPUT_DIR / f"{feature_source}_{label}_pipeline.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _knn_age_intervals(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, k: int, metric: str):
    k_eff = min(k, len(X_train))
    nn = NearestNeighbors(n_neighbors=k_eff, metric=metric).fit(X_train)
    distances, indices = nn.kneighbors(X_query)
    weights = 1.0 / (distances + 1e-8)
    q10s, q50s, q90s = [], [], []
    for i in range(len(X_query)):
        neighbor_ages = y_train[indices[i]]
        w = weights[i] / weights[i].sum()
        q10s.append(weighted_quantile(neighbor_ages, w, 0.10))
        q50s.append(weighted_quantile(neighbor_ages, w, 0.50))
        q90s.append(weighted_quantile(neighbor_ages, w, 0.90))
    return np.array(q10s), np.array(q50s), np.array(q90s)


def evaluate_feature_source(
    feature_source: str, best: dict, df: pd.DataFrame, backbone_model_id: str, backbone_pretrained: bool,
) -> dict:
    extractor = FEATURE_EXTRACTORS[feature_source]
    extractor_kwargs = {"model_id": backbone_model_id, "pretrained": backbone_pretrained} if feature_source == "frozen_backbone" else {}

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    calibration_df = df[df["split"] == "calibration"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    results: dict = {"feature_source": feature_source}

    age_best = best.get("age_knn")
    if age_best is not None:
        pipeline = _load_pipeline(feature_source, "age_knn")
        X_train_raw, _ = extractor(train_df, **extractor_kwargs)
        X_cal_raw, _ = extractor(calibration_df, **extractor_kwargs)
        X_test_raw, _ = extractor(test_df, **extractor_kwargs)
        age_train_mask = train_df["age"].notna().to_numpy()
        age_cal_mask = calibration_df["age"].notna().to_numpy()
        age_test_mask = test_df["age"].notna().to_numpy()

        X_train = pipeline.transform(X_train_raw)[age_train_mask]
        y_train = train_df["age"].to_numpy(dtype=float)[age_train_mask]
        X_cal = pipeline.transform(X_cal_raw)[age_cal_mask]
        y_cal = calibration_df["age"].to_numpy(dtype=float)[age_cal_mask]
        X_test = pipeline.transform(X_test_raw)[age_test_mask]
        y_test = test_df["age"].to_numpy(dtype=float)[age_test_mask]

        k, metric = age_best["k_effective"], age_best["metric"]
        model = KNeighborsRegressor(n_neighbors=k, metric=metric, weights="distance").fit(X_train, y_train)
        preds_test = model.predict(X_test)
        mae = float(np.mean(np.abs(preds_test - y_test)))

        q10_cal, q50_cal, q90_cal = _knn_age_intervals(X_train, y_train, X_cal, k, metric)
        scores = compute_nonconformity_scores(y_cal, q10_cal, q90_cal)
        offset = fit_conformal_offset(scores, alpha=0.10)

        q10_test, q50_test, q90_test = _knn_age_intervals(X_train, y_train, X_test, k, metric)
        calibration_effect = evaluate_calibration_effect(y_test, q10_test, q90_test, offset)

        results["age_knn"] = {
            "mae": mae, "n_test": len(y_test), "hyperparameters": age_best,
            "conformal_offset": offset, "n_calibration": len(y_cal), **calibration_effect,
        }

    kernel_age_best = best.get("age_kernel")
    if kernel_age_best is not None:
        pipeline = _load_pipeline(feature_source, "age_kernel")
        X_train_raw, _ = extractor(train_df, **extractor_kwargs)
        X_test_raw, _ = extractor(test_df, **extractor_kwargs)
        age_train_mask = train_df["age"].notna().to_numpy()
        age_test_mask = test_df["age"].notna().to_numpy()
        X_train = pipeline.transform(X_train_raw)[age_train_mask]
        y_train = train_df["age"].to_numpy(dtype=float)[age_train_mask]
        X_test = pipeline.transform(X_test_raw)[age_test_mask]
        y_test = test_df["age"].to_numpy(dtype=float)[age_test_mask]

        model = NadarayaWatsonRegressor(bandwidth=kernel_age_best["bandwidth"]).fit(X_train, y_train)
        preds_test = model.predict(X_test)
        mae = float(np.mean(np.abs(preds_test - y_test)))
        results["age_kernel"] = {"mae": mae, "n_test": len(y_test), "hyperparameters": kernel_age_best}

    gender_best = best.get("gender_knn")
    if gender_best is not None:
        pipeline = _load_pipeline(feature_source, "gender_knn")
        X_train_raw, _ = extractor(train_df, **extractor_kwargs)
        X_test_raw, _ = extractor(test_df, **extractor_kwargs)
        gender_train_mask = train_df["gender_label"].notna().to_numpy()
        gender_test_mask = test_df["gender_label"].notna().to_numpy()
        X_train = pipeline.transform(X_train_raw)[gender_train_mask]
        y_train = train_df["gender_label"].to_numpy(dtype=int)[gender_train_mask]
        X_test = pipeline.transform(X_test_raw)[gender_test_mask]
        y_test = test_df["gender_label"].to_numpy(dtype=int)[gender_test_mask]

        k, metric = gender_best["k_effective"], gender_best["metric"]
        model = KNeighborsClassifier(n_neighbors=k, metric=metric, weights="distance").fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        prf = gender_precision_recall_f1(y_test, preds)
        results["gender_knn"] = {
            "accuracy": float(np.mean(preds == y_test)), "balanced_accuracy": gender_balanced_accuracy(y_test, preds),
            "f1": prf["f1"], "precision": prf["precision"], "recall": prf["recall"],
            "roc_auc": gender_roc_auc(y_test, probs[:, 1]) if probs.shape[1] > 1 else None,
            "n_test": len(y_test), "hyperparameters": gender_best,
        }

    kernel_gender_best = best.get("gender_kernel")
    if kernel_gender_best is not None:
        pipeline = _load_pipeline(feature_source, "gender_kernel")
        X_train_raw, _ = extractor(train_df, **extractor_kwargs)
        X_test_raw, _ = extractor(test_df, **extractor_kwargs)
        gender_train_mask = train_df["gender_label"].notna().to_numpy()
        gender_test_mask = test_df["gender_label"].notna().to_numpy()
        X_train = pipeline.transform(X_train_raw)[gender_train_mask]
        y_train = train_df["gender_label"].to_numpy(dtype=int)[gender_train_mask]
        X_test = pipeline.transform(X_test_raw)[gender_test_mask]
        y_test = test_df["gender_label"].to_numpy(dtype=int)[gender_test_mask]

        model = ClassConditionalKDEClassifier(bw_method=kernel_gender_best["bw_method"]).fit(X_train, y_train)
        preds = model.predict(X_test)
        prf = gender_precision_recall_f1(y_test, preds)
        results["gender_kernel"] = {
            "accuracy": float(np.mean(preds == y_test)), "balanced_accuracy": gender_balanced_accuracy(y_test, preds),
            "f1": prf["f1"], "n_test": len(y_test), "hyperparameters": kernel_gender_best,
        }

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--feature-sources", nargs="+", choices=["raw_pca", "frozen_backbone"], default=None)
    parser.add_argument("--backbone-model-id", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--offline-smoke", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key.path=value overrides")
    args = parser.parse_args()

    best_params_path = OUTPUT_DIR / "best_params.json"
    if not best_params_path.exists():
        logger.error("No tuned hyperparameters found at %s. Run scripts/tune_nonparametric_baselines.py first.", best_params_path)
        return 1
    best_params = load_json(best_params_path)

    config = load_config(
        REPO_ROOT / "configs" / "data.yaml", *([args.config] if args.config else []),
        overrides=parse_cli_overrides(args.overrides),
    )
    splits_path = REPO_ROOT / config["paths"]["splits_dir"] / "full_metadata_with_splits.csv"
    if not splits_path.exists():
        logger.error("No locked split found at %s.", splits_path)
        return 1
    current_split_sha256 = file_sha256(splits_path)
    if best_params.get("split_sha256") != current_split_sha256:
        logger.error(
            "Tuned hyperparameters were fit against a different split (recorded=%s, current=%s). "
            "Re-run scripts/tune_nonparametric_baselines.py against the current locked split before evaluating.",
            best_params.get("split_sha256"), current_split_sha256,
        )
        return 1

    df = pd.read_csv(splits_path)
    feature_sources = args.feature_sources or list(best_params["feature_sources"])

    all_results = {}
    for feature_source in feature_sources:
        if feature_source not in best_params["feature_sources"]:
            logger.warning("No tuned hyperparameters for feature_source=%s -- skipping.", feature_source)
            continue
        best = best_params["feature_sources"][feature_source]["best"]
        logger.info("Evaluating %s on the test split...", feature_source)
        all_results[feature_source] = evaluate_feature_source(
            feature_source, best, df, args.backbone_model_id, backbone_pretrained=not args.offline_smoke,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(
        {"split_sha256": current_split_sha256, "results": all_results}, OUTPUT_DIR / "test_results.json",
    )

    rows = []
    for feature_source, methods in all_results.items():
        for method_name, metrics in methods.items():
            if method_name == "feature_source":
                continue
            row = {"feature_source": feature_source, "method": method_name}
            row.update({k: v for k, v in metrics.items() if not isinstance(v, dict)})
            rows.append(row)
    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_DIR / "results.csv", index=False)

    print("\n=== Non-parametric baseline comparison (test split) ===")
    print(results_df.to_string(index=False))
    print(f"\nSaved detailed results to {OUTPUT_DIR / 'test_results.json'} and {OUTPUT_DIR / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
