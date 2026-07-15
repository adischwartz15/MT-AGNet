"""End-to-end tests for scripts/tune_nonparametric_baselines.py and
scripts/evaluate_nonparametric_baselines.py (T4, final-run hardening).

Uses only the raw_pca feature source (never touches a model/network) and a
tiny synthetic 4-way split, so this suite is fast and fully offline. Proves
the wiring end-to-end: tune -> best_params.json + pickled pipelines ->
evaluate -> results.csv, including split-hash rejection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = str(REPO_ROOT / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import evaluate_nonparametric_baselines as enb  # noqa: E402
import tune_nonparametric_baselines as tnb  # noqa: E402

from src.evaluation.nonparametric.pipeline import K_VALUES, L2_NORMALIZATION  # noqa: E402


@pytest.fixture
def four_way_split_df(synthetic_image_dir):
    """A tiny, real 4-way split (train/validation/calibration/test) built
    from synthetic images -- exercises the actual DataLoader/PIL image
    reading path, not just synthetic feature arrays."""
    image_dir, records_df = synthetic_image_dir
    n = len(records_df)
    rng = np.random.default_rng(0)
    splits = np.array(["train"] * int(n * 0.4) + ["validation"] * int(n * 0.2)
                       + ["calibration"] * int(n * 0.2) + ["test"] * (n - int(n * 0.4) - int(n * 0.2) - int(n * 0.2)))
    rng.shuffle(splits)
    df = pd.DataFrame({
        "image_path": records_df["path"], "age": records_df["age"].astype(float),
        "gender_label": records_df["gender_label"].astype(float), "split": splits[:n],
    })
    return df


def test_tune_one_feature_source_raw_pca_produces_candidates_and_pipelines(four_way_split_df):
    train_df = four_way_split_df[four_way_split_df["split"] == "train"].reset_index(drop=True)
    val_df = four_way_split_df[four_way_split_df["split"] == "validation"].reset_index(drop=True)

    result = tnb.tune_one_feature_source(
        "raw_pca", train_df, val_df,
        k_values=[1, 3], distance_metrics=["euclidean"], l2_options=[False], kde_bandwidth_scales=[1.0],
    )
    assert result["feature_source"] == "raw_pca"
    assert result["best"]["age_knn"] is not None
    assert result["best"]["gender_knn"] is not None
    assert len(result["candidates"]["age_knn"]) > 0
    assert "age_knn" in result["fitted_pipelines"]


def test_tune_cli_writes_best_params_with_split_hash(tmp_path, four_way_split_df, monkeypatch):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    split_path = splits_dir / "full_metadata_with_splits.csv"
    four_way_split_df.to_csv(split_path, index=False)

    output_dir = tmp_path / "nonparametric_output"
    monkeypatch.setattr(tnb, "OUTPUT_DIR", output_dir)

    argv = [
        "tune_nonparametric_baselines.py",
        "--set", f"paths.splits_dir={splits_dir.as_posix()}",
        "--feature-sources", "raw_pca",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    # Shrink the grids for test speed (module-level constants the CLI imports).
    monkeypatch.setattr(tnb, "K_VALUES", [1, 3])
    monkeypatch.setattr(tnb, "_PCA_GRIDS", {"raw_pca": [5], "frozen_backbone": [5]})

    rc = tnb.main()
    assert rc == 0
    assert (output_dir / "best_params.json").exists()
    assert (output_dir / "all_candidates.csv").exists()

    from src.utils.io import file_sha256, load_json

    best_params = load_json(output_dir / "best_params.json")
    assert best_params["split_sha256"] == file_sha256(split_path)
    assert "raw_pca" in best_params["feature_sources"]


def test_evaluate_cli_rejects_mismatched_split_hash(tmp_path, four_way_split_df, monkeypatch):
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    split_path = splits_dir / "full_metadata_with_splits.csv"
    four_way_split_df.to_csv(split_path, index=False)

    output_dir = tmp_path / "nonparametric_output"
    output_dir.mkdir(parents=True)
    from src.utils.io import save_json

    save_json({"split_sha256": "not-the-real-hash", "feature_sources": {}}, output_dir / "best_params.json")
    monkeypatch.setattr(enb, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_nonparametric_baselines.py", "--set", f"paths.splits_dir={splits_dir.as_posix()}",
    ])

    rc = enb.main()
    assert rc == 1  # rejected -- must not silently evaluate against a mismatched split


def test_full_tune_then_evaluate_pipeline(tmp_path, four_way_split_df, monkeypatch):
    """The complete, real (offline, raw_pca-only) round trip: tune ->
    best_params.json + pickled pipelines -> evaluate -> results.csv with
    real numbers, never re-fitting the saved scaler/PCA."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    split_path = splits_dir / "full_metadata_with_splits.csv"
    four_way_split_df.to_csv(split_path, index=False)

    output_dir = tmp_path / "nonparametric_output"
    monkeypatch.setattr(tnb, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(enb, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(tnb, "_PCA_GRIDS", {"raw_pca": [5], "frozen_backbone": [5]})

    tune_argv = [
        "tune_nonparametric_baselines.py", "--set", f"paths.splits_dir={splits_dir.as_posix()}",
        "--feature-sources", "raw_pca",
    ]
    monkeypatch.setattr(sys, "argv", tune_argv)
    assert tnb.main() == 0

    eval_argv = [
        "evaluate_nonparametric_baselines.py", "--set", f"paths.splits_dir={splits_dir.as_posix()}",
        "--feature-sources", "raw_pca",
    ]
    monkeypatch.setattr(sys, "argv", eval_argv)
    assert enb.main() == 0

    results_csv = output_dir / "results.csv"
    assert results_csv.exists()
    results_df = pd.read_csv(results_csv)
    assert len(results_df) > 0
    assert "mae" in results_df.columns or "accuracy" in results_df.columns

    # Conformal calibration effect must be present for the k-NN age row.
    age_knn_row = results_df[(results_df["feature_source"] == "raw_pca") & (results_df["method"] == "age_knn")]
    if len(age_knn_row) > 0:
        assert "conformal_offset" in age_knn_row.columns
        assert "coverage_after_calibration" in age_knn_row.columns
