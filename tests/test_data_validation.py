"""Tests for data validation, corruption/duplicate filtering, and leakage-safe splitting."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from src.data.metadata import parse_utkface_directory
from src.data.split_utils import assert_no_leakage, split_dataframe
from src.data.validation import validate_and_split, validate_dataset


def test_parse_utkface_directory(synthetic_image_dir):
    image_dir, records_df = synthetic_image_dir
    df = parse_utkface_directory(image_dir)
    assert len(df) == len(records_df)
    assert set(df.columns) >= {"image_path", "age", "gender_label", "race"}
    assert df["age"].between(0, 120).all()
    assert df["gender_label"].isin([0, 1]).all()


def test_validate_dataset_drops_corrupt_file(synthetic_metadata_df, tmp_path):
    df = synthetic_metadata_df.copy()
    corrupt_path = tmp_path / "corrupt.jpg"
    corrupt_path.write_bytes(b"not a real image")
    df = pd.concat([df, pd.DataFrame([{
        "image_path": str(corrupt_path), "age": 30.0, "gender_label": 0.0,
        "race": 0, "subject_id": None, "split": None,
    }])], ignore_index=True)

    clean_df, report = validate_dataset(df, {"min_image_size": 8, "max_file_size_mb": 20})
    assert str(corrupt_path) not in set(clean_df["image_path"])
    assert report["n_dropped_corrupt_or_unreadable"] >= 1


def test_validate_dataset_drops_duplicate_paths(synthetic_metadata_df):
    df = synthetic_metadata_df.copy()
    duplicated_row = df.iloc[[0]]
    df = pd.concat([df, duplicated_row], ignore_index=True)

    clean_df, report = validate_dataset(df, {"min_image_size": 8, "max_file_size_mb": 20, "detect_duplicate_paths": True})
    assert report["n_duplicate_paths_removed"] == 1
    assert clean_df["image_path"].duplicated().sum() == 0


def test_validate_dataset_drops_duplicate_hashes(synthetic_metadata_df, tmp_path):
    df = synthetic_metadata_df.copy()
    original_path = df.iloc[0]["image_path"]
    copy_path = tmp_path / "duplicate_content.jpg"
    shutil.copyfile(original_path, copy_path)
    df = pd.concat([df, pd.DataFrame([{
        "image_path": str(copy_path), "age": 40.0, "gender_label": 1.0,
        "race": 0, "subject_id": None, "split": None,
    }])], ignore_index=True)

    clean_df, report = validate_dataset(
        df, {"min_image_size": 8, "max_file_size_mb": 20, "detect_duplicate_hashes": True}
    )
    assert report["n_duplicate_hashes_removed"] >= 1


def test_split_dataframe_respects_fractions(synthetic_metadata_df):
    df = split_dataframe(synthetic_metadata_df, 0.5, 0.2, 0.1, 0.2, seed=42, subject_level_if_available=False)
    counts = df["split"].value_counts(normalize=True)
    assert abs(counts.get("train", 0) - 0.5) < 0.15


def test_split_dataframe_produces_all_four_splits(synthetic_metadata_df):
    df = split_dataframe(synthetic_metadata_df, 0.4, 0.2, 0.2, 0.2, seed=3, subject_level_if_available=False)
    assert set(df["split"].unique()) <= {"train", "validation", "calibration", "test"}
    # With a reasonably sized synthetic set and non-trivial fractions, expect all four present.
    assert set(df["split"].unique()) == {"train", "validation", "calibration", "test"}


def test_split_dataframe_deterministic_with_seed(synthetic_metadata_df):
    df1 = split_dataframe(synthetic_metadata_df.copy(), seed=7, subject_level_if_available=False)
    df2 = split_dataframe(synthetic_metadata_df.copy(), seed=7, subject_level_if_available=False)
    assert list(df1["split"]) == list(df2["split"])


def test_subject_level_split_keeps_subject_in_one_split(synthetic_metadata_df):
    df = synthetic_metadata_df.copy()
    # Assign every pair of rows the same subject_id.
    df["subject_id"] = [i // 2 for i in range(len(df))]
    split_df = split_dataframe(df, seed=1, subject_level_if_available=True)
    assert_no_leakage(split_df)  # should not raise


def test_assert_no_leakage_raises_on_duplicated_path_across_splits(synthetic_metadata_df):
    df = synthetic_metadata_df.copy()
    df["split"] = "train"
    df.loc[0, "split"] = "train"
    duplicated = df.iloc[[0]].copy()
    duplicated["split"] = "test"
    leaking_df = pd.concat([df, duplicated], ignore_index=True)
    try:
        assert_no_leakage(leaking_df)
        assert False, "expected ValueError due to leakage"
    except ValueError:
        pass


def test_validate_and_split_end_to_end(synthetic_metadata_df):
    data_config = {
        "validation": {"min_image_size": 8, "max_file_size_mb": 20},
        "split": {
            "train_fraction": 0.6, "validation_fraction": 0.15, "calibration_fraction": 0.10,
            "test_fraction": 0.15, "seed": 0, "subject_level_if_available": True,
        },
    }
    split_df, report = validate_and_split(synthetic_metadata_df, data_config)
    assert set(split_df["split"].unique()) <= {"train", "validation", "calibration", "test"}
    assert "age_distribution" in report
    assert "gender_label_distribution" in report
