"""Data validation, corruption/duplicate filtering, and leakage-safe splitting.

Never uses file timestamps, filenames, race metadata, source URLs, or split
metadata as model features -- this module only ever touches ``image_path``,
``age``, ``gender_label``, and (for grouping) ``subject_id``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.data.split_utils import assert_no_leakage, split_dataframe, stratified_split_dataframe
from src.utils.io import file_sha256, save_json

logger = logging.getLogger(__name__)


def _check_image(path: str, min_size: int, max_file_size_mb: float) -> dict | None:
    """Return image diagnostics dict, or None if the file is unreadable/corrupt/too small."""
    p = Path(path)
    if not p.exists():
        return None
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > max_file_size_mb:
        return None
    try:
        with Image.open(p) as img:
            img.verify()
        with Image.open(p) as img:
            width, height = img.size
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    if width < min_size or height < min_size:
        return None
    return {"width": width, "height": height, "size_mb": size_mb}


def validate_dataset(df: pd.DataFrame, validation_cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Filter corrupt/missing/duplicate images and compute a data-quality report.

    Returns the cleaned DataFrame (with ``width``/``height`` columns added)
    and a JSON-serializable report dict.
    """
    min_size = validation_cfg.get("min_image_size", 32)
    max_file_size_mb = validation_cfg.get("max_file_size_mb", 20)

    n_input = len(df)
    diagnostics = []
    keep_mask = []
    for path in df["image_path"]:
        diag = _check_image(path, min_size, max_file_size_mb)
        diagnostics.append(diag)
        keep_mask.append(diag is not None)

    n_corrupt_or_unreadable = n_input - sum(keep_mask)
    df = df.loc[keep_mask].copy()
    diagnostics = [d for d in diagnostics if d is not None]
    df["width"] = [d["width"] for d in diagnostics]
    df["height"] = [d["height"] for d in diagnostics]

    n_missing_age = int(df["age"].isna().sum())
    n_missing_gender = int(df["gender_label"].isna().sum())

    duplicate_paths = int(df["image_path"].duplicated().sum())
    if validation_cfg.get("detect_duplicate_paths", True) and duplicate_paths:
        df = df.drop_duplicates(subset="image_path", keep="first")

    n_duplicate_hashes = 0
    if validation_cfg.get("detect_duplicate_hashes", True):
        hashes = df["image_path"].map(file_sha256)
        df = df.assign(_hash=hashes)
        n_duplicate_hashes = int(df["_hash"].duplicated().sum())
        df = df.drop_duplicates(subset="_hash", keep="first").drop(columns="_hash")

    age_series = df["age"].dropna()
    gender_series = df["gender_label"].dropna()

    report = {
        "n_input_rows": n_input,
        "n_dropped_corrupt_or_unreadable": n_corrupt_or_unreadable,
        "n_duplicate_paths_removed": duplicate_paths,
        "n_duplicate_hashes_removed": n_duplicate_hashes,
        "n_final_rows": len(df),
        "n_missing_age": n_missing_age,
        "n_missing_gender_label": n_missing_gender,
        "age_distribution": {
            "count": int(age_series.count()),
            "mean": float(age_series.mean()) if len(age_series) else None,
            "std": float(age_series.std()) if len(age_series) else None,
            "min": float(age_series.min()) if len(age_series) else None,
            "max": float(age_series.max()) if len(age_series) else None,
            "quantiles": {
                q: float(v) for q, v in age_series.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).items()
            } if len(age_series) else {},
        },
        "gender_label_distribution": {
            str(k): int(v) for k, v in gender_series.value_counts().items()
        },
        "image_size_stats": {
            "width_mean": float(df["width"].mean()) if len(df) else None,
            "height_mean": float(df["height"].mean()) if len(df) else None,
            "width_min": int(df["width"].min()) if len(df) else None,
            "width_max": int(df["width"].max()) if len(df) else None,
        },
        "has_subject_id": bool(df["subject_id"].notna().any()) if "subject_id" in df.columns else False,
    }
    return df.reset_index(drop=True), report


def validate_and_split(df: pd.DataFrame, data_config: dict) -> tuple[pd.DataFrame, dict]:
    """Run validation, then a deterministic (leakage-checked) train/validation/calibration/test split.

    Note: ``data_config["validation"]`` is the *image-quality* validation
    config (corrupt/duplicate detection etc.); the *data split* named
    "validation" is configured separately under ``data_config["split"]``
    and produced by :func:`split_dataframe`. These are two different uses
    of the word "validation" that predate the 4-way split protocol.
    """
    validation_cfg = data_config.get("validation", {})
    split_cfg = data_config.get("split", {})

    clean_df, report = validate_dataset(df, validation_cfg)
    split_df = split_dataframe(
        clean_df,
        train_fraction=split_cfg.get("train_fraction", 0.60),
        validation_fraction=split_cfg.get("validation_fraction", 0.15),
        calibration_fraction=split_cfg.get("calibration_fraction", 0.10),
        test_fraction=split_cfg.get("test_fraction", 0.15),
        seed=split_cfg.get("seed", 42),
        subject_level_if_available=split_cfg.get("subject_level_if_available", True),
    )
    assert_no_leakage(split_df)

    report["split_counts"] = split_df["split"].value_counts().to_dict()
    return split_df, report


def validate_and_stratified_split(df: pd.DataFrame, data_config: dict) -> tuple[pd.DataFrame, dict]:
    """Same contract as :func:`validate_and_split`, but using the locked,
    age-bin x gender-label stratified split (:func:`~src.data.split_utils.stratified_split_dataframe`)
    instead of the plain (unstratified) row/subject shuffle -- see
    ``scripts/lock_split.py`` and ``docs/reproducibility.md`` "Locked
    stratified split" for why this is the split every final experiment uses.
    """
    validation_cfg = data_config.get("validation", {})
    split_cfg = data_config.get("split", {})

    clean_df, report = validate_dataset(df, validation_cfg)
    split_df, stratification_report = stratified_split_dataframe(
        clean_df,
        train_fraction=split_cfg.get("train_fraction", 0.60),
        validation_fraction=split_cfg.get("validation_fraction", 0.15),
        calibration_fraction=split_cfg.get("calibration_fraction", 0.10),
        test_fraction=split_cfg.get("test_fraction", 0.15),
        seed=split_cfg.get("seed", 42),
        subject_level_if_available=split_cfg.get("subject_level_if_available", True),
    )
    assert_no_leakage(split_df)

    report["split_counts"] = split_df["split"].value_counts().to_dict()
    report["stratification"] = stratification_report
    return split_df, report


def save_data_quality_report(report: dict, output_dir: str) -> Path:
    out_path = Path(output_dir) / "data_quality_report.json"
    save_json(report, out_path)
    logger.info("Saved data quality report to %s", out_path)
    return out_path
