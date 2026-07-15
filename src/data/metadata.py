"""Parses raw dataset metadata into a unified pandas DataFrame.

Two adapters are supported:

* ``parse_utkface_directory``: parses the UTKFace filename convention
  ``age_gender_race_date.jpg`` (e.g. ``25_0_2_20170116174525125.jpg``).
* ``parse_csv_metadata``: generic adapter for Kaggle datasets that ship a
  metadata CSV, with configurable column names.

Both return a DataFrame with columns: ``image_path``, ``age``,
``gender_label``, ``race`` (metadata only, never a feature/target/split
criterion), ``subject_id`` (optional), ``split`` (optional). Missing
``age`` or ``gender_label`` values are kept as NaN so masked losses can
skip them; rows missing *both* are dropped since they carry no
supervision signal at all.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_UTKFACE_PATTERN = re.compile(r"^(?P<age>\d+)_(?P<gender>[01])_(?P<race>\d+)_(?P<date>\d+)")


def parse_utkface_directory(image_root: str | Path, glob_pattern: str = "**/*.jpg") -> pd.DataFrame:
    """Parse a directory of UTKFace-style filenames into a metadata DataFrame."""
    image_root = Path(image_root)
    records = []
    skipped = 0
    for path in sorted(image_root.glob(glob_pattern)):
        match = _UTKFACE_PATTERN.match(path.stem)
        if not match:
            skipped += 1
            continue
        records.append(
            {
                "image_path": str(path),
                "age": int(match.group("age")),
                "gender_label": int(match.group("gender")),
                "race": int(match.group("race")),
                "subject_id": None,
                "split": None,
            }
        )
    if skipped:
        logger.warning("Skipped %d files that did not match the UTKFace filename pattern", skipped)
    df = pd.DataFrame.from_records(
        records, columns=["image_path", "age", "gender_label", "race", "subject_id", "split"]
    )
    return df


def parse_csv_metadata(
    metadata_csv: str | Path,
    image_root: str | Path,
    image_path_column: str,
    age_column: str | None,
    gender_label_column: str | None,
    split_column: str | None = None,
    subject_id_column: str | None = None,
    label_mapping: dict | None = None,
) -> pd.DataFrame:
    """Parse a generic Kaggle CSV metadata file into a unified DataFrame.

    Any of ``age_column`` / ``gender_label_column`` may be None or absent
    from a given row -- rows are kept with NaN for that field so masked
    losses can skip them at training time.
    """
    raw = pd.read_csv(metadata_csv)
    image_root = Path(image_root)
    label_mapping = label_mapping or {}

    def _resolve_path(value: str) -> str:
        candidate = Path(value)
        if candidate.is_absolute():
            return str(candidate)
        return str(image_root / candidate)

    df = pd.DataFrame()
    df["image_path"] = raw[image_path_column].astype(str).map(_resolve_path)
    df["age"] = pd.to_numeric(raw[age_column], errors="coerce") if age_column and age_column in raw else float("nan")

    if gender_label_column and gender_label_column in raw:
        gender_raw = raw[gender_label_column]
        if label_mapping:
            gender_raw = gender_raw.map(lambda v: label_mapping.get(str(v), v))
        df["gender_label"] = pd.to_numeric(gender_raw, errors="coerce")
    else:
        df["gender_label"] = float("nan")

    df["race"] = None
    df["subject_id"] = raw[subject_id_column] if subject_id_column and subject_id_column in raw else None
    df["split"] = raw[split_column] if split_column and split_column in raw else None

    return df


def drop_fully_unlabeled(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where both age and gender_label are missing (no supervision signal)."""
    before = len(df)
    mask = df["age"].notna() | df["gender_label"].notna()
    filtered = df.loc[mask].reset_index(drop=True)
    dropped = before - len(filtered)
    if dropped:
        logger.info("Dropped %d rows with neither age nor gender_label labels", dropped)
    return filtered


def load_metadata(data_config: dict) -> pd.DataFrame:
    """Dispatch to the configured adapter based on ``dataset.source``."""
    dataset_cfg = data_config["dataset"]
    source = dataset_cfg.get("source", "utkface")
    if source == "utkface":
        utk_cfg = dataset_cfg.get("utkface", {})
        df = parse_utkface_directory(dataset_cfg["image_root"], utk_cfg.get("glob_pattern", "**/*.jpg"))
    elif source == "csv":
        csv_cfg = dataset_cfg["csv"]
        df = parse_csv_metadata(
            metadata_csv=csv_cfg["metadata_csv"],
            image_root=dataset_cfg["image_root"],
            image_path_column=csv_cfg["image_path_column"],
            age_column=csv_cfg.get("age_column"),
            gender_label_column=csv_cfg.get("gender_label_column"),
            split_column=csv_cfg.get("split_column"),
            subject_id_column=csv_cfg.get("subject_id_column"),
            label_mapping=csv_cfg.get("label_mapping"),
        )
    else:
        raise ValueError(f"Unknown dataset source '{source}', expected 'utkface' or 'csv'")
    return drop_fully_unlabeled(df)
