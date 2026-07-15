#!/usr/bin/env python
"""CLI: create (or verify-and-reuse) the locked, stratified, deterministic
4-way train/validation/calibration/test split every final experiment uses.

Stratifies by age-bin x gender-label (src/data/split_utils.py::AGE_BIN_EDGES)
so every stratum -- not just the dataset overall -- gets approximately the
configured split proportions; preserves subject-level grouping (no
subject's images cross a split boundary) when subject_id is available.

Default behavior: if a valid locked split already exists (its manifest's
recorded split-CSV SHA-256 matches the file's actual current content),
verify it and reuse it -- this script becomes a no-op printing the existing
counts. Pass --force-resplit to intentionally replace it. Whenever an
existing split is about to be overwritten -- an explicit --force-resplit,
or an existing split/manifest that fails validation (missing, corrupted,
or tampered) -- the previous split and manifest are backed up (copied,
never deleted) to data/splits/.backup/pre_regenerate_<UTC-timestamp>/
first.

Usage:
    python scripts/lock_split.py                    # create if missing, else verify+reuse
    python scripts/lock_split.py --force-resplit     # intentionally regenerate
    python scripts/lock_split.py --skip-near-duplicate-audit   # faster, for large datasets
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.metadata import load_metadata  # noqa: E402
from src.data.near_duplicate_audit import find_near_duplicate_candidates, summarize_near_duplicate_audit  # noqa: E402
from src.data.split_utils import AGE_BIN_EDGES, age_bin_label  # noqa: E402
from src.data.validation import save_data_quality_report, validate_and_stratified_split  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, parse_cli_overrides  # noqa: E402
from src.utils.io import file_sha256, load_json, save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.provenance import dependency_versions as _dependency_versions, git_commit_sha as _git_commit_sha  # noqa: E402

logger = get_logger("scripts.lock_split")

SPLIT_FILENAME = "full_metadata_with_splits.csv"
MANIFEST_FILENAME = "split_manifest.json"


def _dataframe_content_fingerprint(df) -> str:
    """SHA-256 of the raw loaded metadata's content (image_path/age/gender_label/
    race, sorted by image_path for a deterministic row order) -- distinct
    from ``split_csv_sha256`` (the *output* split file's hash, which also
    reflects cleaning/dedup/the split column). This is "what raw dataset did
    we split", independent of how it was subsequently processed.
    """
    import hashlib

    cols = [c for c in ("image_path", "age", "gender_label", "race") if c in df.columns]
    canonical = df[cols].sort_values("image_path").to_csv(index=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_write_csv(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)


def _atomic_save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    save_json(data, tmp_path)
    os.replace(tmp_path, path)


def existing_split_is_valid(splits_dir: Path) -> bool:
    """A locked split is valid only if BOTH the split CSV and its manifest
    exist AND the manifest's recorded split_csv_sha256 matches the CSV
    file's actual current content -- never inferred from file existence
    alone (the same principle src/training/persistent_artifacts.py uses for
    seed completion)."""
    split_path = splits_dir / SPLIT_FILENAME
    manifest_path = splits_dir / MANIFEST_FILENAME
    if not split_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = load_json(manifest_path)
    except (OSError, ValueError):
        return False
    recorded_hash = manifest.get("split_csv_sha256")
    if not recorded_hash:
        return False
    return file_sha256(split_path) == recorded_hash


def backup_existing_split(splits_dir: Path) -> Path | None:
    """Copy (never delete) the current split CSV + manifest into a
    timestamped backup subdirectory before regenerating -- called both for
    an explicit --force-resplit and whenever an existing split/manifest
    fails validation (missing, corrupted, or tampered), since either way a
    split is about to be overwritten. Returns the backup directory, or None
    if there was nothing on disk to back up."""
    split_path = splits_dir / SPLIT_FILENAME
    manifest_path = splits_dir / MANIFEST_FILENAME
    if not split_path.exists() and not manifest_path.exists():
        return None

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = splits_dir / ".backup" / f"pre_regenerate_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if split_path.exists():
        shutil.copy2(split_path, backup_dir / SPLIT_FILENAME)
    if manifest_path.exists():
        shutil.copy2(manifest_path, backup_dir / MANIFEST_FILENAME)
    logger.info("Backed up previous split + manifest to %s (not deleted).", backup_dir)
    return backup_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Extra YAML config to merge on top of configs/data.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key.path=value overrides")
    parser.add_argument(
        "--force-resplit", action="store_true",
        help="Intentionally regenerate the split even if a valid one already exists (backs up the previous one first)",
    )
    parser.add_argument(
        "--skip-near-duplicate-audit", action="store_true",
        help="Skip the (non-destructive, report-only) near-duplicate audit -- faster for a very large dataset",
    )
    args = parser.parse_args()

    extra = [REPO_ROOT / "configs" / "data.yaml"]
    if args.config:
        extra.append(args.config)
    config = load_config(*extra, overrides=parse_cli_overrides(args.overrides))
    splits_dir = REPO_ROOT / config["paths"]["splits_dir"]

    if not args.force_resplit and existing_split_is_valid(splits_dir):
        manifest = load_json(splits_dir / MANIFEST_FILENAME)
        logger.info("A valid locked split already exists at %s -- reusing it (pass --force-resplit to replace).", splits_dir)
        print(f"Reusing existing locked split at {splits_dir / SPLIT_FILENAME}")
        print(f"split_csv_sha256={manifest['split_csv_sha256']}")
        print(f"Split counts: {manifest.get('split_counts')}")
        return 0

    # Back up whatever is currently on disk before regenerating -- whether
    # this is an explicit --force-resplit, or the existing split/manifest
    # failed validation (missing, corrupted, or tampered): either way, a
    # split is about to be overwritten, and the previous one must never be
    # silently discarded. backup_existing_split() is a no-op (returns None)
    # if there is genuinely nothing on disk yet.
    backup_existing_split(splits_dir)

    logger.info("Loading metadata (source=%s)", config["dataset"]["source"])
    df = load_metadata(config)
    if len(df) == 0:
        logger.error(
            "No labeled samples found. Populate %s with a dataset (see 'make download-data' "
            "and README.md's 'Dataset format instructions').",
            config["dataset"]["image_root"],
        )
        return 1

    logger.info("Loaded %d raw rows; validating and building the stratified split...", len(df))
    source_metadata_fingerprint = _dataframe_content_fingerprint(df)
    split_df, report = validate_and_stratified_split(df, config)

    near_dup_summary = {"skipped": True}
    if not args.skip_near_duplicate_audit:
        logger.info("Running near-duplicate audit (non-destructive, report-only)...")
        candidates_df = find_near_duplicate_candidates(split_df)
        near_dup_summary = summarize_near_duplicate_audit(candidates_df)
        near_dup_summary["skipped"] = False
        if len(candidates_df) > 0:
            near_dup_report_path = REPO_ROOT / config["validation"]["report_dir"] / "near_duplicate_candidates.csv"
            near_dup_report_path.parent.mkdir(parents=True, exist_ok=True)
            candidates_df.to_csv(near_dup_report_path, index=False)
            logger.warning(
                "%d near-duplicate candidate pair(s) found (%d cross-split) -- see %s for human review. "
                "Nothing was automatically removed.",
                len(candidates_df), int(candidates_df["cross_split"].sum()), near_dup_report_path,
            )

    splits_dir.mkdir(parents=True, exist_ok=True)
    split_path = splits_dir / SPLIT_FILENAME
    _atomic_write_csv(split_df, split_path)

    report_dir = REPO_ROOT / config["validation"]["report_dir"]
    save_data_quality_report(report, report_dir)

    age_bins = split_df["age"].apply(lambda a: age_bin_label(a) if a == a else "unknown")
    age_bin_counts = {str(k): int(v) for k, v in age_bins.value_counts().items()}
    gender_label_counts = {str(k): int(v) for k, v in split_df["gender_label"].value_counts(dropna=False).items()}

    manifest = {
        "split_method": "stratified_age_bin_x_gender_label",
        "split_seed": config["split"].get("seed", 42),
        "split_fractions": {
            "train": config["split"].get("train_fraction", 0.60),
            "validation": config["split"].get("validation_fraction", 0.15),
            "calibration": config["split"].get("calibration_fraction", 0.10),
            "test": config["split"].get("test_fraction", 0.15),
        },
        "stratification_fields": ["age_bin", "gender_label"],
        "age_bin_edges": list(AGE_BIN_EDGES),
        "subject_level_if_available": config["split"].get("subject_level_if_available", True),
        "stratified_by": report["stratification"]["stratified_by"],
        "source_metadata_fingerprint": source_metadata_fingerprint,
        "source_image_count": report["n_input_rows"],
        "split_csv_sha256": file_sha256(split_path),
        "split_counts": report["split_counts"],
        "age_bin_counts": age_bin_counts,
        "gender_label_counts": gender_label_counts,
        "stratum_counts": report["stratification"]["stratum_counts"],
        "zero_allocation_warnings": report["stratification"]["zero_allocation_warnings"],
        "duplicate_path_audit": {
            "n_duplicate_paths_removed": report["n_duplicate_paths_removed"],
        },
        "exact_hash_duplicate_audit": {
            "n_duplicate_hashes_removed": report["n_duplicate_hashes_removed"],
        },
        "near_duplicate_audit_summary": near_dup_summary,
        "identity_disjointness_caveat": (
            "UTKFace does not ship a reliable subject/identity column. When subject_id is "
            "unavailable (the common case for this dataset), splitting is image-level: exact "
            "byte-identical duplicates are removed (see duplicate_path_audit / "
            "exact_hash_duplicate_audit above) and a best-effort perceptual near-duplicate "
            "audit is run (see near_duplicate_audit_summary / near_duplicate_candidates.csv), "
            "but true identity-disjointness across splits (the same real person appearing "
            "under visually distinct photos in more than one split) CANNOT be guaranteed for "
            "this dataset. Report any scientific claims accordingly."
        ),
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit_sha": _git_commit_sha(),
        "dependency_versions": _dependency_versions(),
    }
    _atomic_save_json(manifest, splits_dir / MANIFEST_FILENAME)

    logger.info("Locked stratified split written to %s", split_path)
    print(f"Prepared {len(split_df)} samples. Split counts: {report['split_counts']}")
    print(f"Stratum count: {report['stratification']['n_strata']}")
    print(f"split_csv_sha256={manifest['split_csv_sha256']}")
    if manifest["zero_allocation_warnings"]:
        print(f"WARNING: {len(manifest['zero_allocation_warnings'])} (stratum, split) pair(s) got zero rows -- see manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
