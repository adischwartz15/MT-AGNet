#!/usr/bin/env python
"""CLI: parse raw dataset metadata, validate images, and create a leakage-safe split.

Usage:
    python scripts/prepare_data.py [--config configs/data.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.metadata import load_metadata
from src.data.validation import save_data_quality_report, validate_and_split
from src.utils.config import REPO_ROOT, load_config, parse_cli_overrides
from src.utils.logging import get_logger

logger = get_logger("scripts.prepare_data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Extra YAML config to merge on top of configs/data.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key.path=value overrides")
    args = parser.parse_args()

    extra = [REPO_ROOT / "configs" / "data.yaml"]
    if args.config:
        extra.append(args.config)
    config = load_config(*extra, overrides=parse_cli_overrides(args.overrides))

    logger.info("Loading metadata (source=%s)", config["dataset"]["source"])
    df = load_metadata(config)
    if len(df) == 0:
        logger.error(
            "No labeled samples found. Populate %s with a dataset (see 'make download-data' "
            "and README.md's 'Dataset format instructions').",
            config["dataset"]["image_root"],
        )
        return 1

    logger.info("Loaded %d raw rows; validating and splitting...", len(df))
    split_df, report = validate_and_split(df, config)

    splits_dir = REPO_ROOT / config["paths"]["splits_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)
    out_csv = splits_dir / "full_metadata_with_splits.csv"
    split_df.to_csv(out_csv, index=False)

    report_dir = REPO_ROOT / config["validation"]["report_dir"]
    save_data_quality_report(report, report_dir)

    logger.info("Saved split metadata to %s", out_csv)
    logger.info("Split counts: %s", report["split_counts"])
    print(f"Prepared {len(split_df)} samples. Splits: {report['split_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
