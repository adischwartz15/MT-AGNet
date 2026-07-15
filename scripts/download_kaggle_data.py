#!/usr/bin/env python
"""CLI: download the configured Kaggle dataset into data/raw/.

Usage:
    python scripts/download_kaggle_data.py [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.kaggle_download import INSTRUCTIONS, KaggleCredentialsError, download_dataset
from src.utils.config import REPO_ROOT, load_config, load_env_file
from src.utils.logging import get_logger

logger = get_logger("scripts.download_kaggle_data")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-download even if data/raw/ is non-empty")
    args = parser.parse_args()

    load_env_file()  # populate os.environ from .env (KAGGLE_USERNAME/KEY/DATASET_SLUG) if present
    config = load_config(REPO_ROOT / "configs" / "data.yaml")
    raw_dir = REPO_ROOT / config["paths"]["raw_dir"]

    try:
        manifest = download_dataset(raw_dir, force=args.force)
    except KaggleCredentialsError:
        print(INSTRUCTIONS)
        return 1

    logger.info("Manifest: %s", manifest)
    print(f"Done. {manifest['image_count']} images / {manifest['file_count']} files in {raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
