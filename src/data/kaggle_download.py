"""Downloads a configured dataset through the official Kaggle API.

No scraping, no bypassing licenses, no hardcoded credentials, no private
datasets. Credentials are read exclusively from the ``KAGGLE_USERNAME`` /
``KAGGLE_KEY`` environment variables (the same variables the official
``kaggle`` package expects), and the dataset slug comes from
``KAGGLE_DATASET_SLUG``. Nothing here ever commits credentials, tokens, or
downloaded data to the repository (see .gitignore).
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import zipfile
from pathlib import Path

from src.utils.io import save_json

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

INSTRUCTIONS = """
Kaggle credentials or dataset slug are missing. To download data:

  1. Create a Kaggle account and API token at https://www.kaggle.com/settings
     ("Create New Token" downloads kaggle.json).
  2. Set environment variables (see .env.example):
       KAGGLE_USERNAME=<your-username>
       KAGGLE_KEY=<your-key>
       KAGGLE_DATASET_SLUG=<owner>/<dataset-name>   (e.g. jangedoo/utkface-new)
  3. Re-run: python scripts/download_kaggle_data.py

Credentials are never hardcoded or committed. This script only downloads
data you are entitled to access under Kaggle's terms.
"""


class KaggleCredentialsError(RuntimeError):
    pass


def validate_credentials() -> tuple[str, str, str]:
    username = os.environ.get("KAGGLE_USERNAME", "").strip()
    key = os.environ.get("KAGGLE_KEY", "").strip()
    slug = os.environ.get("KAGGLE_DATASET_SLUG", "").strip()
    if not username or not key or not slug:
        raise KaggleCredentialsError(INSTRUCTIONS)
    return username, key, slug


def _dataset_already_downloaded(raw_dir: Path) -> bool:
    """True only if ``raw_dir`` recursively contains at least one real image file.

    Deliberately not just "the directory exists / has any entry": a fresh
    checkout of this repo already ships ``data/raw/.gitkeep`` (see
    .gitignore), and a failed prior run can leave a stray ``manifest.json``
    or an unextracted ``.zip`` behind. None of those should cause a real
    download to be skipped.
    """
    if not raw_dir.exists():
        return False
    return any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in raw_dir.rglob("*"))


def download_dataset(raw_dir: str | Path, force: bool = False) -> dict:
    """Download and extract ``KAGGLE_DATASET_SLUG`` into ``raw_dir``.

    Returns a manifest dict (also saved as ``raw_dir/manifest.json``) with
    the dataset slug, download timestamp, and file/image counts. Skips the
    download if files already exist in ``raw_dir`` unless ``force=True``.
    """
    username, key, slug = validate_credentials()
    os.environ.setdefault("KAGGLE_USERNAME", username)
    os.environ.setdefault("KAGGLE_KEY", key)

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if _dataset_already_downloaded(raw_dir) and not force:
        logger.info("Data already present in %s (use --force to re-download)", raw_dir)
        return _build_manifest(raw_dir, slug, skipped=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "The 'kaggle' package is required. Install it with: pip install kaggle"
        ) from exc

    api = KaggleApi()
    api.authenticate()
    logger.info("Downloading Kaggle dataset '%s' into %s", slug, raw_dir)
    api.dataset_download_files(slug, path=str(raw_dir), unzip=False, quiet=False)

    for zip_path in raw_dir.glob("*.zip"):
        logger.info("Extracting %s", zip_path.name)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
        zip_path.unlink()

    manifest = _build_manifest(raw_dir, slug, skipped=False)
    save_json(manifest, raw_dir / "manifest.json")
    return manifest


def _build_manifest(raw_dir: Path, slug: str, skipped: bool) -> dict:
    all_files = [p for p in raw_dir.rglob("*") if p.is_file() and p.name != "manifest.json"]
    image_files = [p for p in all_files if p.suffix.lower() in IMAGE_EXTENSIONS]
    return {
        "dataset_slug": slug,
        "downloaded_at": dt.datetime.utcnow().isoformat() + "Z",
        "skipped_existing_download": skipped,
        "file_count": len(all_files),
        "image_count": len(image_files),
    }
