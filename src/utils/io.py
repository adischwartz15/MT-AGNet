"""Small IO helpers: JSON/YAML round-trips and image hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_yaml(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def _json_default(obj: Any) -> Any:
    import numpy as np

    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a file's contents (for duplicate detection)."""
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


CHECKPOINT_BEST_SUFFIXES = ("_best_balanced_score", "_best_age_mae", "_best_gender_accuracy")


def checkpoint_experiment_name(checkpoint_path: str | Path) -> str:
    """Derive the experiment name from a checkpoint filename.

    E.g. "exp_c_shared_adapters_best_balanced_score.pt" -> "exp_c_shared_adapters".
    Shared by scripts/evaluate.py and scripts/calibrate.py so both derive
    the same experiment name from the same checkpoint filename.
    """
    stem = Path(checkpoint_path).stem
    for suffix in CHECKPOINT_BEST_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem
