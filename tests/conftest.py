"""Shared pytest fixtures: synthetic (non-real) images and configs for fast tests.

Synthetic data is used only for tests/smoke tests and must never be mixed
with real Kaggle experiment results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.utils.config import REPO_ROOT, load_full_config

# Real, repo-level directories that hold generated artifacts (splits,
# checkpoints, outputs, results, logs) -- tests must operate exclusively
# through tmp_path/--set overrides and must NEVER write here. Guards
# against a real regression found in this project: an earlier draft of
# tests/test_lock_split.py didn't override paths.splits_dir via --set (the
# only override tier that beats this repo's own .env DATA_DIR setting --
# see src/utils/config.py::load_config), so scripts/lock_split.py silently
# wrote synthetic pytest-fixture data into the real data/splits/ directory.
_PROTECTED_REAL_DIRS = [
    REPO_ROOT / "data" / "splits",
    REPO_ROOT / "checkpoints",
    REPO_ROOT / "outputs",
    REPO_ROOT / "results",
    REPO_ROOT / "logs",
]
_ALWAYS_ALLOWED_NAMES = {".gitkeep"}


def _snapshot(paths: list[Path]) -> dict[Path, set[Path]]:
    snapshot = {}
    for root in paths:
        if root.exists():
            snapshot[root] = {p for p in root.rglob("*") if p.is_file()}
        else:
            snapshot[root] = set()
    return snapshot


@pytest.fixture(autouse=True)
def _guard_real_artifact_directories():
    """Fails the test loudly (rather than leaving silent residue) if it
    wrote into a real, repo-level artifact directory instead of tmp_path."""
    before = _snapshot(_PROTECTED_REAL_DIRS)
    yield
    after = _snapshot(_PROTECTED_REAL_DIRS)
    new_files = []
    for root in _PROTECTED_REAL_DIRS:
        added = after[root] - before[root]
        new_files.extend(p for p in added if p.name not in _ALWAYS_ALLOWED_NAMES)
    assert not new_files, (
        "Test wrote into a real, repo-level artifact directory instead of an "
        f"isolated tmp_path: {new_files}. Use --set <path key>=<tmp_path>/... "
        "(the highest-priority config override tier) rather than a YAML file "
        "or relying on defaults, since this repo's .env DATA_DIR setting "
        "silently wins over a plain YAML override."
    )


@pytest.fixture
def synthetic_image_dir(tmp_path):
    """Create a small UTKFace-style synthetic image directory."""
    image_dir = tmp_path / "raw"
    image_dir.mkdir()
    rng = np.random.default_rng(0)
    records = []
    for i in range(40):
        age = int(rng.integers(1, 90))
        gender = int(rng.integers(0, 2))
        filename = f"{age}_{gender}_0_2017011617452{i:04d}.jpg"
        array = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
        Image.fromarray(array).save(image_dir / filename)
        records.append({"age": age, "gender_label": gender, "path": str(image_dir / filename)})
    return image_dir, pd.DataFrame(records)


@pytest.fixture
def synthetic_metadata_df(synthetic_image_dir):
    _, records_df = synthetic_image_dir
    n = len(records_df)
    df = pd.DataFrame(
        {
            "image_path": records_df["path"],
            "age": records_df["age"].astype(float),
            "gender_label": records_df["gender_label"].astype(float),
            "race": 0,
            "subject_id": None,
            "split": None,
        }
    )
    return df


@pytest.fixture
def tiny_config():
    """A full config with a tiny model/backbone-friendly image size for fast CPU tests."""
    config = load_full_config()
    config["dataset"]["image_size"] = 32
    config["model"]["adapters"]["bottleneck_dim"] = 16
    config["model"]["age_head"]["hidden_dim"] = 16
    config["model"]["gender_head"]["hidden_dim"] = 16
    config["training"]["batch_size"] = 4
    config["training"]["num_workers"] = 0
    config["training"]["mixed_precision"] = False
    config["training"]["stages"]["stage_a"]["epochs"] = 1
    config["training"]["stages"]["stage_b"]["epochs"] = 1
    config["training"]["stages"]["stage_c"]["epochs"] = 1
    config["training"]["warm_up_from_scratch"]["epochs"] = 1
    config["training"]["early_stopping_patience"] = 100
    config["seed"] = 0
    return config
