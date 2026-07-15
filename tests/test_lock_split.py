"""Tests for scripts/lock_split.py -- the locked, stratified split CLI (T5).

End-to-end tests run the real main() against a synthetic UTKFace-style
directory (via monkeypatched sys.argv and REPO_ROOT), never real data, and
never any heavy training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = str(REPO_ROOT / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import lock_split as ls  # noqa: E402


def test_dataframe_content_fingerprint_deterministic_and_order_independent():
    df1 = pd.DataFrame({"image_path": ["b.jpg", "a.jpg"], "age": [10, 20], "gender_label": [0, 1]})
    df2 = pd.DataFrame({"image_path": ["a.jpg", "b.jpg"], "age": [20, 10], "gender_label": [1, 0]})
    assert ls._dataframe_content_fingerprint(df1) == ls._dataframe_content_fingerprint(df2)


def test_dataframe_content_fingerprint_changes_with_content():
    df1 = pd.DataFrame({"image_path": ["a.jpg"], "age": [10], "gender_label": [0]})
    df2 = pd.DataFrame({"image_path": ["a.jpg"], "age": [11], "gender_label": [0]})
    assert ls._dataframe_content_fingerprint(df1) != ls._dataframe_content_fingerprint(df2)


def test_existing_split_is_valid_false_when_missing(tmp_path):
    assert ls.existing_split_is_valid(tmp_path) is False


def test_existing_split_is_valid_false_on_tampered_csv(tmp_path):
    split_path = tmp_path / ls.SPLIT_FILENAME
    split_path.write_text("image_path,split\na.jpg,train\n", encoding="utf-8")
    manifest_path = tmp_path / ls.MANIFEST_FILENAME
    from src.utils.io import save_json

    save_json({"split_csv_sha256": "not-the-real-hash"}, manifest_path)
    assert ls.existing_split_is_valid(tmp_path) is False


def test_existing_split_is_valid_true_when_hash_matches(tmp_path):
    from src.utils.io import file_sha256, save_json

    split_path = tmp_path / ls.SPLIT_FILENAME
    split_path.write_text("image_path,split\na.jpg,train\n", encoding="utf-8")
    manifest_path = tmp_path / ls.MANIFEST_FILENAME
    save_json({"split_csv_sha256": file_sha256(split_path)}, manifest_path)
    assert ls.existing_split_is_valid(tmp_path) is True


def test_backup_existing_split_copies_never_deletes(tmp_path):
    split_path = tmp_path / ls.SPLIT_FILENAME
    split_path.write_text("data", encoding="utf-8")
    manifest_path = tmp_path / ls.MANIFEST_FILENAME
    manifest_path.write_text("{}", encoding="utf-8")

    backup_dir = ls.backup_existing_split(tmp_path)
    assert backup_dir is not None
    assert (backup_dir / ls.SPLIT_FILENAME).exists()
    assert (backup_dir / ls.MANIFEST_FILENAME).exists()
    assert split_path.exists()  # original never deleted
    assert manifest_path.exists()


def test_backup_existing_split_none_when_nothing_to_backup(tmp_path):
    assert ls.backup_existing_split(tmp_path) is None


# -- end-to-end main() --------------------------------------------------------------


@pytest.fixture
def lock_split_env(tmp_path, synthetic_image_dir, monkeypatch):
    """Builds --set CLI overrides rather than a YAML config file: this
    project's documented config precedence is YAML defaults < .env < --set
    (see src/utils/config.py::load_config), and this repo's own .env sets
    DATA_DIR, which would otherwise silently re-derive paths.splits_dir out
    from under a plain YAML override -- --set is the only override tier
    that actually wins here."""
    image_dir, _ = synthetic_image_dir
    splits_dir = tmp_path / "splits"
    argv = [
        "lock_split.py",
        "--set", f"dataset.source=utkface",
        "--set", f"dataset.image_root={image_dir.as_posix()}",
        "--set", f"paths.splits_dir={splits_dir.as_posix()}",
        "--set", f"validation.report_dir={(tmp_path / 'quality').as_posix()}",
        "--set", "validation.min_image_size=8",
    ]
    return argv, splits_dir


def test_main_creates_locked_split_end_to_end(lock_split_env, monkeypatch):
    argv, splits_dir = lock_split_env
    monkeypatch.setattr(sys, "argv", argv)
    rc = ls.main()
    assert rc == 0
    assert (splits_dir / ls.SPLIT_FILENAME).exists()
    manifest = __import__("src.utils.io", fromlist=["load_json"]).load_json(splits_dir / ls.MANIFEST_FILENAME)
    assert manifest["split_method"] == "stratified_age_bin_x_gender_label"
    assert "stratum_counts" in manifest
    assert "age_bin_edges" in manifest
    assert manifest["identity_disjointness_caveat"]
    assert manifest["split_csv_sha256"]
    from src.utils.io import file_sha256

    assert manifest["split_csv_sha256"] == file_sha256(splits_dir / ls.SPLIT_FILENAME)


def test_main_second_run_reuses_without_resplitting(lock_split_env, monkeypatch):
    argv, splits_dir = lock_split_env
    monkeypatch.setattr(sys, "argv", argv)
    ls.main()
    first_mtime = (splits_dir / ls.SPLIT_FILENAME).stat().st_mtime_ns

    monkeypatch.setattr(sys, "argv", argv)
    rc = ls.main()
    assert rc == 0
    second_mtime = (splits_dir / ls.SPLIT_FILENAME).stat().st_mtime_ns
    assert first_mtime == second_mtime  # untouched -- reused, not regenerated


def test_main_force_resplit_backs_up_and_regenerates(lock_split_env, monkeypatch):
    argv, splits_dir = lock_split_env
    monkeypatch.setattr(sys, "argv", argv)
    ls.main()

    monkeypatch.setattr(sys, "argv", argv + ["--force-resplit"])
    rc = ls.main()
    assert rc == 0
    backup_root = splits_dir / ".backup"
    assert backup_root.exists()
    backups = list(backup_root.iterdir())
    assert len(backups) == 1
    assert (backups[0] / ls.SPLIT_FILENAME).exists()
    assert (splits_dir / ls.SPLIT_FILENAME).exists()  # new split still present


def test_main_backs_up_invalid_split_before_regenerating_without_force_flag(lock_split_env, monkeypatch):
    """A corrupted/tampered existing split must be backed up before being
    overwritten even without --force-resplit -- regenerating is not
    optional once validation fails, but silently discarding the previous
    (still potentially informative, e.g. for a diff) split is never
    acceptable either."""
    argv, splits_dir = lock_split_env
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / ls.SPLIT_FILENAME).write_text("image_path,split\ntampered.jpg,train\n", encoding="utf-8")
    from src.utils.io import save_json

    save_json({"split_csv_sha256": "not-the-real-hash"}, splits_dir / ls.MANIFEST_FILENAME)
    assert ls.existing_split_is_valid(splits_dir) is False

    monkeypatch.setattr(sys, "argv", argv)  # no --force-resplit
    rc = ls.main()
    assert rc == 0

    backup_root = splits_dir / ".backup"
    assert backup_root.exists()
    backups = list(backup_root.iterdir())
    assert len(backups) == 1
    backed_up_csv = (backups[0] / ls.SPLIT_FILENAME).read_text(encoding="utf-8")
    assert "tampered.jpg" in backed_up_csv  # the invalid split's actual content, preserved
    assert ls.existing_split_is_valid(splits_dir) is True  # freshly regenerated split is valid


def test_main_skip_near_duplicate_audit_flag(lock_split_env, monkeypatch):
    argv, splits_dir = lock_split_env
    monkeypatch.setattr(sys, "argv", argv + ["--skip-near-duplicate-audit"])
    rc = ls.main()
    assert rc == 0
    from src.utils.io import load_json

    manifest = load_json(splits_dir / ls.MANIFEST_FILENAME)
    assert manifest["near_duplicate_audit_summary"]["skipped"] is True


def test_locked_split_has_no_leakage(lock_split_env, monkeypatch):
    argv, splits_dir = lock_split_env
    monkeypatch.setattr(sys, "argv", argv)
    ls.main()

    from src.data.split_utils import assert_no_leakage

    df = pd.read_csv(splits_dir / ls.SPLIT_FILENAME)
    assert_no_leakage(df)  # raises on any leakage
    assert df["split"].isna().sum() == 0
