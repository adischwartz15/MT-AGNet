"""Tests for the Kaggle download skip-logic (must not confuse placeholder
files or incomplete downloads with an actual dataset)."""

from __future__ import annotations

from src.data.kaggle_download import _dataset_already_downloaded


def test_missing_directory_is_not_downloaded(tmp_path):
    assert _dataset_already_downloaded(tmp_path / "does_not_exist") is False


def test_empty_directory_is_not_downloaded(tmp_path):
    assert _dataset_already_downloaded(tmp_path) is False


def test_gitkeep_placeholder_alone_is_not_downloaded(tmp_path):
    """A fresh checkout ships data/raw/.gitkeep -- that must not look like real data."""
    (tmp_path / ".gitkeep").write_text("")
    assert _dataset_already_downloaded(tmp_path) is False


def test_stray_manifest_without_images_is_not_downloaded(tmp_path):
    """A manifest.json from a previously failed/partial run is not itself data."""
    (tmp_path / "manifest.json").write_text("{}")
    assert _dataset_already_downloaded(tmp_path) is False


def test_unextracted_zip_alone_is_not_downloaded(tmp_path):
    """A downloaded-but-not-yet-extracted .zip should not count as 'already downloaded'."""
    (tmp_path / "dataset.zip").write_bytes(b"PK\x03\x04")
    assert _dataset_already_downloaded(tmp_path) is False


def test_top_level_image_is_downloaded(tmp_path):
    (tmp_path / "0_0_0_20170101000000000.jpg").write_bytes(b"\xff\xd8\xff")
    assert _dataset_already_downloaded(tmp_path) is True


def test_nested_image_is_found_recursively(tmp_path):
    nested = tmp_path / "UTKFace" / "part1"
    nested.mkdir(parents=True)
    (nested / "25_1_2_20170101000000000.jpg").write_bytes(b"\xff\xd8\xff")
    (tmp_path / ".gitkeep").write_text("")
    assert _dataset_already_downloaded(tmp_path) is True


def test_case_insensitive_extension_match(tmp_path):
    (tmp_path / "photo.JPG").write_bytes(b"\xff\xd8\xff")
    assert _dataset_already_downloaded(tmp_path) is True
