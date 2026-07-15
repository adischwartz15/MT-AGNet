"""Tests for src/data/near_duplicate_audit.py (T5) -- perceptual-hash-based
near-duplicate candidate detection. Synthetic PIL images only.
"""

from __future__ import annotations

import pandas as pd
from PIL import Image

from src.data.near_duplicate_audit import (
    compute_hashes,
    difference_hash,
    find_near_duplicate_candidates,
    hamming_distance,
    summarize_near_duplicate_audit,
)


def _make_gradient_image(path, size=(64, 64), seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size[1], size[0], 3), dtype="uint8")
    Image.fromarray(arr, mode="RGB").save(path)


def test_identical_image_has_zero_hamming_distance(tmp_path):
    path = tmp_path / "a.jpg"
    _make_gradient_image(path, seed=1)
    h1 = difference_hash(path)
    h2 = difference_hash(path)
    assert hamming_distance(h1, h2) == 0


def test_resized_copy_is_near_duplicate(tmp_path):
    """A resized copy of the same image should hash very close to the original."""
    original_path = tmp_path / "orig.jpg"
    _make_gradient_image(original_path, size=(200, 200), seed=2)
    with Image.open(original_path) as img:
        resized = img.resize((80, 80))
    resized_path = tmp_path / "resized.jpg"
    resized.save(resized_path)

    h1 = difference_hash(original_path)
    h2 = difference_hash(resized_path)
    assert hamming_distance(h1, h2) <= 8  # small distance for a resize of the same content


def test_unrelated_images_have_large_distance_on_average(tmp_path):
    paths = []
    for i in range(6):
        p = tmp_path / f"img_{i}.jpg"
        _make_gradient_image(p, seed=100 + i)
        paths.append(p)
    hashes = [difference_hash(p) for p in paths]
    distances = [hamming_distance(hashes[i], hashes[j]) for i in range(len(hashes)) for j in range(i + 1, len(hashes))]
    # Random noise images should mostly NOT collide within a tight threshold.
    assert sum(1 for d in distances if d <= 4) <= len(distances) // 2


def test_compute_hashes_handles_unreadable_file_gracefully(tmp_path):
    bad_path = tmp_path / "not_an_image.jpg"
    bad_path.write_text("this is not image data", encoding="utf-8")
    hashes = compute_hashes([str(bad_path)])
    assert hashes[str(bad_path)] is None


def test_find_near_duplicate_candidates_flags_cross_split_pair(tmp_path):
    p1 = tmp_path / "a.jpg"
    _make_gradient_image(p1, seed=5)
    with Image.open(p1) as img:
        img.resize((50, 50)).save(tmp_path / "b.jpg")
    p2 = tmp_path / "b.jpg"

    df = pd.DataFrame({"image_path": [str(p1), str(p2)], "split": ["train", "test"]})
    candidates = find_near_duplicate_candidates(df, max_hamming_distance=10)
    assert len(candidates) >= 1
    row = candidates.iloc[0]
    assert row["cross_split"] == True  # noqa: E712 (explicit bool comparison for clarity)


def test_summarize_near_duplicate_audit_empty():
    empty = pd.DataFrame(columns=["path_a", "path_b", "hamming_distance", "split_a", "split_b", "cross_split"])
    summary = summarize_near_duplicate_audit(empty)
    assert summary["n_candidate_pairs"] == 0
    assert summary["n_cross_split_pairs"] == 0


def test_summarize_near_duplicate_audit_counts():
    df = pd.DataFrame({
        "path_a": ["a", "c"], "path_b": ["b", "d"], "hamming_distance": [2, 3],
        "split_a": ["train", "train"], "split_b": ["test", "train"], "cross_split": [True, False],
    })
    summary = summarize_near_duplicate_audit(df)
    assert summary["n_candidate_pairs"] == 2
    assert summary["n_cross_split_pairs"] == 1
    assert summary["min_hamming_distance"] == 2
