"""Non-destructive near-duplicate audit for a prepared split.

Exact SHA-256 duplicate removal (``src/data/validation.py::validate_dataset``)
only catches byte-identical files -- it misses resized copies, recompressed
copies, or near-identical crops of the same underlying photo. This module
flags *candidate* near-duplicate pairs using a perceptual hash (difference
hash / dHash -- no extra dependency beyond PIL/NumPy, already required) and
reports which ones land in different splits, without ever auto-deleting
anything: a false positive here would silently shrink the dataset for no
reason, so every decision about what to do with a flagged pair is left to
a human reviewing the report.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def difference_hash(image_path: str | Path, hash_size: int = 8) -> int:
    """A simple, dependency-free perceptual hash (dHash, Krawetz 2013):
    resize to ``(hash_size + 1) x hash_size`` grayscale, then hash each
    pixel to 1 if it's brighter than its right neighbor. Two visually
    similar images (including resized/recompressed/mildly-cropped copies)
    hash to a small Hamming distance; two unrelated images hash to
    approximately random (large) Hamming distance.
    """
    with Image.open(image_path) as img:
        gray = img.convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
        pixels = np.asarray(gray, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def compute_hashes(image_paths: list[str], hash_size: int = 8) -> dict[str, int | None]:
    """Compute a dHash for every path; unreadable images map to ``None``
    (excluded from pairwise comparison, never treated as a match)."""
    hashes: dict[str, int | None] = {}
    for path in image_paths:
        try:
            hashes[path] = difference_hash(path, hash_size)
        except (OSError, ValueError):
            hashes[path] = None
    return hashes


def find_near_duplicate_candidates(
    df: pd.DataFrame,
    hash_size: int = 8,
    max_hamming_distance: int = 4,
    path_column: str = "image_path",
    split_column: str = "split",
    max_pairs_reported: int = 5000,
) -> pd.DataFrame:
    """Return a DataFrame of candidate near-duplicate pairs within ``df``.

    ``max_hamming_distance`` (out of ``hash_size * hash_size`` total bits,
    e.g. 64 for the default ``hash_size=8``) is the similarity threshold --
    smaller means stricter (fewer, more-confident candidates). This is a
    **candidate list for human review**, not a verdict: a low-but-nonzero
    distance can also occur for two genuinely different photos with similar
    overall brightness/composition (e.g. two portraits under similar
    lighting), especially at a coarse ``hash_size``.

    At or below ``_EXACT_SCAN_THRESHOLD`` (500) valid images, every pair is
    compared exactly. Above it, candidates are pre-filtered by bucketing on
    the hash's top byte (only paths sharing the same top byte are compared
    against each other) to avoid a full O(n^2) scan on a large dataset --
    an **approximate** pre-filter: a genuinely near-duplicate pair whose
    hashes happen to differ in the top byte's bits will be missed above the
    threshold. A deliberate, documented trade-off for a best-effort,
    human-reviewed audit (not a certified-exhaustive one) that still runs
    in practical time on a dataset of thousands of images. Capped at
    ``max_pairs_reported`` candidate pairs overall (sorted by distance,
    closest first) so a dataset with many true near-duplicate clusters
    still produces a boundedly-sized report.
    """
    hashes = compute_hashes(df[path_column].tolist(), hash_size)
    valid_paths = [p for p, h in hashes.items() if h is not None]

    # Below this size, an exact full pairwise scan is cheap enough (and more
    # reliable -- bucketing is an approximate pre-filter that can miss a true
    # near-duplicate pair split across two buckets) that there's no reason to
    # accept the approximation at all.
    _EXACT_SCAN_THRESHOLD = 500
    if len(valid_paths) <= _EXACT_SCAN_THRESHOLD:
        buckets: dict[int, list[str]] = {0: valid_paths}
    else:
        total_bits = hash_size * hash_size  # difference_hash's bit width for a given hash_size
        top_byte_shift = max(0, total_bits - 8)
        buckets = {}
        for path in valid_paths:
            top_byte = hashes[path] >> top_byte_shift
            buckets.setdefault(top_byte, []).append(path)

    path_to_split = (
        dict(zip(df[path_column], df[split_column])) if split_column in df.columns else {}
    )

    candidates = []
    for bucket_paths in buckets.values():
        for path_a, path_b in itertools.combinations(bucket_paths, 2):
            dist = hamming_distance(hashes[path_a], hashes[path_b])
            if dist <= max_hamming_distance:
                split_a = path_to_split.get(path_a)
                split_b = path_to_split.get(path_b)
                candidates.append({
                    "path_a": path_a, "path_b": path_b, "hamming_distance": dist,
                    "split_a": split_a, "split_b": split_b, "cross_split": split_a != split_b,
                })

    candidates.sort(key=lambda c: c["hamming_distance"])
    candidates = candidates[:max_pairs_reported]
    return pd.DataFrame(candidates, columns=["path_a", "path_b", "hamming_distance", "split_a", "split_b", "cross_split"])


def summarize_near_duplicate_audit(candidates_df: pd.DataFrame) -> dict:
    """A compact summary suitable for embedding directly in the split manifest."""
    if candidates_df.empty:
        return {"n_candidate_pairs": 0, "n_cross_split_pairs": 0, "candidates_truncated": False}
    return {
        "n_candidate_pairs": int(len(candidates_df)),
        "n_cross_split_pairs": int(candidates_df["cross_split"].sum()),
        "min_hamming_distance": int(candidates_df["hamming_distance"].min()),
    }
