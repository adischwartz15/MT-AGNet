"""Tests for src/data/split_utils.py::stratified_split_dataframe (T5) --
deterministic age-bin x gender-label stratified 4-way splitting, with and
without subject-level grouping. Synthetic data only, no real UTKFace.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.split_utils import (
    AGE_BIN_EDGES,
    SPLIT_NAMES,
    age_bin_label,
    assert_no_leakage,
    stratified_split_dataframe,
)


def _synthetic_df(n=800, seed=0, with_subjects=False, subjects_per_row=1):
    rng = np.random.default_rng(seed)
    ages = rng.integers(0, 90, size=n)
    genders = rng.integers(0, 2, size=n)
    df = pd.DataFrame({
        "image_path": [f"img_{i}.jpg" for i in range(n)],
        "age": ages.astype(float),
        "gender_label": genders,
    })
    if with_subjects:
        # subjects_per_row images per subject (same age/gender, as in UTKFace
        # where repeated photos of one "subject" would share label metadata).
        n_subjects = n // subjects_per_row
        subject_ids = np.repeat(np.arange(n_subjects), subjects_per_row)[:n]
        df["subject_id"] = subject_ids
    return df


# -- age_bin_label ------------------------------------------------------------------


def test_age_bin_label_basic():
    assert age_bin_label(5) == "0-9"
    assert age_bin_label(25) == "20-29"
    assert age_bin_label(120) == "80-120"  # clamped into the last bin
    assert age_bin_label(0) == "0-9"


def test_age_bin_label_covers_all_default_edges():
    labels = {age_bin_label(a) for a in range(0, 121)}
    expected_bins = len(AGE_BIN_EDGES) - 1
    assert len(labels) == expected_bins


# -- row-level stratified split -------------------------------------------------------


def test_every_row_gets_a_split_assigned():
    df = _synthetic_df()
    split_df, _ = stratified_split_dataframe(df, subject_level_if_available=False, seed=1)
    assert split_df["split"].isna().sum() == 0
    assert set(split_df["split"].unique()) <= set(SPLIT_NAMES)


def test_split_proportions_approximately_match_targets():
    df = _synthetic_df(n=4000, seed=2)
    split_df, _ = stratified_split_dataframe(
        df, train_fraction=0.6, validation_fraction=0.15, calibration_fraction=0.10, test_fraction=0.15,
        subject_level_if_available=False, seed=3,
    )
    counts = split_df["split"].value_counts(normalize=True)
    assert abs(counts["train"] - 0.6) < 0.02
    assert abs(counts["validation"] - 0.15) < 0.02
    assert abs(counts["calibration"] - 0.10) < 0.02
    assert abs(counts["test"] - 0.15) < 0.02


def test_split_proportions_hold_within_each_stratum_not_just_globally():
    """The key property stratification buys: even a small/skewed stratum
    gets approximately the right proportions, not just the dataset overall."""
    rng = np.random.default_rng(7)
    # Heavily skew: most rows are young+male, a small stratum is old+female.
    ages = np.concatenate([rng.integers(0, 10, size=900), rng.integers(80, 100, size=100)])
    genders = np.concatenate([np.zeros(900, dtype=int), np.ones(100, dtype=int)])
    df = pd.DataFrame({
        "image_path": [f"img_{i}.jpg" for i in range(1000)], "age": ages.astype(float), "gender_label": genders,
    })
    split_df, report = stratified_split_dataframe(df, subject_level_if_available=False, seed=4)
    old_female = split_df[(split_df["age"] >= 80) & (split_df["gender_label"] == 1)]
    assert len(old_female) == 100
    train_frac = (old_female["split"] == "train").mean()
    assert abs(train_frac - 0.6) < 0.1  # still approximately on-target even for the small stratum


def test_deterministic_same_seed_same_split():
    df = _synthetic_df(seed=5)
    a, _ = stratified_split_dataframe(df, subject_level_if_available=False, seed=42)
    b, _ = stratified_split_dataframe(df, subject_level_if_available=False, seed=42)
    assert (a["split"] == b["split"]).all()


def test_different_seed_gives_different_split():
    df = _synthetic_df(n=500, seed=6)
    a, _ = stratified_split_dataframe(df, subject_level_if_available=False, seed=1)
    b, _ = stratified_split_dataframe(df, subject_level_if_available=False, seed=2)
    assert not (a["split"] == b["split"]).all()


def test_missing_age_or_gender_goes_to_unknown_stratum_not_dropped():
    df = _synthetic_df(n=200, seed=8)
    df.loc[0:10, "age"] = np.nan
    df.loc[20:30, "gender_label"] = np.nan
    split_df, report = stratified_split_dataframe(df, subject_level_if_available=False, seed=9)
    assert len(split_df) == len(df)  # no rows dropped
    assert split_df["split"].isna().sum() == 0
    assert any("unknown" in k for k in report["stratum_counts"])


# -- subject-level stratified split ---------------------------------------------------


def test_subject_level_split_has_no_leakage():
    df = _synthetic_df(n=1000, with_subjects=True, subjects_per_row=4, seed=10)
    split_df, report = stratified_split_dataframe(df, subject_level_if_available=True, seed=11)
    assert_no_leakage(split_df)  # raises if any subject_id spans multiple splits
    assert report["stratified_by"] == "subject_level_age_bin_x_gender_label"


def test_subject_level_split_covers_every_row():
    df = _synthetic_df(n=400, with_subjects=True, subjects_per_row=2, seed=12)
    split_df, _ = stratified_split_dataframe(df, subject_level_if_available=True, seed=13)
    assert split_df["split"].isna().sum() == 0
    assert len(split_df) == len(df)


def test_subject_level_falls_back_to_row_level_for_rows_without_subject_id():
    df = _synthetic_df(n=300, with_subjects=True, subjects_per_row=3, seed=14)
    # Some rows have no subject_id at all.
    df.loc[0:20, "subject_id"] = np.nan
    split_df, _ = stratified_split_dataframe(df, subject_level_if_available=True, seed=15)
    assert split_df["split"].isna().sum() == 0


# -- report structure -----------------------------------------------------------------


def test_report_contains_required_fields():
    df = _synthetic_df(n=200, seed=16)
    _, report = stratified_split_dataframe(df, subject_level_if_available=False, seed=17)
    assert report["age_bin_edges"] == list(AGE_BIN_EDGES)
    assert "n_strata" in report
    assert "stratum_counts" in report
    assert "zero_allocation_warnings" in report
    assert report["n_strata"] == len(report["stratum_counts"])


def test_zero_allocation_reported_not_silently_hidden():
    """A stratum with only 1 row and a small target fraction (e.g.
    calibration=0.1) will legitimately get 0 rows in some splits -- this
    must be recorded in the report, not silently ignored."""
    df = pd.DataFrame({
        "image_path": ["a.jpg", "b.jpg", "c.jpg"], "age": [45.0, 45.0, 45.0], "gender_label": [0, 0, 0],
    })
    _, report = stratified_split_dataframe(
        df, train_fraction=0.9, validation_fraction=0.05, calibration_fraction=0.03, test_fraction=0.02,
        subject_level_if_available=False, seed=18,
    )
    assert len(report["zero_allocation_warnings"]) > 0
