"""Deterministic train/validation/calibration/test splitting with subject-level leakage prevention.

Four-way split protocol, each split used for exactly one purpose so no
data ever informs a decision it shouldn't:

* ``train``       -- model fitting (gradient updates).
* ``validation``  -- early stopping and checkpoint selection only
  (``src/training/trainer.py``). Never used to fit conformal intervals or
  to report final numbers.
* ``calibration`` -- fitting split-conformal prediction intervals only
  (``src/evaluation/calibration.py`` / ``scripts/calibrate.py``). Never
  used for early stopping or final evaluation.
* ``test``        -- final evaluation only, touched once per checkpoint.

When a ``subject_id`` column is available and ``subject_level_if_available``
is True, splitting is done at the subject (group) level so the same
person's images never appear in more than one split. Otherwise, splitting
falls back to a per-row random split. All splitting is seeded for
reproducibility and is saved to ``data/splits/`` so every experiment in
the ablation suite can reuse the identical split.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "validation", "calibration", "test")

# Fixed before any final result is observed (see docs/reproducibility.md
# "Locked stratified split"). Matches src/evaluation/metrics.py's own
# age_uncertainty_by_bucket default bucket boundaries (0/10/20/.../80),
# extended to a closed upper edge of 121 so age==120 (this project's
# age_max) falls inside the last bin rather than being excluded by a
# half-open [lo, hi) bin ending at 120.
AGE_BIN_EDGES: tuple[int, ...] = (0, 10, 20, 30, 40, 50, 60, 70, 80, 121)


def age_bin_label(age: float, edges: tuple[int, ...] = AGE_BIN_EDGES) -> str:
    """The half-open age bin ``[lo, hi)`` (closed at the top for the last
    bin) containing ``age``, as a stable string label -- e.g. ``"20-29"``.
    Ages outside ``[edges[0], edges[-1])`` clamp to the nearest edge bin
    (UTKFace ages are documented to lie in ``[0, 116]``, comfortably inside
    the default edges, but this never raises on a stray out-of-range value)."""
    clamped = min(max(age, edges[0]), edges[-1] - 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= clamped < hi:
            return f"{lo}-{hi - 1}"
    return f"{edges[-2]}-{edges[-1] - 1}"  # unreachable given the clamp above; defensive only


def _normalize_fractions(
    train_fraction: float, validation_fraction: float, calibration_fraction: float, test_fraction: float
) -> tuple[float, float, float, float]:
    total = train_fraction + validation_fraction + calibration_fraction + test_fraction
    if abs(total - 1.0) > 1e-6:
        logger.warning("Split fractions sum to %.4f, renormalizing to 1.0", total)
        train_fraction, validation_fraction, calibration_fraction, test_fraction = (
            train_fraction / total,
            validation_fraction / total,
            calibration_fraction / total,
            test_fraction / total,
        )
    return train_fraction, validation_fraction, calibration_fraction, test_fraction


def split_dataframe(
    df: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.15,
    calibration_fraction: float = 0.10,
    test_fraction: float = 0.15,
    seed: int = 42,
    subject_level_if_available: bool = True,
) -> pd.DataFrame:
    """Return ``df`` with an added ``split`` column in ``SPLIT_NAMES``.

    If ``df`` already has a non-null ``split`` column (e.g. supplied by the
    dataset itself via a CSV split column), it is respected and returned
    unchanged -- in that case it is the caller's responsibility to ensure
    it already distinguishes calibration from validation.
    """
    if "split" in df.columns and df["split"].notna().all():
        logger.info("Using pre-existing split column from dataset metadata")
        return df

    train_fraction, validation_fraction, calibration_fraction, test_fraction = _normalize_fractions(
        train_fraction, validation_fraction, calibration_fraction, test_fraction
    )
    fractions = [train_fraction, validation_fraction, calibration_fraction, test_fraction]
    rng = np.random.default_rng(seed)

    has_subjects = subject_level_if_available and "subject_id" in df.columns and df["subject_id"].notna().any()

    df = df.copy()
    if has_subjects:
        subjects = df["subject_id"].dropna().unique()
        rng.shuffle(subjects)
        n = len(subjects)
        n_train = int(round(n * train_fraction))
        n_validation = int(round(n * validation_fraction))
        n_calibration = int(round(n * calibration_fraction))
        train_subjects = set(subjects[:n_train])
        validation_subjects = set(subjects[n_train : n_train + n_validation])
        calibration_subjects = set(subjects[n_train + n_validation : n_train + n_validation + n_calibration])

        def _assign(subject_id):
            if subject_id in train_subjects:
                return "train"
            if subject_id in validation_subjects:
                return "validation"
            if subject_id in calibration_subjects:
                return "calibration"
            return "test"

        # Rows without a subject_id fall back to independent random assignment.
        no_subject_mask = df["subject_id"].isna()
        df["split"] = df["subject_id"].map(_assign)
        if no_subject_mask.any():
            n_no_subject = int(no_subject_mask.sum())
            assignments = rng.choice(list(SPLIT_NAMES), size=n_no_subject, p=fractions)
            df.loc[no_subject_mask, "split"] = assignments
        logger.info("Subject-level split across %d unique subjects", n)
    else:
        n = len(df)
        indices = rng.permutation(n)
        n_train = int(round(n * train_fraction))
        n_validation = int(round(n * validation_fraction))
        n_calibration = int(round(n * calibration_fraction))
        split_labels = np.empty(n, dtype=object)
        split_labels[indices[:n_train]] = "train"
        split_labels[indices[n_train : n_train + n_validation]] = "validation"
        split_labels[indices[n_train + n_validation : n_train + n_validation + n_calibration]] = "calibration"
        split_labels[indices[n_train + n_validation + n_calibration :]] = "test"
        df["split"] = split_labels
        logger.info("Row-level random split (no usable subject_id column found)")

    return df


def _largest_remainder_allocate(n: int, fractions: list[float]) -> list[int]:
    """Deterministically allocate ``n`` indistinguishable items across
    ``len(fractions)`` buckets proportional to ``fractions``, using the
    largest-remainder (Hamilton) method: floor each bucket's exact share,
    then hand out the leftover items one at a time to the buckets with the
    largest fractional remainder. Unlike naive per-bucket ``round()``,
    this always allocates the full ``n`` (the bucket counts sum exactly to
    ``n``) and is a deterministic function of ``(n, fractions)`` alone --
    no randomness, so the same stratum always splits the same way.
    """
    raw = [n * f for f in fractions]
    base = [int(x) for x in raw]
    remainder = n - sum(base)
    order = sorted(range(len(fractions)), key=lambda i: (raw[i] - base[i], -i), reverse=True)
    for i in range(remainder):
        base[order[i % len(fractions)]] += 1
    return base


def stratified_split_dataframe(
    df: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.15,
    calibration_fraction: float = 0.10,
    test_fraction: float = 0.15,
    seed: int = 42,
    subject_level_if_available: bool = True,
    age_bin_edges: tuple[int, ...] = AGE_BIN_EDGES,
) -> tuple[pd.DataFrame, dict]:
    """Deterministic 4-way split stratified by age-bin x gender-label.

    Each stratum (one age bin x one gender label, or ``"unknown"`` for a
    missing age/gender value -- rows are never dropped for this) is split
    independently via :func:`_largest_remainder_allocate`, so the overall
    train/validation/calibration/test proportions are preserved *within*
    every stratum, not just in aggregate -- a naive single global shuffle
    can (and for a skewed age/gender distribution, will) leave some strata
    almost entirely in one split.

    When ``subject_id`` is available (and ``subject_level_if_available``),
    stratification and allocation happen at the *subject* level (each
    subject's own age-bin/gender stratum, by that subject's most common
    row-level stratum), so no subject's images cross a split boundary --
    exactly the same leakage guarantee :func:`split_dataframe` provides,
    plus stratification. Rows with no ``subject_id`` fall back to row-level
    stratified allocation for those rows only.

    Returns ``(df_with_split_column, stratification_report)``. The report
    records exact per-stratum counts (before and after allocation) and
    explicitly flags any stratum where a split ended up with zero rows
    despite the stratum itself being non-empty (mathematically possible for
    a very small stratum under a very small target fraction) -- reported,
    never silently hidden.
    """
    train_fraction, validation_fraction, calibration_fraction, test_fraction = _normalize_fractions(
        train_fraction, validation_fraction, calibration_fraction, test_fraction
    )
    fractions = [train_fraction, validation_fraction, calibration_fraction, test_fraction]
    rng = np.random.default_rng(seed)

    df = df.copy()
    age_bin = df["age"].apply(lambda a: age_bin_label(a, age_bin_edges) if pd.notna(a) else "unknown")
    gender_stratum = df["gender_label"].apply(lambda g: str(g) if pd.notna(g) else "unknown")
    stratum = age_bin.astype(str) + "|" + gender_stratum.astype(str)

    has_subjects = subject_level_if_available and "subject_id" in df.columns and df["subject_id"].notna().any()

    split_labels = pd.Series(index=df.index, dtype=object)
    stratum_report: dict[str, dict] = {}
    zero_allocation_warnings: list[str] = []

    def _allocate_group(group_ids: list, group_stratum_label: str) -> None:
        ids = list(group_ids)
        rng.shuffle(ids)
        counts = _largest_remainder_allocate(len(ids), fractions)
        stratum_report[group_stratum_label] = {
            "n": len(ids), "train": counts[0], "validation": counts[1],
            "calibration": counts[2], "test": counts[3],
        }
        for split_name, count in zip(SPLIT_NAMES, counts):
            if count == 0 and len(ids) > 0:
                zero_allocation_warnings.append(f"stratum={group_stratum_label!r} split={split_name!r} n_stratum={len(ids)}")
        cursor = 0
        for split_name, count in zip(SPLIT_NAMES, counts):
            for idx in ids[cursor : cursor + count]:
                split_labels.loc[idx] = split_name
            cursor += count

    if has_subjects:
        no_subject_mask = df["subject_id"].isna()
        with_subject = df.loc[~no_subject_mask]
        subject_stratum = stratum.loc[~no_subject_mask].groupby(with_subject["subject_id"]).agg(
            lambda s: s.mode().iloc[0]
        )
        for group_label, subject_ids in subject_stratum.groupby(subject_stratum).groups.items():
            row_ids = with_subject.index[with_subject["subject_id"].isin(subject_ids)].tolist()
            # Allocate at the SUBJECT level (never split a subject's rows
            # across splits), then assign every row of an allocated subject
            # to that subject's split.
            subject_id_list = list(subject_ids)
            rng.shuffle(subject_id_list)
            counts = _largest_remainder_allocate(len(subject_id_list), fractions)
            stratum_report[group_label] = {
                "n_subjects": len(subject_id_list), "train": counts[0], "validation": counts[1],
                "calibration": counts[2], "test": counts[3],
            }
            for split_name, count in zip(SPLIT_NAMES, counts):
                if count == 0 and len(subject_id_list) > 0:
                    zero_allocation_warnings.append(
                        f"stratum={group_label!r} split={split_name!r} n_subjects={len(subject_id_list)}"
                    )
            cursor = 0
            for split_name, count in zip(SPLIT_NAMES, counts):
                for sid in subject_id_list[cursor : cursor + count]:
                    rows_for_subject = with_subject.index[with_subject["subject_id"] == sid]
                    split_labels.loc[rows_for_subject] = split_name
                cursor += count

        if no_subject_mask.any():
            for group_label, idx in stratum.loc[no_subject_mask].groupby(stratum.loc[no_subject_mask]).groups.items():
                _allocate_group(list(idx), f"{group_label} (no subject_id)")
        logger.info("Subject-level stratified split across %d strata", len(stratum_report))
    else:
        for group_label, idx in stratum.groupby(stratum).groups.items():
            _allocate_group(list(idx), group_label)
        logger.info("Row-level stratified split across %d strata", len(stratum_report))

    df["split"] = split_labels
    if zero_allocation_warnings:
        logger.warning(
            "%d (stratum, split) pair(s) received zero rows despite a non-empty stratum "
            "(unavoidable for a very small stratum under a small target fraction): %s",
            len(zero_allocation_warnings), zero_allocation_warnings,
        )

    report = {
        "age_bin_edges": list(age_bin_edges),
        "stratified_by": "age_bin_x_gender_label" if not has_subjects else "subject_level_age_bin_x_gender_label",
        "n_strata": len(stratum_report),
        "stratum_counts": stratum_report,
        "zero_allocation_warnings": zero_allocation_warnings,
    }
    return df, report


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Raise if any image path or (when available) subject_id spans multiple splits."""
    dup_paths = df.groupby("image_path")["split"].nunique()
    leaking_paths = dup_paths[dup_paths > 1]
    if len(leaking_paths) > 0:
        raise ValueError(f"Data leakage: {len(leaking_paths)} image paths appear in multiple splits")

    if "subject_id" in df.columns and df["subject_id"].notna().any():
        subj_df = df.dropna(subset=["subject_id"])
        dup_subjects = subj_df.groupby("subject_id")["split"].nunique()
        leaking_subjects = dup_subjects[dup_subjects > 1]
        if len(leaking_subjects) > 0:
            raise ValueError(f"Data leakage: {len(leaking_subjects)} subjects appear in multiple splits")
