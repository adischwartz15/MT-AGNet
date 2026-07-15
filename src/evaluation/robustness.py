"""Deterministic robustness/corruption evaluation on the held-out test set.

Corruptions are applied to PIL images *before* the standard eval
transform, so severities are comparable across corruption types. All
randomness (noise, occlusion location, crop side) is seeded per-sample
for reproducibility.
"""

from __future__ import annotations

import io
import itertools
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

CORRUPTION_NAMES = (
    "gaussian_blur", "gaussian_noise", "low_resolution", "jpeg_compression",
    "low_brightness", "high_brightness", "low_contrast", "high_contrast",
    "grayscale", "partial_occlusion", "partial_crop",
)


def gaussian_blur(image: Image.Image, sigma: float, seed: int = 0) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def gaussian_noise(image: Image.Image, std: float, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    noisy = array + rng.normal(0, std, array.shape)
    noisy = np.clip(noisy, 0.0, 1.0)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def low_resolution(image: Image.Image, scale_factor: float, seed: int = 0) -> Image.Image:
    w, h = image.size
    small = image.resize((max(1, int(w * scale_factor)), max(1, int(h * scale_factor))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def jpeg_compression(image: Image.Image, quality: int, seed: int = 0) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def low_brightness(image: Image.Image, factor: float, seed: int = 0) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


def high_brightness(image: Image.Image, factor: float, seed: int = 0) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


def low_contrast(image: Image.Image, factor: float, seed: int = 0) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(factor)


def high_contrast(image: Image.Image, factor: float, seed: int = 0) -> Image.Image:
    return ImageEnhance.Contrast(image).enhance(factor)


def grayscale(image: Image.Image, blend_factor: float, seed: int = 0) -> Image.Image:
    """Desaturate by blending toward full grayscale; blend_factor=1.0 is fully grayscale (still 3-channel)."""
    rgb = image.convert("RGB")
    gray_as_rgb = rgb.convert("L").convert("RGB")
    return Image.blend(rgb, gray_as_rgb, alpha=min(max(blend_factor, 0.0), 1.0))


def partial_occlusion(image: Image.Image, occlusion_fraction: float, seed: int = 0) -> Image.Image:
    rng = random.Random(seed)
    img = image.convert("RGB").copy()
    w, h = img.size
    box_w, box_h = int(w * occlusion_fraction ** 0.5), int(h * occlusion_fraction ** 0.5)
    x = rng.randint(0, max(0, w - box_w))
    y = rng.randint(0, max(0, h - box_h))
    array = np.asarray(img).copy()
    array[y : y + box_h, x : x + box_w, :] = 0
    return Image.fromarray(array)


def partial_crop(image: Image.Image, crop_fraction: float, seed: int = 0) -> Image.Image:
    rng = random.Random(seed)
    w, h = image.size
    side = rng.choice(["left", "right", "top", "bottom"])
    if side in ("left", "right"):
        cut = int(w * crop_fraction)
        box = (cut, 0, w, h) if side == "left" else (0, 0, w - cut, h)
    else:
        cut = int(h * crop_fraction)
        box = (0, cut, w, h) if side == "top" else (0, 0, w, h - cut)
    cropped = image.crop(box)
    return cropped.resize((w, h), Image.BILINEAR)


_CORRUPTION_FUNCS = {
    "gaussian_blur": gaussian_blur,
    "gaussian_noise": gaussian_noise,
    "low_resolution": low_resolution,
    "jpeg_compression": jpeg_compression,
    "low_brightness": low_brightness,
    "high_brightness": high_brightness,
    "low_contrast": low_contrast,
    "high_contrast": high_contrast,
    "grayscale": grayscale,
    "partial_occlusion": partial_occlusion,
    "partial_crop": partial_crop,
}


def apply_corruption(image: Image.Image, name: str, param: float, seed: int = 0) -> Image.Image:
    if name not in _CORRUPTION_FUNCS:
        raise ValueError(f"Unknown corruption '{name}', expected one of {CORRUPTION_NAMES}")
    return _CORRUPTION_FUNCS[name](image, param, seed=seed)


_DEFAULT_AGE_BUCKET_EDGES = (0, 13, 20, 35, 50, 65, 200)


def stratified_sample(
    df: pd.DataFrame, max_samples: int | None, seed: int = 42,
    age_bucket_edges: tuple[int, ...] = _DEFAULT_AGE_BUCKET_EDGES,
) -> pd.DataFrame:
    """Deterministic stratified sample over (age bucket, gender label) strata.

    Used instead of ``df.head(max_samples)`` for robustness evaluation --
    the first N rows of a prepared split CSV are not a random or
    representative sample, so truncating to them would silently bias the
    corruption evaluation toward whatever subgroup happens to sort first.
    Samples proportionally to each stratum's share of ``df`` (deterministic
    given ``seed``, so repeat runs pick the same rows). Returns ``df``
    unchanged (in its original row order) if ``max_samples`` is ``None`` or
    already covers the whole split.
    """
    if max_samples is None or max_samples >= len(df):
        return df.reset_index(drop=True)

    working = df.reset_index(drop=True).copy()
    age_bucket = pd.cut(working["age"], bins=list(age_bucket_edges), right=False)
    gender = working["gender_label"].astype("object").where(working["gender_label"].notna(), "missing")
    strata = list(zip(age_bucket.astype(str), gender.astype(str)))

    rng = np.random.default_rng(seed)
    frac = max_samples / len(working)
    parts = []
    working["_stratum"] = strata
    for _, group in working.groupby("_stratum", sort=True):
        n = min(int(round(len(group) * frac)), len(group))
        if n <= 0:
            continue
        idx = rng.choice(group.index.to_numpy(), size=n, replace=False)
        parts.append(group.loc[sorted(idx)])
    if not parts:
        return working.drop(columns="_stratum").iloc[0:0]
    sampled = pd.concat(parts).drop(columns="_stratum")
    return sampled.sort_index().reset_index(drop=True)


def iter_corruption_configs(robustness_cfg: dict):
    """Yield (corruption_name, severity_level, param_value) tuples from the config."""
    for name, spec in robustness_cfg["corruptions"].items():
        for severity, param in zip(spec["severities"], spec["params"]):
            yield name, severity, param


def corruption_summary(robustness_cfg: dict) -> dict:
    """Programmatic corruption-count summary -- computed directly from
    ``configs/robustness.yaml`` (never a hand-maintained doc claim like "11
    corruptions" that can silently drift out of sync with the actual
    config). Callers (``scripts/run_robustness.py``'s saved summary,
    documentation generation) should read this instead of hardcoding a count.
    """
    corruption_names = sorted(robustness_cfg["corruptions"])
    n_conditions = sum(1 for _ in iter_corruption_configs(robustness_cfg))
    severities_per_type = {
        name: len(spec["severities"]) for name, spec in robustness_cfg["corruptions"].items()
    }
    return {
        "n_corruption_types": len(corruption_names),
        "corruption_type_names": corruption_names,
        "n_total_conditions": n_conditions,  # sum over types of (severities per type) -- not simply n_types * a fixed severity count if they differ
        "severities_per_type": severities_per_type,
    }


def _predict_batch(model, images_tensor, device, gender_confidence_threshold: float):
    """Run one forward pass and return numpy prediction arrays for a batch of images."""
    import torch

    model.eval()
    with torch.no_grad():
        images_tensor = images_tensor.to(device)
        outputs = model(images_tensor)
        probs = torch.softmax(outputs["gender_logits"], dim=-1).cpu().numpy()
    q10 = outputs["age_output"]["q10"].cpu().numpy()
    q50 = outputs["age_output"]["q50"].cpu().numpy()
    q90 = outputs["age_output"]["q90"].cpu().numpy()
    predicted_class = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    abstain = confidence < gender_confidence_threshold
    return {
        "q10": q10, "q50": q50, "q90": q90,
        "predicted_class": predicted_class, "confidence": confidence, "abstain": abstain,
    }


def evaluate_condition(
    model,
    df,
    transform,
    device: str,
    gender_confidence_threshold: float,
    corruption_name: str | None,
    severity: int | None,
    param: float | None,
    seed: int,
    batch_size: int = 32,
    calibration_offset: float | None = None,
):
    """Run the model over ``df`` with an optional corruption applied, returning a metrics dict.

    ``corruption_name=None`` evaluates the clean (uncorrupted) baseline.
    ``calibration_offset``, when given, is the *fixed* conformal offset
    already fit on the clean calibration split (see
    ``src/evaluation/calibration.py:fit_conformal_offset``) -- it is only
    ever applied here, never refit on corrupted data, since re-fitting per
    corruption would answer a different question ("how wide would a
    calibration fit on this corruption need to be") than the one this
    evaluation asks ("how much does the clean-fit calibration's coverage
    guarantee degrade under distribution shift"). Both raw and
    (if ``calibration_offset`` is given) calibrated coverage/width are
    reported side by side so that degradation is visible in both.
    """
    import torch
    from PIL import Image

    from src.evaluation.calibration import apply_conformal_offset
    from src.evaluation.metrics import (
        abstention_rate, age_mae, age_rmse, gender_accuracy, interval_coverage, mean_interval_width,
    )

    all_q10, all_q50, all_q90 = [], [], []
    all_pred_class, all_confidence, all_abstain = [], [], []
    ages, genders, age_valid, gender_valid = [], [], [], []

    rows = df.to_dict("records")
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        tensors = []
        for i, row in enumerate(batch_rows):
            with Image.open(row["image_path"]) as img:
                img = img.convert("RGB")
                if corruption_name is not None:
                    img = apply_corruption(img, corruption_name, param, seed=seed + start + i)
                tensors.append(transform(img))
            ages.append(row["age"])
            genders.append(row["gender_label"])
            age_valid.append(row["age"] == row["age"])  # not NaN
            gender_valid.append(row["gender_label"] == row["gender_label"])
        batch_tensor = torch.stack(tensors)
        preds = _predict_batch(model, batch_tensor, device, gender_confidence_threshold)
        all_q10.append(preds["q10"]); all_q50.append(preds["q50"]); all_q90.append(preds["q90"])
        all_pred_class.append(preds["predicted_class"])
        all_confidence.append(preds["confidence"])
        all_abstain.append(preds["abstain"])

    q10 = np.concatenate(all_q10); q50 = np.concatenate(all_q50); q90 = np.concatenate(all_q90)
    pred_class = np.concatenate(all_pred_class)
    confidence = np.concatenate(all_confidence)
    abstain = np.concatenate(all_abstain)
    ages = np.array(ages, dtype=np.float64)
    genders = np.array(genders, dtype=np.float64)
    age_valid = np.array(age_valid)
    gender_valid = np.array(gender_valid)

    metrics = {
        "corruption": corruption_name or "clean",
        "severity": severity or 0,
        "param": param,
        "n_samples": len(rows),
    }
    if age_valid.any():
        metrics["age_mae"] = age_mae(ages[age_valid], q50[age_valid])
        metrics["age_rmse"] = age_rmse(ages[age_valid], q50[age_valid])
        metrics["interval_coverage"] = interval_coverage(ages[age_valid], q10[age_valid], q90[age_valid])
        metrics["mean_interval_width"] = mean_interval_width(q10[age_valid], q90[age_valid])
        if calibration_offset is not None:
            q10_cal, q90_cal = apply_conformal_offset(q10[age_valid], q90[age_valid], calibration_offset)
            metrics["interval_coverage_calibrated"] = interval_coverage(ages[age_valid], q10_cal, q90_cal)
            metrics["mean_interval_width_calibrated"] = mean_interval_width(q10_cal, q90_cal)
    if gender_valid.any():
        combined_abstain = abstain[gender_valid]
        metrics["gender_accuracy"] = gender_accuracy(
            genders[gender_valid], pred_class[gender_valid], combined_abstain
        )
        metrics["abstention_rate"] = abstention_rate(abstain[gender_valid])
        metrics["mean_confidence"] = float(confidence[gender_valid].mean())
    return metrics


_DEGRADATION_METRICS = (
    "age_mae", "age_rmse", "interval_coverage", "mean_interval_width",
    "interval_coverage_calibrated", "mean_interval_width_calibrated",
    "gender_accuracy", "abstention_rate",
)


def compute_degradation(results_df: pd.DataFrame, metrics: tuple[str, ...] = _DEGRADATION_METRICS) -> pd.DataFrame:
    """Add ``{metric}_delta`` and ``{metric}_pct_change`` columns relative to the clean row.

    ``delta = corrupted_value - clean_value``; ``pct_change`` is
    ``delta / clean_value * 100`` (``NaN`` when the clean value is 0, to
    avoid a divide-by-zero fabricating an infinite/undefined percentage).
    The clean row itself gets ``delta=0`` / ``pct_change=0`` for every
    metric, since it is being compared against itself.
    """
    df = results_df.copy()
    clean_rows = df[df["corruption"] == "clean"]
    if clean_rows.empty:
        raise ValueError("results_df has no 'clean' baseline row to compute degradation against")
    clean_row = clean_rows.iloc[0]

    for metric in metrics:
        if metric not in df.columns:
            continue
        clean_value = clean_row.get(metric)
        if clean_value is None or (isinstance(clean_value, float) and clean_value != clean_value):
            df[f"{metric}_delta"] = float("nan")
            df[f"{metric}_pct_change"] = float("nan")
            continue
        df[f"{metric}_delta"] = df[metric] - clean_value
        df[f"{metric}_pct_change"] = (
            (df[metric] - clean_value) / clean_value * 100.0 if clean_value != 0 else float("nan")
        )
    return df


def build_robustness_diff_table(results_by_model: dict[str, pd.DataFrame], metrics: tuple[str, ...] = _DEGRADATION_METRICS) -> pd.DataFrame:
    """One row per (corruption, severity, model pair): every pairwise model comparison.

    With exactly two models this produces the same single-pair columns as
    before (backward compatible). With more than two models (e.g.
    SimpleCNN, PlainDeep18NoSkip, Custom ResNet-18) this produces *all*
    ``C(n, 2)`` pairwise comparisons -- concatenated, with an added
    ``comparison`` column identifying which pair each row is -- rather
    than silently comparing only the first two models by dict insertion
    order and dropping the rest. In particular this guarantees SimpleCNN
    vs ResNet, PlainDeep18NoSkip vs ResNet, and SimpleCNN vs
    PlainDeep18NoSkip are all present when all three models are supplied.
    """
    names = list(results_by_model)
    if len(names) < 2:
        raise ValueError("build_robustness_diff_table needs at least two models to compare")

    rows = []
    for name_a, name_b in itertools.combinations(names, 2):
        df_a = results_by_model[name_a].set_index(["corruption", "severity"])
        df_b = results_by_model[name_b].set_index(["corruption", "severity"])
        for key in df_a.index:
            if key not in df_b.index:
                continue
            row = {"corruption": key[0], "severity": key[1], "comparison": f"{name_b}_vs_{name_a}"}
            for metric in metrics:
                if metric not in df_a.columns or metric not in df_b.columns:
                    continue
                value_a, value_b = df_a.loc[key, metric], df_b.loc[key, metric]
                row[f"{name_a}_{metric}"] = value_a
                row[f"{name_b}_{metric}"] = value_b
                row[f"diff_{metric}_({name_b}_minus_{name_a})"] = value_b - value_a
            rows.append(row)
    return pd.DataFrame(rows)
