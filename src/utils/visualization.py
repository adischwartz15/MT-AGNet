"""Shared Matplotlib plotting helpers used by evaluation/reporting scripts.

All functions save a PNG to ``out_path`` and return the path. Kept
dependency-light (Matplotlib only) so plotting works in headless/CI
environments (the ``Agg`` backend is forced).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig: plt.Figure, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_training_curves(history: dict[str, list[float]], out_path: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    axes[0].plot(epochs, history.get("train_loss", []), label="train")
    axes[0].plot(epochs, history.get("val_loss", []), label="val")
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    if "val_age_mae" in history:
        axes[1].plot(epochs, history["val_age_mae"], label="val age MAE", color="tab:orange")
    if "val_gender_accuracy" in history:
        ax2 = axes[1].twinx()
        ax2.plot(epochs, history["val_gender_accuracy"], label="val gender acc", color="tab:green")
        ax2.set_ylabel("gender accuracy")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("age MAE")
    return _save(fig, out_path)


def plot_age_scatter(y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=8, alpha=0.4)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("True age")
    ax.set_ylabel("Predicted age (q50)")
    ax.set_title("Predicted vs true age")
    return _save(fig, out_path)


def plot_error_histogram(errors: np.ndarray, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(errors, bins=30, color="tab:blue", alpha=0.8)
    ax.set_xlabel("Prediction error (years)")
    ax.set_ylabel("Count")
    ax.set_title("Age error distribution")
    return _save(fig, out_path)


def plot_interval_coverage(bucket_labels: list[str], coverage: np.ndarray, target: float, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(bucket_labels, coverage, color="tab:purple", alpha=0.8)
    ax.axhline(target, color="red", linestyle="--", label=f"target={target:.2f}")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("q10-q90 interval coverage by age bucket")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, out_path)


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Dataset gender-label confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save(fig, out_path)


def plot_loss_balancing(history: dict[str, list[float]], out_path: str | Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history.get("age_loss", [])) + 1)
    axes[0].plot(epochs, history.get("age_loss", []), label="age loss")
    axes[0].plot(epochs, history.get("gender_loss", []), label="gender loss")
    axes[0].set_title("Per-task loss")
    axes[0].legend()
    if "effective_age_weight" in history:
        axes[1].plot(epochs, history["effective_age_weight"], label="effective age weight")
        axes[1].plot(epochs, history["effective_gender_weight"], label="effective gender weight")
        axes[1].set_title("Effective task weights")
        axes[1].legend()
    return _save(fig, out_path)


def plot_gradient_cosine_similarity(values: np.ndarray, out_path: str | Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(values, bins=30, color="tab:red", alpha=0.7)
    ax.axvline(float(np.mean(values)), color="black", linestyle="--", label=f"mean={np.mean(values):.3f}")
    ax.set_xlabel("cosine similarity(grad_age, grad_gender)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    return _save(fig, out_path)


def plot_embedding_scatter(
    coords: np.ndarray,
    labels: np.ndarray | None,
    label_names: dict[int, str] | None,
    out_path: str | Path,
    title: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    if labels is None:
        ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.6)
    else:
        unique = np.unique(labels)
        cmap = plt.get_cmap("tab10")
        for i, value in enumerate(unique):
            mask = labels == value
            name = label_names.get(int(value), str(value)) if label_names else str(value)
            ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6, label=name, color=cmap(i % 10))
        ax.legend(markerscale=2, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    return _save(fig, out_path)


def plot_robustness_curves(df, metric: str, out_path: str | Path) -> Path:
    """``df`` is a pandas DataFrame with columns corruption, severity, and ``metric``."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for corruption, group in df.groupby("corruption"):
        group = group.sort_values("severity")
        ax.plot(group["severity"], group[metric], marker="o", label=corruption)
    ax.set_xlabel("severity")
    ax.set_ylabel(metric)
    ax.set_title(f"Robustness: {metric} vs corruption severity")
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, out_path)


def plot_interval_width_by_bucket(bucket_labels: list[str], mean_widths: np.ndarray, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(bucket_labels, mean_widths, color="tab:orange", alpha=0.85)
    ax.set_ylabel("Mean q10-q90 interval width (years)")
    ax.set_title("Prediction interval width by age bucket")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, out_path)


def plot_coverage_width_tradeoff(
    coverage_before: float, width_before: float, coverage_after: float, width_after: float,
    target_coverage: float, out_path: str | Path,
) -> Path:
    """Scatter of (coverage, mean width) before vs. after conformal calibration.

    Calibration is expected to move the point toward the target coverage
    line, typically at the cost of a wider interval -- this plot makes
    that trade-off (not a free improvement) visible.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter([width_before], [coverage_before], s=120, color="tab:red", label="Before calibration", zorder=3)
    ax.scatter([width_after], [coverage_after], s=120, color="tab:green", label="After calibration", zorder=3)
    ax.annotate(
        "", xy=(width_after, coverage_after), xytext=(width_before, coverage_before),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5),
    )
    ax.axhline(target_coverage, color="black", linestyle="--", linewidth=1, label=f"target={target_coverage:.2f}")
    ax.set_xlabel("Mean interval width (years)")
    ax.set_ylabel("Empirical q10-q90 coverage")
    ax.set_title("Coverage-width trade-off: before vs. after calibration")
    ax.legend(fontsize=8)
    return _save(fig, out_path)


def plot_parameter_latency_comparison(
    labels: list[str], param_counts: list[float], latencies: list[float], out_path: str | Path,
) -> Path:
    """Dual-axis bar chart comparing parameter count and per-image latency across experiments/backbones."""
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    width = 0.35
    ax1.bar(x - width / 2, param_counts, width, color="tab:blue", label="Total parameters")
    ax1.set_ylabel("Total parameters", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)

    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, latencies, width, color="tab:orange", label="Latency (ms/image)")
    ax2.set_ylabel("Inference latency (ms/image)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_title("Parameter count vs. inference latency")
    return _save(fig, out_path)


def plot_mean_std_bar(labels: list[str], means: np.ndarray, stds: np.ndarray, metric_name: str, out_path: str | Path) -> Path:
    """Bar chart with error bars for a mean +/- std metric across seeds/experiments."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4, color="tab:purple", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name}: mean +/- std across seeds")
    return _save(fig, out_path)


def plot_pareto(
    labels: list[str], x_values: list[float], y_values: list[float],
    x_label: str, y_label: str, title: str, out_path: str | Path,
) -> Path:
    """Scatter plot with one labeled point per model -- e.g. age MAE vs. parameter count/latency."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.get_cmap("tab10")
    for i, (label, x, y) in enumerate(zip(labels, x_values, y_values)):
        ax.scatter([x], [y], s=90, color=cmap(i % 10), label=label, zorder=3)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def plot_risk_coverage_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], y_label: str, title: str, out_path: str | Path) -> Path:
    """One risk-vs-coverage line per model. ``curves`` maps model label -> (coverages, risks)."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for label, (coverages, risks) in curves.items():
        ax.plot(coverages, risks, marker="o", markersize=3, label=label)
    ax.set_xlabel("Coverage (fraction of samples accepted)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def plot_age_error_cdf(errors_by_model: dict[str, np.ndarray], out_path: str | Path) -> Path:
    """Empirical CDF of absolute age error, one curve per model."""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for label, errors in errors_by_model.items():
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(sorted_errors, cumulative, label=label)
    ax.set_xlabel("Absolute age error (years)")
    ax.set_ylabel("Cumulative fraction of samples")
    ax.set_title("CDF of absolute age error")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, out_path)


def plot_tail_error_bars(tail_rates_by_model: dict[str, dict[str, float]], out_path: str | Path) -> Path:
    """Grouped bar chart of error-tail rates (e.g. >5/>10/>15/>20 years) per model."""
    models = list(tail_rates_by_model.keys())
    thresholds = list(next(iter(tail_rates_by_model.values())).keys())
    x = np.arange(len(thresholds))
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, model in enumerate(models):
        rates = [tail_rates_by_model[model][t] for t in thresholds]
        ax.bar(x + i * width, rates, width, label=model, color=cmap(i % 10))
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(thresholds)
    ax.set_xlabel("Absolute age error threshold")
    ax.set_ylabel("Fraction of samples exceeding threshold")
    ax.set_title("Tail-error rates by model")
    ax.legend(fontsize=8)
    return _save(fig, out_path)


def save_gradcam_overlay(image_rgb: np.ndarray, heatmap: np.ndarray, out_path: str | Path, title: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(image_rgb)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.45)
    axes[1].set_title(title)
    axes[1].axis("off")
    return _save(fig, out_path)
