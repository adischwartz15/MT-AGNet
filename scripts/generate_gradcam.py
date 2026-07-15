#!/usr/bin/env python
"""CLI: generate Grad-CAM "model attention visualization" overlays for sample images.

Produces overlays for correct predictions, incorrect predictions,
low-confidence examples, the single widest-age-interval example (i.e. the
sample the model is least certain about), and (if present) blurry / partial /
robustness-corrupted images, saved to outputs/gradcam/.

Usage:
    python scripts/generate_gradcam.py --checkpoint checkpoints/multitask_best_balanced_score.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.transforms import EvalTransform
from src.evaluation.gradcam import GradCAM, resize_heatmap
from src.evaluation.robustness import apply_corruption
from src.inference.artifacts import load_model_checkpoint
from src.inference.quality import compute_quality_diagnostics
from src.utils.config import REPO_ROOT, resolve_device
from src.utils.logging import get_logger
from src.utils.visualization import save_gradcam_overlay

logger = get_logger("scripts.generate_gradcam")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-samples", type=int, default=12)
    args = parser.parse_args()

    device = resolve_device("auto")
    model, config, _ = load_model_checkpoint(args.checkpoint, device)

    splits_path = REPO_ROOT / config["paths"]["splits_dir"] / "full_metadata_with_splits.csv"
    if not splits_path.exists():
        logger.error("No prepared split found at %s.", splits_path)
        return 1
    df = pd.read_csv(splits_path)
    test_df = df[df["split"] == "test"].dropna(subset=["gender_label"]).head(args.num_samples)

    transform = EvalTransform(config["dataset"]["image_size"])
    gradcam = GradCAM(model, config["gradcam"]["target_layer"])
    confidence_threshold = config["model"]["gender_head"].get("confidence_threshold", 0.80)
    class_names = config["model"]["gender_head"]["class_names"]

    output_dir = REPO_ROOT / "outputs" / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    records = test_df.to_dict("records")

    # Cheap no-grad pre-pass to find the single widest raw q10-q90 age
    # interval among the selected samples -- the example the model itself is
    # least certain about age-wise, which the correct/incorrect/low-confidence
    # gender-label categories below don't otherwise surface.
    widest_interval_idx, widest_interval_width = None, -1.0
    with torch.no_grad():
        for i, row in enumerate(records):
            with Image.open(row["image_path"]) as img:
                image_tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
            age_out = model(image_tensor)["age_output"]
            width = float((age_out["q90"] - age_out["q10"])[0].item())
            if width > widest_interval_width:
                widest_interval_idx, widest_interval_width = i, width

    for i, row in enumerate(records):
        with Image.open(row["image_path"]) as img:
            rgb = img.convert("RGB")
            image_tensor = transform(rgb).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(image_tensor)
                probs = torch.softmax(outputs["gender_logits"], dim=-1)[0].cpu().numpy()
            predicted = int(probs.argmax())
            confidence = float(probs.max())
            true_label = int(row["gender_label"]) if row["gender_label"] == row["gender_label"] else None
            correct = true_label is not None and predicted == true_label
            category = "low_confidence" if confidence < confidence_threshold else ("correct" if correct else "incorrect")

            age_result = gradcam.generate(image_tensor.clone(), task="age")
            gender_result = gradcam.generate(image_tensor.clone(), task="gender")

            image_np = np.asarray(rgb.resize((config["dataset"]["image_size"],) * 2))
            age_heatmap = resize_heatmap(age_result["heatmap"], image_np.shape[:2][::-1])
            gender_heatmap = resize_heatmap(gender_result["heatmap"], image_np.shape[:2][::-1])

            save_gradcam_overlay(
                image_np / 255.0, age_heatmap, output_dir / f"{category}_{i}_age_attention.png",
                "Model attention visualization: age (q50)",
            )
            save_gradcam_overlay(
                image_np / 255.0, gender_heatmap, output_dir / f"{category}_{i}_gender_attention.png",
                f"Model attention visualization: gender-label ({class_names[predicted]})",
            )

            if i == widest_interval_idx:
                save_gradcam_overlay(
                    image_np / 255.0, age_heatmap, output_dir / f"widest_interval_{i}_age_attention.png",
                    f"Model attention visualization: age (widest q10-q90 interval, width={widest_interval_width:.1f})",
                )

            quality = compute_quality_diagnostics(rgb, "jpg", Path(row["image_path"]).stat().st_size)
            if quality.warnings and i < 3:
                blurred = apply_corruption(rgb, "gaussian_blur", 2.0, seed=i)
                blurred_tensor = transform(blurred).unsqueeze(0).to(device)
                blurred_result = gradcam.generate(blurred_tensor, task="age")
                blurred_heatmap = resize_heatmap(blurred_result["heatmap"], image_np.shape[:2][::-1])
                save_gradcam_overlay(
                    np.asarray(blurred.resize((config["dataset"]["image_size"],) * 2)) / 255.0,
                    blurred_heatmap, output_dir / f"blurry_{i}_age_attention.png",
                    "Model attention visualization: age (blurred input)",
                )

    logger.info("Saved Grad-CAM overlays to %s", output_dir)
    print(f"Saved Grad-CAM overlays to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
