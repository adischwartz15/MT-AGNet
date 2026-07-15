#!/usr/bin/env python
"""CLI: build the non-parametric k-NN embedding-space baseline from a trained checkpoint.

Usage:
    python scripts/build_knn_index.py --checkpoint checkpoints/multitask_best_balanced_score.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import FaceMultiTaskDataset
from src.data.transforms import resolve_eval_transform
from src.evaluation.knn_baseline import KNNEmbeddingBaseline
from src.inference.artifacts import load_model_checkpoint
from src.utils.config import REPO_ROOT, resolve_device
from src.utils.logging import get_logger

logger = get_logger("scripts.build_knn_index")


@torch.no_grad()
def _extract_embeddings(model, dataset, device, batch_size=64):
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    age_embeds, gender_embeds, ages, age_masks, genders, gender_masks = [], [], [], [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        emb = model.encode(images)
        age_embeds.append(emb["age_embedding"].cpu().numpy())
        gender_embeds.append(emb["gender_embedding"].cpu().numpy())
        ages.append(batch["age"].numpy())
        age_masks.append(batch["age_mask"].numpy())
        genders.append(batch["gender_label"].numpy())
        gender_masks.append(batch["gender_mask"].numpy())
    return (
        np.concatenate(age_embeds), np.concatenate(gender_embeds),
        np.concatenate(ages), np.concatenate(age_masks).astype(bool),
        np.concatenate(genders), np.concatenate(gender_masks).astype(bool),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    device = resolve_device("auto")
    model, config, _ = load_model_checkpoint(args.checkpoint, device)

    splits_path = REPO_ROOT / config["paths"]["splits_dir"] / "full_metadata_with_splits.csv"
    if not splits_path.exists():
        logger.error("No prepared split found at %s.", splits_path)
        return 1
    df = pd.read_csv(splits_path)
    # Model-aware preprocessing (see src/data/transforms.py::resolve_eval_transform)
    # -- a pretrained-ResNet checkpoint's own resolved transform, never this
    # project's 128px/IMAGENET-constant default for such a model.
    train_dataset = FaceMultiTaskDataset(df[df["split"] == "train"], resolve_eval_transform(model, config))

    age_embeds, gender_embeds, ages, age_mask, genders, gender_mask = _extract_embeddings(model, train_dataset, device)

    # Age and gender adapters can output different embedding spaces; fit one
    # index per task using its own adapter's embedding.
    knn_cfg = config["knn"]
    k = args.k or knn_cfg.get("k", 15)
    age_head_cfg = config["model"]["age_head"]
    knn_kwargs = dict(
        distance_weighted=knn_cfg.get("distance_weighted", True),
        metric=knn_cfg.get("metric", "euclidean"),
        age_min=age_head_cfg.get("age_min", 0.0),
        age_max=age_head_cfg.get("age_max", 120.0),
    )
    age_knn = KNNEmbeddingBaseline(k=k, **knn_kwargs)
    age_knn.fit(age_embeds, ages, age_mask, genders, np.zeros_like(gender_mask), num_classes=config["model"]["gender_head"]["num_classes"])

    gender_knn = KNNEmbeddingBaseline(k=k, **knn_kwargs)
    gender_knn.fit(gender_embeds, ages, np.zeros_like(age_mask), genders, gender_mask, num_classes=config["model"]["gender_head"]["num_classes"])

    # Merge into a single baseline object exposing both task indices.
    combined = KNNEmbeddingBaseline(k=k, **knn_kwargs)
    combined.age_index = age_knn.age_index
    combined.age_values = age_knn.age_values
    combined.age_distance_scale = age_knn.age_distance_scale
    combined.gender_index = gender_knn.gender_index
    combined.gender_values = gender_knn.gender_values
    combined.num_classes = config["model"]["gender_head"]["num_classes"]

    index_dir = REPO_ROOT / knn_cfg.get("index_dir", "./outputs/knn")
    index_dir.mkdir(parents=True, exist_ok=True)
    out_path = index_dir / "knn_baseline.pkl"
    combined.save(out_path)
    logger.info("Saved k-NN baseline (k=%d) to %s", k, out_path)
    print(f"Saved k-NN index to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
