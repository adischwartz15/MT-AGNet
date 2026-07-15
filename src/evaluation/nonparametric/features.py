"""Feature extraction for the two non-parametric baseline pipelines
(final-run hardening T4):

* Pipeline 1 (raw/PCA): flattened, standardized raw pixels. Tests whether
  simple pixel-distance similarity is sufficient -- a genuinely "unlearned"
  feature, never touching any trained model.
* Pipeline 2 (frozen backbone): a frozen, ImageNet-pretrained backbone's
  pooled features (no adapters, no fine-tuning). Tests whether a generic
  pretrained representation alone is sufficient. Reuses
  ``src/models/pretrained_resnet.py`` purely as a frozen feature extractor
  (adapters disabled, backbone frozen) -- never task-fine-tuned embeddings,
  which is why this project's own trained multi-task embeddings are
  explicitly NOT used as the "fair"/"unlearned" baseline here (they may
  still be analyzed separately, but must be labelled
  "post-training embedding-space analysis", never as this baseline).

Both extractors return ``(features, sample_ids)`` with deterministic
ordering matching a shuffle=False DataLoader over the dataset's own row
order -- callers use this to align features 1:1 with age/gender labels.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import FaceMultiTaskDataset
from src.data.transforms import EvalTransform

# Standard deterministic raw-pixel preprocessing: resize + center-crop (no
# augmentation), IMAGENET_MEAN/STD normalization is NOT applied here --
# StandardScaler (fit train-only) does the actual standardization for this
# pipeline, on the flattened pixel vector itself, per the mission's
# documented raw/PCA pipeline order.
RAW_PIXEL_IMAGE_SIZE = 64  # small enough that flatten+PCA is tractable; still resolves visible facial structure


def extract_raw_pixel_features(df: pd.DataFrame, image_size: int = RAW_PIXEL_IMAGE_SIZE, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Flattened raw-pixel feature matrix for Pipeline 1 (raw/PCA).

    Uses a plain deterministic resize+center-crop (no learned model, no
    augmentation) at ``image_size`` -- this project's own
    ``EvalTransform`` with no special normalization constants beyond the
    fixed IMAGENET_MEAN/STD scale (harmless here: any fixed affine
    transform of the pixel values is removed again by the StandardScaler
    fit downstream on the flattened vector).
    """
    transform = EvalTransform(image_size)
    dataset = FaceMultiTaskDataset(df, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    chunks = []
    for batch in loader:
        images = batch["image"]  # [B, 3, H, W]
        chunks.append(images.reshape(images.shape[0], -1).numpy())
    features = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 3 * image_size * image_size))
    sample_ids = df["image_path"].to_numpy()
    return features, sample_ids


@torch.no_grad()
def extract_frozen_backbone_features(
    df: pd.DataFrame, model_id: str = "resnet18", pretrained_source: str = "imagenet1k_v1",
    device: str = "cpu", batch_size: int = 64, pretrained: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Frozen, ImageNet-pretrained backbone feature matrix for Pipeline 2.

    Constructs ``PretrainedResNetFaceOnlyMultiTask`` with adapters disabled
    and the backbone frozen, uses ONLY ``model.backbone`` (the pooled
    avg-pool feature, adapters/heads never touched) with the backbone's own
    OFFICIAL preprocessing (``model.build_transforms()`` -- input size,
    mean/std, interpolation, crop_pct). This is never fine-tuned: the
    backbone is frozen for the entire call, and this function never trains
    anything.

    ``pretrained=False`` (real ImageNet weights skipped) exists only for
    offline tests -- every real run of this baseline must use the default
    ``pretrained=True``, or Pipeline 2 stops testing what it claims to.
    """
    from src.models.pretrained_resnet import build_pretrained_resnet_model

    config = {
        "model": {
            "pretrained_resnet": {"model_id": model_id, "pretrained": pretrained, "pretrained_source": pretrained_source},
            "adapters": {"enabled": False},
            "age_head": {"hidden_dim": 8, "age_min": 0, "age_max": 120},
            "gender_head": {"hidden_dim": 8, "num_classes": 2},
            "loss_balancing": {"mode": "fixed"},
        }
    }
    model = build_pretrained_resnet_model(config)
    model.freeze_backbone()
    model.eval()
    model.to(device)

    _, eval_transform = model.build_transforms()
    dataset = FaceMultiTaskDataset(df, eval_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    chunks = []
    for batch in loader:
        images = batch["image"].to(device)
        pooled = model.backbone(images)  # fc replaced with Identity -> pooled embedding directly
        chunks.append(pooled.cpu().numpy())
    features = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, model.embedding_dim))
    sample_ids = df["image_path"].to_numpy()
    return features, sample_ids


FEATURE_EXTRACTORS: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
    "raw_pca": extract_raw_pixel_features,
    "frozen_backbone": extract_frozen_backbone_features,
}
