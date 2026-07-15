"""PyTorch ``Dataset`` implementations for multi-task training and pretraining.

Samples may have age only, gender_label only, both, or (rarely, if
upstream validation missed it) neither. Missing labels are represented by
a placeholder value plus a boolean mask so the training loop can compute
masked losses that simply skip unavailable labels for a given sample.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FaceMultiTaskDataset(Dataset):
    """Supervised dataset yielding image, age (+mask), and gender label (+mask)."""

    def __init__(self, df: pd.DataFrame, transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        with Image.open(row["image_path"]) as img:
            image = self.transform(img.convert("RGB"))

        age_valid = pd.notna(row["age"])
        gender_valid = pd.notna(row["gender_label"])

        return {
            "image": image,
            "age": torch.tensor(float(row["age"]) if age_valid else 0.0, dtype=torch.float32),
            "age_mask": torch.tensor(age_valid, dtype=torch.bool),
            "gender_label": torch.tensor(int(row["gender_label"]) if gender_valid else 0, dtype=torch.long),
            "gender_mask": torch.tensor(gender_valid, dtype=torch.bool),
            "index": idx,
        }

    def image_path(self, idx: int) -> str:
        return self.df.iloc[idx]["image_path"]


class SimCLRPretrainDataset(Dataset):
    """Unlabeled dataset yielding two augmented views of each image (SimCLR-style)."""

    def __init__(self, df: pd.DataFrame, simclr_transform: Callable[[Image.Image], tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = simclr_transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        with Image.open(row["image_path"]) as img:
            view1, view2 = self.transform(img.convert("RGB"))
        return view1, view2


def build_datasets(df: pd.DataFrame, train_transform, eval_transform) -> dict[str, FaceMultiTaskDataset]:
    """Split a full metadata DataFrame (with a ``split`` column) into the four datasets.

    Keys match the split protocol exactly: ``train`` (model fitting),
    ``validation`` (early stopping / checkpoint selection only),
    ``calibration`` (fitting conformal intervals only), ``test`` (final
    evaluation only). See ``src/data/split_utils.py``.
    """
    datasets = {}
    for split_name in ("train", "validation", "calibration", "test"):
        transform = train_transform if split_name == "train" else eval_transform
        subset = df[df["split"] == split_name]
        datasets[split_name] = FaceMultiTaskDataset(subset, transform)
    return datasets
