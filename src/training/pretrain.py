"""Optional lightweight SimCLR-style self-supervised pretraining.

Uses the same manually implemented ``CustomResNet18`` backbone plus a
small projection head (discarded after pretraining -- only the 512-d
encoder is kept). This is optional and compute-hungry relative to
supervised training; see docs/reproducibility.md for expected compute.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.custom_resnet import CustomResNet18, build_backbone
from src.utils.seed import seed_worker

logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """MLP projection head mapping the 512-d embedding to a smaller contrastive space."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 512, projection_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy loss (SimCLR)."""
    batch_size = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)
    similarity = representations @ representations.t() / temperature

    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
    similarity.masked_fill_(mask, float("-inf"))

    positive_indices = torch.cat(
        [torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]
    ).to(z1.device)

    return F.cross_entropy(similarity, positive_indices)


def pretrain_simclr(
    backbone_cfg: dict,
    pretrain_cfg: dict,
    train_dataset,
    device: str,
    checkpoint_dir: str | Path,
) -> dict:
    """Run SimCLR pretraining and save the encoder checkpoint.

    Returns a small history dict (per-epoch loss, epoch time) for reporting.
    """
    encoder: CustomResNet18 = build_backbone(backbone_cfg).to(device)
    projector = ProjectionHead(
        input_dim=backbone_cfg.get("embedding_dim", 512),
        projection_dim=pretrain_cfg.get("projection_dim", 128),
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=pretrain_cfg.get("lr", 3.0e-4),
        weight_decay=pretrain_cfg.get("weight_decay", 1e-6),
    )
    num_workers = pretrain_cfg.get("num_workers", 2)
    loader = DataLoader(
        train_dataset,
        batch_size=pretrain_cfg.get("batch_size", 128),
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=(device == "cuda"),
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )

    history = {"loss": [], "epoch_time_seconds": []}
    temperature = pretrain_cfg.get("temperature", 0.5)

    for epoch in range(pretrain_cfg.get("epochs", 20)):
        start = time.time()
        total_loss, n_batches = 0.0, 0
        encoder.train()
        projector.train()
        for view1, view2 in loader:
            view1, view2 = view1.to(device), view2.to(device)
            z1 = projector(encoder(view1))
            z2 = projector(encoder(view2))
            loss = nt_xent_loss(z1, z2, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        elapsed = time.time() - start
        avg_loss = total_loss / max(1, n_batches)
        history["loss"].append(avg_loss)
        history["epoch_time_seconds"].append(elapsed)
        logger.info("[pretrain] epoch %d | nt_xent_loss=%.4f (%.1fs)", epoch + 1, avg_loss, elapsed)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_name = pretrain_cfg.get("checkpoint_name", "simclr_encoder.pt")
    out_path = checkpoint_dir / checkpoint_name
    torch.save({"encoder_state_dict": encoder.state_dict(), "history": history}, out_path)
    logger.info("Saved SimCLR encoder checkpoint to %s", out_path)

    return {"history": history, "checkpoint_path": str(out_path)}
