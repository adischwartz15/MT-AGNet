"""Checkpoint save/load helpers.

Checkpoints are plain ``torch.save`` dicts containing the model state,
optimizer state, epoch, config snapshot, and metric history. Only
checkpoints written by this repository (or a compatible file explicitly
supplied by the user) are ever loaded -- there is no automatic download
path anywhere in this module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy
import torch

logger = logging.getLogger(__name__)

# Checkpoints saved by this repository's trainers embed a NumPy RNG state
# (see PersistentArtifactManager, saved for reproducible resume) alongside
# the tensor state dicts. PyTorch >=2.6 defaults
# torch.load to weights_only=True, whose unpickler rejects any global not
# explicitly allow-listed -- these four are exactly the NumPy array/dtype
# reconstruction primitives needed to unpickle that RNG state (verified
# against a real checkpoint: each was added one at a time, following the
# unpickler's own "Unsupported global" errors, until loading succeeded).
# None of them execute arbitrary code; this keeps weights_only=True's
# protection against pickle payloads for everything else.
_SAFE_CHECKPOINT_GLOBALS = [
    numpy._core.multiarray._reconstruct,
    numpy.ndarray,
    numpy.dtype,
    numpy.dtypes.UInt32DType,
]


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
        "extra": extra or {},
    }
    torch.save(payload, path)
    logger.info("Saved checkpoint to %s (epoch=%d)", path, epoch)
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with torch.serialization.safe_globals(_SAFE_CHECKPOINT_GLOBALS):
        return torch.load(path, map_location=map_location, weights_only=True)


class BestMetricTracker:
    """Tracks the best value seen for a metric and whether it just improved.

    ``mode`` is ``"min"`` (e.g. MAE) or ``"max"`` (e.g. accuracy).
    """

    def __init__(self, mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        self.best_value: float | None = None
        self.best_epoch: int | None = None

    def is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value
        return value > self.best_value

    def update(self, value: float, epoch: int | None = None) -> bool:
        improved = self.is_improvement(value)
        if improved:
            self.best_value = value
            self.best_epoch = epoch
        return improved
