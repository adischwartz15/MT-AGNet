"""Locates and loads trained model artifacts (checkpoint, calibration, kNN index)."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import torch.nn as nn

from src.models.multitask_model import MultiTaskFaceModel
from src.training.checkpointing import load_checkpoint

logger = logging.getLogger(__name__)


def _construct_model_offline(family: str, saved_config: dict) -> nn.Module:
    """Reconstruct a model architecture for loading a **local** checkpoint's
    weights -- never triggers a pretrained-weight download, regardless of
    what the saved config's own ``pretrained: true`` flag says, since the
    real (fine-tuned) weights are loaded from ``checkpoint["model_state_dict"]``
    immediately after this returns; downloading fresh ImageNet weights first
    would be wasted work at best and an offline failure at worst.

    Deep-copies ``saved_config`` before forcing any reconstruction-only
    ``pretrained`` flag to ``False``, so the caller's returned provenance
    config (the original, unmutated ``saved_config``) still faithfully
    records what the checkpoint was actually *trained* with.
    """
    if family == "pretrained_resnet":
        from src.models.pretrained_resnet import build_pretrained_resnet_model

        construction_config = copy.deepcopy(saved_config)
        construction_config["model"]["pretrained_resnet"]["pretrained"] = False
        return build_pretrained_resnet_model(construction_config)
    if family == "core":
        return MultiTaskFaceModel(saved_config)
    raise ValueError(f"Unknown model.family '{family}', expected 'core' or 'pretrained_resnet'")


def load_model_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict, dict]:
    """Load a model from a checkpoint produced by this repository's trainer.

    ``config["model"]["family"]`` selects the model class to reconstruct:
    ``"core"`` (the default -- every existing checkpoint/config lacks this
    key, so it always resolves to the original ``MultiTaskFaceModel`` path
    unchanged) or ``"pretrained_resnet"`` (the pretrained-torchvision-ResNet
    bridge baseline, see ``src/models/pretrained_resnet.py``).

    Reconstruction never re-downloads pretrained weights (see
    :func:`_construct_model_offline`) -- the returned ``config`` is the
    checkpoint's original, unmutated saved config (i.e. still records
    ``pretrained: true`` if that's what training actually used), not the
    ``pretrained: false`` reconstruction-only config used internally to
    build the architecture before loading its real weights.
    """
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    saved_config = checkpoint["config"]
    family = saved_config["model"].get("family", "core")
    model = _construct_model_offline(family, saved_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, saved_config, checkpoint
