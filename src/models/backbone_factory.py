"""Backbone factory: builds the backbone named by ``model.backbone.name``.

All supported backbones expose the same interface --
``embedding_dim`` (attribute), ``forward(x)``, ``forward_features(x)``
(returning ``layer1``-``layer4`` feature maps for Grad-CAM), and
``num_parameters()`` -- so callers (``MultiTaskFaceModel``, Grad-CAM,
progressive-freezing stage logic) do not need to know which one is
active. ``custom_resnet18`` (the project's main research backbone) is
the default. ``simple_cnn`` and ``plain_deep18_no_skip`` exist only as
controlled baselines for two distinct research questions (see
``configs/experiments.yaml``):

* ``simple_cnn`` -- efficiency/accuracy trade-off: a compact, differently
  shaped CNN vs. the full ResNet-18 (``exp_0_simple_cnn_shared_adapters_learned_balance``).
* ``plain_deep18_no_skip`` -- a *depth/width-matched* counterpart to
  ``custom_resnet18`` with only the residual additions removed, isolating
  the causal contribution of skip connections specifically
  (``exp_0b_plain_deep18_no_skip_shared_adapters_learned_balance``).
"""

from __future__ import annotations

import torch.nn as nn

from src.models.custom_resnet import build_backbone as _build_custom_resnet18
from src.models.plain_deep18_no_skip import build_backbone as _build_plain_deep18_no_skip
from src.models.simple_cnn import build_simple_cnn as _build_simple_cnn

_BACKBONE_BUILDERS = {
    "custom_resnet18": _build_custom_resnet18,
    "simple_cnn": _build_simple_cnn,
    "plain_deep18_no_skip": _build_plain_deep18_no_skip,
}


def build_backbone(config: dict) -> nn.Module:
    """Build the backbone named by ``config["name"]`` (default ``"custom_resnet18"``)."""
    name = config.get("name", "custom_resnet18")
    if name not in _BACKBONE_BUILDERS:
        raise ValueError(f"Unknown backbone '{name}', expected one of {list(_BACKBONE_BUILDERS)}")
    return _BACKBONE_BUILDERS[name](config)
