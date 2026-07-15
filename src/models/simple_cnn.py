"""A conventional (non-residual) CNN backbone, used only as a controlled baseline.

This exists to answer one focused question: does the manually
implemented Custom ResNet-18's residual design provide measurable value
over an otherwise-comparable plain CNN, when everything else in the
multi-task system (adapters, heads, loss balancing, training setup,
data split, evaluation pipeline) is held constant? See
``configs/experiments.yaml: exp_0_simple_cnn_shared_adapters_learned_balance``
and ``docs/experiment_plan.md``.

This is a deliberately conventional stacked-ConvBlock design with **no
skip connections or residual additions** -- that is the entire point of
the comparison. It is not intended to be tuned into a competitive
standalone architecture, and it must never be described as the project's
main backbone; ``CustomResNet18`` remains that. No prebuilt architecture,
model hub, or pretrained weights are used here either, consistent with
the rest of this repository.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_LAYER_NAMES = ("layer1", "layer2", "layer3", "layer4")


class ConvBlock(nn.Module):
    """Conv3x3 -> BatchNorm -> ReLU (+ optional 2x2 max-pool), no residual path."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class SimpleCNNBackbone(nn.Module):
    """Plain stacked-ConvBlock CNN, no residual connections.

    Channel progression 3 -> 32 -> 64 -> 128 -> 256 -> 512, with a 2x2
    max-pool after each of the first four blocks and none after the
    fifth (the network instead relies on adaptive average pooling to
    collapse spatial dimensions). ``layer4`` bundles the fourth ConvBlock
    (256 channels, pooled) with the fifth, non-pooled 512-channel
    ConvBlock, so it exposes the same 512-channel final feature map that
    ``CustomResNet18.layer4`` does -- this keeps Grad-CAM
    (``src/evaluation/gradcam.py``) and the parameter/analysis reporting
    directly comparable between the two backbones. There is no separate
    "stem" stage: ``layer1`` fulfills that role, matching the
    progressive-freezing stage names (`stem`, `layer1`-`layer4`) as
    closely as a 4-stage plain CNN reasonably can -- an unmatched "stem"
    entry in a freeze/unfreeze list is simply a no-op for this backbone.
    """

    def __init__(self, in_channels: int = 3, embedding_dim: int = 512) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.layer1 = ConvBlock(in_channels, 32)
        self.layer2 = ConvBlock(32, 64)
        self.layer3 = ConvBlock(64, 128)
        self.layer4 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 512, pool=False),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        final_channels = 512
        self.embedding_proj: nn.Module
        if final_channels != embedding_dim:
            self.embedding_proj = nn.Linear(final_channels, embedding_dim)
        else:
            self.embedding_proj = nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return intermediate feature maps, keyed by layer name (for Grad-CAM)."""
        features: dict[str, torch.Tensor] = {}
        out = x
        for name in _LAYER_NAMES:
            out = getattr(self, name)(out)
            features[name] = out
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        pooled = self.avgpool(features["layer4"]).flatten(1)
        embedding = self.embedding_proj(pooled)
        return embedding

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_simple_cnn(config: dict) -> SimpleCNNBackbone:
    """Build a :class:`SimpleCNNBackbone` from a ``model.backbone`` config dict."""
    return SimpleCNNBackbone(
        in_channels=config.get("in_channels", 3),
        embedding_dim=config.get("embedding_dim", 512),
    )
