"""Manually implemented ResNet-18 backbone.

Every layer here is written from scratch with ``torch.nn`` primitives.
This module must never import ``torchvision.models``, ``timm``, or any
other prebuilt architecture / pretrained-weight source. The only way to
initialize non-random weights is via a checkpoint produced by this
repository (supervised training or SimCLR-style self-supervised
pretraining) or a compatible local checkpoint explicitly supplied by the
user through ``model.pretrained_checkpoint``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_LAYER_NAMES = ("layer1", "layer2", "layer3", "layer4")


class BasicBlock(nn.Module):
    """Standard ResNet "basic" residual block (two 3x3 convolutions).

    ``out = ReLU(BN(conv2(ReLU(BN(conv1(x))))) + shortcut(x))``

    The shortcut is the identity when the spatial resolution and channel
    count are unchanged, otherwise a 1x1 strided convolution + BatchNorm
    ("downsampling block") to match dimensions.
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class CustomResNet18(nn.Module):
    """Manually implemented ResNet-18 with block layout [2, 2, 2, 2].

    Produces a 512-dimensional embedding via global average pooling over
    the final feature map (``layer4`` output for the default 512-d
    embedding, matching the standard ResNet-18 design).
    """

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 64,
        block_layout: tuple[int, int, int, int] = (2, 2, 2, 2),
        embedding_dim: int = 512,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self._in_channels = stem_channels

        # Stem: 7x7 stride-2 conv + BN + ReLU + 3x3 stride-2 max pool.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        channels = [stem_channels, stem_channels * 2, stem_channels * 4, stem_channels * 8]
        strides = [1, 2, 2, 2]
        self.layer1 = self._make_layer(channels[0], block_layout[0], strides[0])
        self.layer2 = self._make_layer(channels[1], block_layout[1], strides[1])
        self.layer3 = self._make_layer(channels[2], block_layout[2], strides[2])
        self.layer4 = self._make_layer(channels[3], block_layout[3], strides[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        final_channels = channels[3] * BasicBlock.expansion
        self.embedding_proj: nn.Module
        if final_channels != embedding_dim:
            self.embedding_proj = nn.Linear(final_channels, embedding_dim)
        else:
            self.embedding_proj = nn.Identity()

        self._initialize_weights(zero_init_residual)

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self._in_channels, out_channels, stride=stride)]
        self._in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self._in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self, zero_init_residual: bool) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

        if zero_init_residual:
            # Zero-init the last BN in each residual branch so residual
            # blocks start as identity mappings, a standard trick that
            # improves early-training stability (He et al., 2016 follow-up).
            for module in self.modules():
                if isinstance(module, BasicBlock):
                    nn.init.constant_(module.bn2.weight, 0.0)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return intermediate feature maps, keyed by layer name (for Grad-CAM)."""
        features: dict[str, torch.Tensor] = {}
        out = self.stem(x)
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


def build_backbone(config: dict) -> CustomResNet18:
    """Build a :class:`CustomResNet18` from a ``model.backbone`` config dict."""
    return CustomResNet18(
        in_channels=config.get("in_channels", 3),
        stem_channels=config.get("stem_channels", 64),
        block_layout=tuple(config.get("block_layout", [2, 2, 2, 2])),
        embedding_dim=config.get("embedding_dim", 512),
        zero_init_residual=config.get("zero_init_residual", True),
    )
