"""A depth/width-matched plain (non-residual) counterpart to CustomResNet18.

This exists to isolate the causal contribution of residual skip
connections specifically, as distinct from the SimpleCNN-vs-ResNet
comparison (which also differs in depth, width, and stage design and is
therefore an efficiency/accuracy trade-off, not a clean ablation of
residual connections). See
``configs/experiments.yaml: exp_0b_plain_deep18_no_skip_shared_adapters_learned_balance``
and ``docs/experiment_plan.md``.

``PlainDeep18NoSkip`` copies ``CustomResNet18``'s stem, stage widths,
number of convolutional layers per stage (block layout [2, 2, 2, 2], two
3x3 convolutions per block), normalization, activation, embedding size,
and strided main-path downsampling exactly -- the only change is that
``PlainBlock.forward`` never adds an identity/projection shortcut, so
there is no 1x1 projection-shortcut submodule at all. This means
``PlainDeep18NoSkip`` has strictly *fewer* parameters than
``CustomResNet18`` (missing the three 1x1 conv + BatchNorm downsample
shortcuts at the start of layer2/layer3/layer4, where spatial resolution
and channel count change) -- an unavoidable, explicitly documented
difference of a few thousand parameters out of ~11M, not a design choice
that favors either architecture (see ``docs/experiment_plan.md`` for the
exact counts). No prebuilt architecture, model hub, or pretrained
weights are used here either, consistent with the rest of this
repository.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_LAYER_NAMES = ("layer1", "layer2", "layer3", "layer4")


class PlainBlock(nn.Module):
    """``CustomResNet18.BasicBlock`` with the residual addition removed.

    ``out = ReLU(BN(conv2(ReLU(BN(conv1(x))))))`` -- identical main-path
    convolutions (same kernel sizes, strides, channel counts, BatchNorm,
    and ReLU placement as ``src/models/custom_resnet.py:BasicBlock``), but
    no ``x + ...`` addition and consequently no downsample/projection
    shortcut submodule at all.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out


class PlainDeep18NoSkip(nn.Module):
    """Depth/width-matched plain counterpart to CustomResNet18, no skip connections.

    Same stem, stage widths (64/128/256/512), block layout [2, 2, 2, 2],
    embedding projection, and weight initialization as
    ``src/models/custom_resnet.py:CustomResNet18`` -- the sole intended
    difference is the absence of residual additions (see ``PlainBlock``).
    """

    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 64,
        block_layout: tuple[int, int, int, int] = (2, 2, 2, 2),
        embedding_dim: int = 512,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self._in_channels = stem_channels

        # Same stem as CustomResNet18: 7x7 stride-2 conv + BN + ReLU + 3x3 stride-2 max pool.
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

        final_channels = channels[3] * PlainBlock.expansion
        self.embedding_proj: nn.Module
        if final_channels != embedding_dim:
            self.embedding_proj = nn.Linear(final_channels, embedding_dim)
        else:
            self.embedding_proj = nn.Identity()

        self._initialize_weights()

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = [PlainBlock(self._in_channels, out_channels, stride=stride)]
        self._in_channels = out_channels * PlainBlock.expansion
        for _ in range(1, num_blocks):
            layers.append(PlainBlock(self._in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
        # No zero_init_residual equivalent: there is no residual branch to
        # zero-init the last BatchNorm of (that trick only makes sense when
        # a block can start as an identity mapping, which requires the
        # addition this backbone deliberately omits).

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


def build_backbone(config: dict) -> PlainDeep18NoSkip:
    """Build a :class:`PlainDeep18NoSkip` from a ``model.backbone`` config dict."""
    return PlainDeep18NoSkip(
        in_channels=config.get("in_channels", 3),
        stem_channels=config.get("stem_channels", 64),
        block_layout=tuple(config.get("block_layout", [2, 2, 2, 2])),
        embedding_dim=config.get("embedding_dim", 512),
    )
