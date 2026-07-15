"""Task-specific residual bottleneck adapters.

Each adapter lets a task (age or dataset gender-label) adjust the shared
512-d backbone embedding without duplicating the backbone itself:

    adapter_output = z + up(dropout(gelu(down(z))))

The bottleneck dimension is configurable (default 256) and is intentionally
much smaller than the 512-d backbone, so adapters add few parameters
relative to the shared backbone (see ``count_parameters`` /
``scripts/generate_architecture_report.py`` for the actual comparison).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """Residual bottleneck adapter: down-project, GELU, dropout, up-project, add."""

    def __init__(self, input_dim: int = 512, bottleneck_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)

        # Near-identity initialization: the adapter starts close to a no-op
        # so early training does not disturb the shared representation.
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        delta = self.up_proj(self.dropout(self.activation(self.down_proj(z))))
        return z + delta

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class AgeAdapter(BottleneckAdapter):
    """Bottleneck adapter specialized for the age-estimation task."""


class GenderAdapter(BottleneckAdapter):
    """Bottleneck adapter specialized for the dataset gender-label task."""


class IdentityAdapter(nn.Module):
    """No-op adapter used when ``model.adapters.enabled`` is False."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def num_parameters(self) -> int:
        return 0
