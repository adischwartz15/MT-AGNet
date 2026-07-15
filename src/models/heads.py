"""Task heads: age quantile regression and dataset gender-label classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgeQuantileHead(nn.Module):
    """Predicts (q10, q50, q90) for age, guaranteeing q10 <= q50 <= q90.

    Parameterization (safe by construction):
        q50 = age_min + sigmoid(center_raw) * (age_max - age_min)
        q10 = q50 - softplus(lower_delta)
        q90 = q50 + softplus(upper_delta)

    ``softplus`` guarantees the deltas are non-negative, so ordering holds
    for any network output. q10/q90 are then clamped into
    [age_min, age_max] only for display; the raw (unclamped) values are
    also returned since clamping can otherwise silently break the pinball
    loss gradient near the boundary.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 128, dropout: float = 0.1,
                 age_min: float = 0.0, age_max: float = 120.0) -> None:
        super().__init__()
        self.age_min = age_min
        self.age_max = age_max
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.center_head = nn.Linear(hidden_dim, 1)
        self.lower_delta_head = nn.Linear(hidden_dim, 1)
        self.upper_delta_head = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(z)
        center_raw = torch.sigmoid(self.center_head(h)).squeeze(-1)
        q50 = self.age_min + center_raw * (self.age_max - self.age_min)

        lower_delta = F.softplus(self.lower_delta_head(h)).squeeze(-1)
        upper_delta = F.softplus(self.upper_delta_head(h)).squeeze(-1)

        q10 = q50 - lower_delta
        q90 = q50 + upper_delta

        q10_clamped = torch.clamp(q10, self.age_min, self.age_max)
        q90_clamped = torch.clamp(q90, self.age_min, self.age_max)
        q50_clamped = torch.clamp(q50, self.age_min, self.age_max)

        return {
            "q10": q10_clamped,
            "q50": q50_clamped,
            "q90": q90_clamped,
            "q10_raw": q10,
            "q50_raw": q50,
            "q90_raw": q90,
        }


class GenderClassificationHead(nn.Module):
    """Softmax classifier over dataset gender labels.

    Outputs raw logits; converting logits -> probabilities -> "Not sure"
    abstention is handled at inference time (see
    ``src/inference/predictor.py``) using the configurable confidence
    threshold, so the head itself stays a plain classifier usable for
    both training (cross-entropy) and inference.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 128, dropout: float = 0.1,
                 num_classes: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.logit_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.trunk(z)
        return self.logit_head(h)
