"""Pinball (quantile) loss for the age q10/q50/q90 head."""

from __future__ import annotations

import torch


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    """Per-sample pinball loss for a single quantile, no reduction.

    ``L_tau(y, yhat) = max(tau * (y - yhat), (tau - 1) * (y - yhat))``
    """
    error = target - pred
    return torch.maximum(quantile * error, (quantile - 1.0) * error)


def multi_quantile_pinball_loss(
    q10: torch.Tensor,
    q50: torch.Tensor,
    q90: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    quantiles: tuple[float, float, float] = (0.10, 0.50, 0.90),
) -> torch.Tensor:
    """Mean pinball loss averaged over the three quantiles, with optional masking.

    ``mask`` is a boolean/float tensor of shape ``(batch,)`` marking which
    samples have a valid age label; masked-out samples contribute zero loss
    and are excluded from the averaging denominator. If every sample in the
    batch is masked out, returns a zero-valued tensor (with gradient) so the
    task simply contributes nothing to the total loss for that batch.
    """
    losses = (
        pinball_loss(q10, target, quantiles[0])
        + pinball_loss(q50, target, quantiles[1])
        + pinball_loss(q90, target, quantiles[2])
    ) / 3.0

    if mask is None:
        return losses.mean()

    mask = mask.float()
    valid_count = mask.sum()
    if valid_count.item() == 0:
        return losses.sum() * 0.0
    return (losses * mask).sum() / valid_count
