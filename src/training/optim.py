"""Optimizer parameter-grouping and scheduler helpers shared by the core
``Trainer``.

The single source of truth for two training-recipe decisions the mission's
final-run hardening requires to be correct and consistent across both
trainers:

1. **No weight decay on biases, normalization parameters, and scalar
   loss-balancing parameters.** Applying AdamW weight decay to a bias, a
   BatchNorm/LayerNorm scale/shift, or a learned log-variance is a common
   but incorrect default: decaying a normalization scale fights the
   normalization, and decaying ``log_var_age``/``log_var_gender`` biases the
   homoscedastic-uncertainty weighting toward an arbitrary prior. The rule
   used here -- "no decay for any parameter with ``ndim <= 1``" -- captures
   exactly biases (1-D), norm weight+bias (1-D), and scalar log-variances
   (0-D), while decaying only the ``>= 2``-D conv/linear weight tensors.

2. **Every trainable parameter appears in exactly one optimizer group.**
   ``build_param_groups`` asserts this (no missing, no duplicated
   parameter), so a refactor that accidentally drops or double-counts a
   parameter fails loudly instead of silently training a subset.

Neither function imports any optional dependency.
"""

from __future__ import annotations

import torch


def is_no_decay_param(param: torch.nn.Parameter) -> bool:
    """True for parameters that must not receive weight decay.

    ``ndim <= 1`` is a name-free, backbone-agnostic classifier that matches
    exactly the parameters the mission lists as no-decay: biases (1-D),
    BatchNorm/LayerNorm/other normalization weights and biases (1-D), and
    scalar loss-balancing parameters such as ``log_var_age`` (0-D). Every
    convolutional or linear *weight* tensor is ``>= 2``-D and therefore
    decays.
    """
    return param.ndim <= 1


def build_param_groups(
    named_parameters,
    lr_for_param,
    weight_decay: float,
    force_no_decay_ids: set[int] | None = None,
) -> list[dict]:
    """Bucket trainable parameters into AdamW groups by (learning rate, decay).

    Parameters
    ----------
    named_parameters:
        An iterable of ``(name, parameter)`` pairs, typically
        ``model.named_parameters()``.
    lr_for_param:
        Callable ``(name, param) -> float`` giving each parameter's learning
        rate (this is where differential/per-component LRs are expressed).
    weight_decay:
        The weight decay applied to decay-eligible parameters. No-decay
        parameters always get ``0.0``.
    force_no_decay_ids:
        Optional set of ``id(param)`` values to force into the no-decay
        bucket regardless of dimensionality (e.g. parameters a backbone
        explicitly declares via ``no_weight_decay()``).

    Returns a list of ``{"params", "lr", "weight_decay"}`` dicts. Raises if
    any trainable parameter would be dropped or duplicated.
    """
    force_no_decay_ids = force_no_decay_ids or set()
    # Key each bucket by (lr, weight_decay) so parameters sharing both land
    # in one group; ``dict`` preserves insertion order for stable, testable
    # group ordering.
    buckets: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
    seen_ids: set[int] = set()
    n_trainable = 0

    for name, param in named_parameters:
        if not param.requires_grad:
            continue
        n_trainable += 1
        if id(param) in seen_ids:
            raise ValueError(f"Parameter '{name}' appeared twice in named_parameters().")
        seen_ids.add(id(param))
        no_decay = is_no_decay_param(param) or id(param) in force_no_decay_ids
        wd = 0.0 if no_decay else weight_decay
        lr = float(lr_for_param(name, param))
        buckets.setdefault((lr, wd), []).append(param)

    groups = [
        {"params": params, "lr": lr, "weight_decay": wd}
        for (lr, wd), params in buckets.items()
    ]

    n_grouped = sum(len(g["params"]) for g in groups)
    if n_grouped != n_trainable:
        raise ValueError(
            f"Parameter grouping mismatch: {n_trainable} trainable parameters but "
            f"{n_grouped} ended up in optimizer groups (some were dropped or duplicated)."
        )
    if not groups:
        raise ValueError("build_param_groups() produced zero optimizer groups (no trainable parameters).")
    return groups


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    warmup_start_factor: float = 0.1,
):
    """Linear warmup (from ``warmup_start_factor * base_lr``) then cosine decay.

    Fixes the previous ``start_factor = 1.0 / warmup_epochs`` bug: with
    ``warmup_epochs == 1`` that gave ``start_factor == 1.0`` -- i.e. no
    warmup at all, the LR started at the full base value. Here the initial
    LR is explicitly ``warmup_start_factor * base_lr`` (strictly below the
    base LR for any ``warmup_start_factor < 1``) and rises linearly to the
    base LR over ``warmup_epochs`` before cosine annealing begins.

    Built from ``LinearLR`` + ``CosineAnnealingLR`` via ``SequentialLR`` so
    every parameter group's own base LR is scaled correctly (the optimizer
    can have more than one group). This scheduler is **epoch-based** -- the
    caller steps it once per epoch, and only after at least one real
    optimizer step that epoch (see the AMP-skip handling in the trainers).
    """
    if not 0.0 < warmup_start_factor <= 1.0:
        raise ValueError(f"warmup_start_factor must be in (0, 1], got {warmup_start_factor}.")

    warmup_epochs = max(0, min(warmup_epochs, max(0, total_epochs - 1)))
    cosine_epochs = max(1, total_epochs - warmup_epochs)

    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_start_factor, end_factor=1.0, total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs],
    )
