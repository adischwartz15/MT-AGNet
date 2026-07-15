"""Shared, human-readable live-progress formatting for the trainer.

Pure string-building (no I/O beyond :func:`emit`'s ``print``) so every piece
is directly unit-testable without running real training.

Why this module exists: a Colab/Kaggle cell only shows what actually reaches
stdout, flushed, as it happens -- ``logger.info(...)`` alone is not enough
unless a handler is actually attached and reachable from the current
logger's propagation chain (easy to get wrong silently; a misconfigured
logger just produces no output at all, not an error). Every function here
returns a plain string; callers are expected to pass it to :func:`emit`,
which both prints (with ``flush=True``, so it is visible immediately even
when the process's stdout is piped, e.g. through a notebook's
``subprocess.Popen`` progress relay) and logs it.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def emit(text: str) -> None:
    """Print immediately (unbuffered) and mirror to the logger.

    ``flush=True`` forces the line to the OS pipe right away regardless of
    whether stdout is a TTY (line-buffered) or a pipe (normally fully
    buffered) -- the difference that silently delays a notebook's live
    progress display until a large internal buffer fills or the process
    exits. Also logged (at INFO) so it lands in the run's log file when one
    is configured, but the ``print`` is the primary channel: unlike
    ``logging.getLogger(__name__).info(...)`` alone, this is visible even
    when no logging handler has been attached anywhere in the calling
    module's propagation chain.
    """
    print(text, flush=True)
    logger.info(text)


def _fmt(value, spec: str = "{:.4f}", na: str = "n/a") -> str:
    if value is None:
        return na
    try:
        if value != value:  # NaN
            return na
    except TypeError:
        return na
    try:
        return spec.format(value)
    except (ValueError, TypeError):
        return str(value)


def _tag(experiment_name: str, seed: int | None) -> str:
    return f"[{experiment_name} | seed={seed}]" if seed is not None else f"[{experiment_name}]"


def format_lr_groups(named_lrs: dict[str, float | None]) -> str:
    """``{"backbone": 3e-5, "adapters": 3e-4, "balance": None}`` ->
    ``"backbone=3.00e-05 adapters=3.00e-04 balance=frozen/inactive"`` --
    ``None`` means that component had no trainable parameters this stage
    (e.g. a frozen backbone in Stage 1, or no loss-balancing parameters
    under fixed weighting), never a fabricated LR value."""
    if not named_lrs:
        return "n/a"
    parts = []
    for name, lr in named_lrs.items():
        parts.append(f"{name}={_fmt(lr, '{:.2e}')}" if lr is not None else f"{name}=frozen/inactive")
    return " ".join(parts)


def describe_trainable_backbone_parts(model) -> str:
    """Generic (model-agnostic) summary of which immediate backbone
    submodules currently have trainable parameters -- works for any model
    exposing a ``.backbone`` ``nn.Module`` attribute (e.g. ResNet-18/50 via
    torchvision) without needing model-specific code."""
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return "n/a (model has no .backbone attribute)"
    parts = []
    for name, child in backbone.named_children():
        params = list(child.parameters())
        if not params:
            continue
        n_trainable = sum(1 for p in params if p.requires_grad)
        if n_trainable == 0:
            parts.append(f"{name}=frozen")
        elif n_trainable == len(params):
            parts.append(f"{name}=trainable")
        else:
            parts.append(f"{name}=partially-trainable({n_trainable}/{len(params)})")
    return ", ".join(parts) if parts else "n/a (backbone has no direct child modules with parameters)"


def format_stage_announcement(experiment_name: str, seed: int | None, stage_name: str, model) -> str:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return (
        f"{_tag(experiment_name, seed)} === {stage_name} === "
        f"trainable_params={trainable:,}/{total:,}\n"
        f"  backbone parts: {describe_trainable_backbone_parts(model)}"
    )


def format_epoch_report(
    *,
    experiment_name: str,
    seed: int | None,
    stage_name: str,
    epoch: int,
    total_epochs: int,
    train_metrics: dict,
    val_metrics: dict,
    lr_groups: dict[str, float | None],
    selection_score: float,
    is_best: bool,
    best_score: float | None,
    best_epoch: int | None,
    early_stopping_bad_epochs: int,
    early_stopping_patience: int,
    epoch_seconds: float,
    checkpoint_path,
) -> str:
    """The single, canonical per-epoch progress block both trainers print.

    Every field the mission requires (experiment/seed/stage/epoch, train
    total/age/gender loss, val loss/age MAE/RMSE, val gender accuracy/
    balanced accuracy/F1, selective accuracy/coverage/abstention, per-group
    LRs, log-variances, effective loss weights, selection score, best
    score+epoch, early-stopping counter, epoch duration, last checkpoint
    path) is included; any metric a particular run doesn't produce (e.g. no
    gender task this run, or balanced accuracy not computed this epoch)
    renders as ``"n/a"`` rather than a fabricated 0.0 or being silently
    dropped.
    """
    tag = _tag(experiment_name, seed)
    lines = [
        f"{tag} {stage_name} | Epoch {epoch:02d}/{total_epochs:02d} | {epoch_seconds:.1f}s",
        (
            f"  train: total={_fmt(train_metrics.get('loss'))} "
            f"age={_fmt(train_metrics.get('age_loss'))} gender={_fmt(train_metrics.get('gender_loss'))}"
        ),
        (
            f"  val:   total={_fmt(val_metrics.get('loss'))} "
            f"age_mae={_fmt(val_metrics.get('age_mae'), '{:.3f}')} "
            f"age_rmse={_fmt(val_metrics.get('age_rmse'), '{:.3f}')}"
        ),
        (
            f"         gender_acc={_fmt(val_metrics.get('gender_accuracy'), '{:.3f}')} "
            f"balanced_acc={_fmt(val_metrics.get('gender_balanced_accuracy'), '{:.3f}')} "
            f"f1={_fmt(val_metrics.get('gender_f1'), '{:.3f}')}"
        ),
        (
            f"         selective_acc={_fmt(val_metrics.get('gender_selective_accuracy'), '{:.3f}')} "
            f"coverage={_fmt(val_metrics.get('gender_coverage'), '{:.3f}')} "
            f"abstention={_fmt(val_metrics.get('gender_abstention'), '{:.3f}')}"
        ),
        f"  lr: {format_lr_groups(lr_groups)}",
        (
            f"  log_var: age={_fmt(train_metrics.get('log_var_age'))} "
            f"gender={_fmt(train_metrics.get('log_var_gender'))} | "
            f"loss_weights: age={_fmt(train_metrics.get('effective_age_weight'))} "
            f"gender={_fmt(train_metrics.get('effective_gender_weight'))}"
        ),
        (
            f"  selection_score={_fmt(selection_score)} best={'yes' if is_best else 'no'} "
            f"(best_score={_fmt(best_score)} @ epoch {best_epoch if best_epoch is not None else 'n/a'}) | "
            f"early_stop={early_stopping_bad_epochs}/{early_stopping_patience}"
        ),
        f"  checkpoint: {checkpoint_path if checkpoint_path is not None else 'n/a (no improvement yet)'}",
    ]
    return "\n".join(lines)


def format_multi_seed_preflight(
    experiment_name: str,
    requested_seeds: list[int],
    completed_seeds: list[int],
    incomplete_resumable_seeds: list[int],
    missing_seeds: list[int],
    will_run_now_seeds: list[int],
) -> str:
    """Printed once, before a multi-seed loop starts training anything, so
    it's immediately clear from the top of a Colab/Kaggle cell's output
    what this run is and isn't about to do -- e.g. that "3 requested" means
    "1 reused, 1 resumed, 1 fresh," not 3 full training runs."""
    return "\n".join(
        [
            f"[{experiment_name}] Multi-seed run plan:",
            f"  requested seeds:            {list(requested_seeds)}",
            f"  already completed (reused): {list(completed_seeds)}",
            f"  incomplete (will resume):   {list(incomplete_resumable_seeds)}",
            f"  missing (will start fresh): {list(missing_seeds)}",
            f"  will run now:               {list(will_run_now_seeds)}",
        ]
    )


def format_resume_announcement(
    experiment_name: str,
    seed: int | None,
    resume_source: str,
    stage: str | None,
    epoch: int | None,
    global_step: int | None,
    best_score: float | None,
    checkpoint_path,
    checkpoint_sha256: str | None,
    split_sha256: str | None,
) -> str:
    tag = _tag(experiment_name, seed)
    return "\n".join(
        [
            f"{tag} Resuming training:",
            f"  resume source:     {resume_source}",
            f"  stage:             {stage if stage is not None else 'n/a'}",
            f"  epoch:             {epoch if epoch is not None else 'n/a'}",
            f"  global step:       {global_step if global_step is not None else 'n/a'}",
            f"  best score so far: {_fmt(best_score)}",
            f"  checkpoint path:   {checkpoint_path if checkpoint_path is not None else 'n/a'}",
            f"  checkpoint sha256: {checkpoint_sha256 or 'n/a'}",
            f"  split sha256:      {split_sha256 or 'n/a'}",
        ]
    )
