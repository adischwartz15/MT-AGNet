"""Progressive multi-stage trainer for the multi-task face model."""

from __future__ import annotations

import contextlib
import csv
import datetime
import json
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import gender_balanced_accuracy, gender_precision_recall_f1
from src.losses.multitask_loss import compute_multitask_loss
from src.models.multitask_model import MultiTaskFaceModel
from src.training.callbacks import EarlyStopping
from src.training.checkpointing import BestMetricTracker, save_checkpoint
from src.training.optim import build_param_groups, build_warmup_cosine_scheduler
from src.training.progress import emit, format_epoch_report, format_stage_announcement
from src.training.stages import Stage, build_stage_plan
from src.utils.provenance import dependency_versions, git_commit_sha
from src.utils.seed import seed_worker

logger = logging.getLogger(__name__)


def _build_optimizer(
    model: MultiTaskFaceModel, lr: float, weight_decay: float, differential_lr_cfg: dict | None = None,
) -> torch.optim.Optimizer:
    """Build the stage optimizer, optionally with a lower LR for backbone parameters.

    Differential (discriminative) learning rates -- a much smaller LR for
    the backbone than for the adapters/heads -- let the backbone keep
    training (rather than being fully frozen, the only alternative in the
    common no-pretrained-checkpoint path) while still protecting it from
    large, potentially destabilizing early updates. Applies identically
    regardless of which backbone/architecture is active (via
    ``model.backbone_parameters()``), so it does not change the relative
    comparison between experiments -- see ``configs/training.yaml:
    training.differential_lr``.
    """
    differential_lr_cfg = differential_lr_cfg or {}

    if not differential_lr_cfg.get("enabled", False):
        def lr_for(_name, _param):
            return lr
    else:
        multiplier = differential_lr_cfg.get("backbone_lr_multiplier", 0.1)
        backbone_param_ids = {id(p) for p in model.backbone_parameters()}

        def lr_for(_name, param):
            return lr * multiplier if id(param) in backbone_param_ids else lr

    # build_param_groups applies zero weight decay to biases, normalization
    # parameters, and the scalar log-variance loss-balancing parameters
    # (ndim <= 1), and decays only the >= 2-D conv/linear weight tensors --
    # see src/training/optim.py. It also asserts every trainable parameter
    # lands in exactly one group.
    groups = build_param_groups(model.named_parameters(), lr_for, weight_decay)
    return torch.optim.AdamW(groups)


def _build_scheduler(
    optimizer: torch.optim.Optimizer, total_epochs: int, warmup_epochs: int, warmup_start_factor: float = 0.1,
):
    """Linear warmup then cosine annealing. Thin wrapper around
    :func:`src.training.optim.build_warmup_cosine_scheduler`, where the
    real warmup fix (an explicit ``warmup_start_factor`` instead of the old
    ``1.0 / warmup_epochs``) lives."""
    return build_warmup_cosine_scheduler(optimizer, total_epochs, warmup_epochs, warmup_start_factor)


def resolve_loss_balancing(loss_cfg: dict, current_epoch: int) -> tuple[str, dict]:
    """Resolve the effective loss-balancing mode/fixed-weights for one epoch.

    Implements the loss-balancing warmup: for the first N *global* epochs
    (``loss_cfg.learned_uncertainty.warmup_epochs``, not reset per training
    stage), trains with equal fixed weights (1.0/1.0) even when the
    configured mode is ``learned_uncertainty`` -- the log-variance
    parameters haven't seen enough loss signal yet to be a meaningful
    weighting this early, and letting them influence the total loss from
    epoch 1 risks early gradient interference between the two tasks.
    ``current_epoch`` is 1-indexed. A pure function (no ``Trainer`` state)
    so this policy is directly unit-testable without running real training.
    """
    configured_mode = loss_cfg["mode"]
    warmup_epochs = (
        loss_cfg.get("learned_uncertainty", {}).get("warmup_epochs", 0)
        if configured_mode == "learned_uncertainty" else 0
    )
    in_warmup = configured_mode == "learned_uncertainty" and current_epoch <= warmup_epochs
    if in_warmup:
        return "fixed", {"age_weight": 1.0, "gender_weight": 1.0}
    return configured_mode, loss_cfg.get("fixed", {"age_weight": 1.0, "gender_weight": 1.0})


class Trainer:
    """Runs the Stage A/B/C (or warm-up) progressive training loop.

    Saves three separate "best" checkpoints during training: lowest
    validation age MAE, highest validation gender-label accuracy, and best
    balanced multi-task score (``gender_accuracy - normalized_age_mae``).
    """

    def __init__(
        self,
        model: MultiTaskFaceModel,
        config: dict,
        train_dataset,
        val_dataset,
        device: str = "cpu",
        checkpoint_dir: str | Path = "./checkpoints",
        experiment_name: str = "multitask",
        gender_class_weights: torch.Tensor | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.training_cfg = config["training"]
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.gender_class_weights = gender_class_weights.to(device) if gender_class_weights is not None else None
        self.confidence_threshold = config["model"]["gender_head"].get("confidence_threshold", 0.80)

        # Incremental per-epoch artifacts (history.csv/json, a live status
        # file) default to living alongside the checkpoint directory unless
        # an explicit output_dir is given -- this is what lets a notebook
        # recover training progress after a session interruption without
        # waiting for train() to return.
        self.output_dir = Path(output_dir) if output_dir is not None else self.checkpoint_dir.parent
        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_csv_path = self.metrics_dir / f"{experiment_name}_history.csv"
        self.history_json_path = self.metrics_dir / f"{experiment_name}_history.json"
        self.status_path = self.log_dir / f"{experiment_name}_status.json"

        self.train_dataset_size = len(train_dataset)
        self.val_dataset_size = len(val_dataset)

        batch_size = self.training_cfg.get("batch_size", 64)
        num_workers = self.training_cfg.get("num_workers", 2)
        # pin_memory speeds up host->device transfer, but only actually helps
        # (and is only supported) when transferring to a CUDA device.
        # worker_init_fn=seed_worker gives each DataLoader worker process its
        # own deterministic-but-distinct RNG state, so augmentation
        # randomness is reproducible across runs even with num_workers > 0
        # (without it, workers can otherwise end up sharing correlated RNG
        # state inherited from the parent process).
        pin_memory = device == "cuda"
        # Explicit seeded generator for the shuffled train loader, so the
        # per-epoch batch ordering is reproducible across runs/resumes (not
        # left to the global RNG state at DataLoader-construction time). The
        # seed is recorded in the run manifest by the caller.
        self.dataloader_seed = int(self.training_cfg.get("seed", self.config.get("seed", 42)))
        self.train_generator = torch.Generator()
        self.train_generator.manual_seed(self.dataloader_seed)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            drop_last=len(train_dataset) > batch_size, pin_memory=pin_memory,
            worker_init_fn=seed_worker if num_workers > 0 else None,
            generator=self.train_generator,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            pin_memory=pin_memory, worker_init_fn=seed_worker if num_workers > 0 else None,
        )

        self.mixed_precision = self.training_cfg.get("mixed_precision", True) and device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision)
        self.grad_clip_norm = self.training_cfg.get("grad_clip_norm", 1.0)

        # Optional hard caps on batches-per-epoch (config-driven, default
        # None = unlimited). Distinct from epoch count: a "smoke test" that
        # caps epochs to 1 still iterates the *entire* dataset once, which
        # can be slow on a large dataset -- these let a fast integration
        # check also cap batches-per-epoch, without affecting any real
        # training run that doesn't set them.
        self.max_train_batches = self.training_cfg.get("max_train_batches_per_epoch")
        self.max_val_batches = self.training_cfg.get("max_val_batches_per_epoch")

        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "val_age_mae": [], "val_age_rmse": [], "val_gender_accuracy": [],
            "val_gender_balanced_accuracy": [], "val_gender_f1": [],
            "val_gender_selective_accuracy": [], "val_gender_coverage": [], "val_gender_abstention": [],
            "age_loss": [], "gender_loss": [], "effective_age_weight": [], "effective_gender_weight": [],
            "log_var_age": [], "log_var_gender": [], "lr": [], "epoch_time_seconds": [],
        }
        self.epoch_times: list[float] = []

        self.trackers = {
            "age_mae": BestMetricTracker(mode="min"),
            "gender_accuracy": BestMetricTracker(mode="max"),
            "balanced_score": BestMetricTracker(mode="max"),
        }
        # Last checkpoint path written for *any* tracked metric this run --
        # a live progress line's "checkpoint:" field, not the multi-file
        # per-metric bookkeeping the trackers themselves already do.
        self._last_checkpoint_path: Path | None = None
        self.run_manifest_path = self.log_dir / f"{experiment_name}_run_manifest.json"
        self.last_checkpoint_path = self.checkpoint_dir / f"{experiment_name}_last.pt"

    def _loss_mode(self) -> str:
        return self.config["model"]["loss_balancing"]["mode"]

    def _run_batches(
        self, loader: DataLoader, optimizer: torch.optim.Optimizer | None, current_epoch: int,
    ) -> dict[str, float]:
        is_train = optimizer is not None
        self.model.train(is_train)

        total_loss, total_age_loss, total_gender_loss = 0.0, 0.0, 0.0
        n_age_batches, n_gender_batches, n_batches = 0, 0, 0
        eff_age_w, eff_gender_w, lv_age, lv_gender = 0.0, 0.0, 0.0, 0.0
        age_abs_errors = []
        gender_correct, gender_total = 0, 0
        gender_correct_accepted, gender_total_accepted = 0, 0
        gender_confidences = []
        gender_true_labels, gender_pred_labels = [], []
        any_optimizer_step = False

        loss_cfg = self.config["model"]["loss_balancing"]
        mode, fixed = resolve_loss_balancing(loss_cfg, current_epoch)
        learned_uncertainty_cfg = loss_cfg.get("learned_uncertainty", {})
        gender_loss_scale = learned_uncertainty_cfg.get("gender_loss_scale", 1.0)
        log_var_clamp_min = learned_uncertainty_cfg.get("log_var_clamp_min")
        log_var_clamp_max = learned_uncertainty_cfg.get("log_var_clamp_max")
        max_batches = self.max_train_batches if is_train else self.max_val_batches

        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = batch["image"].to(self.device)
            age_target = batch["age"].to(self.device)
            age_mask = batch["age_mask"].to(self.device)
            gender_target = batch["gender_label"].to(self.device)
            gender_mask = batch["gender_mask"].to(self.device)

            autocast_ctx = (
                torch.autocast(device_type="cuda") if self.mixed_precision else contextlib.nullcontext()
            )
            with torch.set_grad_enabled(is_train):
                with autocast_ctx:
                    outputs = self.model(images)
                    loss_out = compute_multitask_loss(
                        outputs["age_output"], outputs["gender_logits"], age_target, age_mask,
                        gender_target, gender_mask, mode=mode,
                        fixed_age_weight=fixed.get("age_weight", 1.0),
                        fixed_gender_weight=fixed.get("gender_weight", 1.0),
                        log_var_age=self.model.log_var_age, log_var_gender=self.model.log_var_gender,
                        gender_class_weights=self.gender_class_weights,
                        gender_loss_scale=gender_loss_scale,
                        log_var_clamp_min=log_var_clamp_min,
                        log_var_clamp_max=log_var_clamp_max,
                    )

            if is_train:
                optimizer.zero_grad()
                self.scaler.scale(loss_out.total_loss).backward()
                if self.grad_clip_norm:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                scale_before_step = self.scaler.get_scale()
                self.scaler.step(optimizer)
                self.scaler.update()
                # GradScaler silently skips optimizer.step() on an inf/NaN gradient (it only
                # shrinks the scale instead); if that happens on every batch of an epoch the
                # epoch-level LR scheduler must not step either, or it runs ahead of the
                # optimizer and torch raises "lr_scheduler.step() before optimizer.step()".
                if self.scaler.get_scale() >= scale_before_step:
                    any_optimizer_step = True

            total_loss += loss_out.total_loss.item()
            n_batches += 1
            if loss_out.age_loss is not None:
                total_age_loss += loss_out.age_loss.item()
                n_age_batches += 1
                eff_age_w += loss_out.effective_age_weight
                if loss_out.log_var_age is not None:
                    lv_age += loss_out.log_var_age
                with torch.no_grad():
                    valid = age_mask.bool()
                    if valid.any():
                        err = (outputs["age_output"]["q50"][valid] - age_target[valid]).abs()
                        age_abs_errors.append(err.detach().cpu())
            if loss_out.gender_loss is not None:
                total_gender_loss += loss_out.gender_loss.item()
                n_gender_batches += 1
                eff_gender_w += loss_out.effective_gender_weight
                if loss_out.log_var_gender is not None:
                    lv_gender += loss_out.log_var_gender
                with torch.no_grad():
                    valid = gender_mask.bool()
                    if valid.any():
                        probs = torch.softmax(outputs["gender_logits"][valid], dim=-1)
                        preds = probs.argmax(dim=-1)
                        confidence = probs.max(dim=-1).values
                        correct = preds == gender_target[valid]
                        gender_correct += correct.sum().item()
                        gender_total += int(valid.sum().item())
                        # Selective accuracy/coverage/abstention (confidence-threshold aware) are
                        # tracked purely for the live console line / history.csv -- checkpoint
                        # selection (_balanced_score, BestMetricTracker) always uses the raw,
                        # non-abstention-aware "gender_accuracy" below, unchanged from before.
                        accepted = confidence >= self.confidence_threshold
                        gender_correct_accepted += (correct & accepted).sum().item()
                        gender_total_accepted += int(accepted.sum().item())
                        gender_confidences.append(confidence.detach().cpu())
                        gender_true_labels.append(gender_target[valid].detach().cpu())
                        gender_pred_labels.append(preds.detach().cpu())

        gender_abstention_value = float("nan")
        if gender_confidences:
            all_confidence = torch.cat(gender_confidences)
            gender_abstention_value = float((all_confidence < self.confidence_threshold).float().mean().item())

        # Balanced accuracy / F1 (raw argmax, full coverage -- never
        # confidence-thresholded) let a live progress line show class-
        # imbalance-robust performance alongside raw/selective accuracy
        # without a separate evaluation pass; "n/a" (not 0.0) when no
        # gender-label batches occurred this epoch or a class is entirely
        # absent (see src.evaluation.metrics for the exact semantics).
        gender_balanced_acc_value = float("nan")
        gender_f1_value = float("nan")
        if gender_true_labels:
            y_true = torch.cat(gender_true_labels).numpy()
            y_pred = torch.cat(gender_pred_labels).numpy()
            gender_balanced_acc_value = gender_balanced_accuracy(y_true, y_pred)
            gender_f1_value = gender_precision_recall_f1(y_true, y_pred)["f1"]

        metrics = {
            "loss": total_loss / max(1, n_batches),
            "age_loss": total_age_loss / max(1, n_age_batches),
            "gender_loss": total_gender_loss / max(1, n_gender_batches),
            "effective_age_weight": eff_age_w / max(1, n_age_batches),
            "effective_gender_weight": eff_gender_w / max(1, n_gender_batches),
            "log_var_age": lv_age / max(1, n_age_batches),
            "log_var_gender": lv_gender / max(1, n_gender_batches),
            "age_mae": float(torch.cat(age_abs_errors).mean()) if age_abs_errors else float("nan"),
            "age_rmse": float(torch.sqrt((torch.cat(age_abs_errors) ** 2).mean())) if age_abs_errors else float("nan"),
            "gender_accuracy": gender_correct / max(1, gender_total) if gender_total else float("nan"),
            "gender_balanced_accuracy": gender_balanced_acc_value,
            "gender_f1": gender_f1_value,
            "gender_selective_accuracy": (
                gender_correct_accepted / max(1, gender_total_accepted) if gender_total_accepted else float("nan")
            ),
            "gender_abstention": gender_abstention_value,
            "gender_coverage": (
                1.0 - gender_abstention_value if gender_abstention_value == gender_abstention_value else float("nan")
            ),
            "optimizer_stepped": any_optimizer_step,
        }
        return metrics

    # The single, centralized main validation selection criterion. Both the
    # main "best" checkpoint AND early stopping use exactly this score and
    # mode (higher is better), so the checkpoint reported as "best" is always
    # the one training actually stopped at / around -- they can never diverge
    # (the previous bug: checkpoint selection on balanced_score but early
    # stopping on validation total loss). The separate age-MAE-best and
    # gender-accuracy-best checkpoints remain as diagnostics only.
    #
    # Score S = gender_accuracy - age_mae / age_max, using the raw (non-
    # selective, coverage-independent) validation gender accuracy -- never
    # the confidence-thresholded selective accuracy, whose value depends on
    # the abstention coverage. Selection uses validation data only.
    SELECTION_METRIC = "balanced_score"
    SELECTION_MODE = "max"

    def _balanced_score(self, age_mae: float, gender_acc: float, age_max: float) -> float:
        if age_mae != age_mae:  # NaN check
            return gender_acc if gender_acc == gender_acc else float("-inf")
        if gender_acc != gender_acc:
            return -age_mae
        normalized_mae = age_mae / max(age_max, 1e-6)
        return gender_acc - normalized_mae

    def train(self) -> dict:
        has_pretrained = bool(self.config["model"].get("pretrained_checkpoint"))
        stages = build_stage_plan(self.training_cfg, has_pretrained)
        age_max = self.config["model"]["age_head"].get("age_max", 120)
        total_epochs_planned = sum(stage.epochs for stage in stages)
        seed_display = self.training_cfg.get("seed", self.config.get("seed"))

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        start_line = (
            f"[{self.experiment_name} | seed={seed_display}] Starting training | device={self.device} | "
            f"train_samples={self.train_dataset_size} | val_samples={self.val_dataset_size} | "
            f"trainable_params={trainable_params:,}/{total_params:,} | "
            "checkpoint_selection=balanced_score (also tracked: age_mae, gender_accuracy)"
        )
        emit(start_line)
        self._write_run_manifest(stages, seed_display, trainable_params, total_params)

        loss_cfg = self.config["model"]["loss_balancing"]
        if loss_cfg["mode"] == "learned_uncertainty":
            warmup_epochs = loss_cfg.get("learned_uncertainty", {}).get("warmup_epochs", 0)
            if warmup_epochs > 0:
                warmup_line = (
                    f"[{self.experiment_name} | seed={seed_display}] Loss-balancing warmup: "
                    f"training with equal fixed weights for the first {warmup_epochs} epoch(s) "
                    "before switching to learned homoscedastic-uncertainty weighting."
                )
                emit(warmup_line)

        differential_lr_cfg = self.training_cfg.get("differential_lr", {})
        if differential_lr_cfg.get("enabled", False):
            diff_lr_line = (
                f"[{self.experiment_name} | seed={seed_display}] Differential learning rates enabled: "
                f"backbone_lr_multiplier={differential_lr_cfg.get('backbone_lr_multiplier', 0.1)} "
                "(backbone trains at a fraction of the stage LR; adapters/heads use the full stage LR)."
            )
            emit(diff_lr_line)

        global_epoch = 0
        for stage in stages:
            self.model.set_stage_trainable(stage.freeze_backbone, stage.unfreeze_layers)
            emit(format_stage_announcement(self.experiment_name, seed_display, stage.name, self.model))
            optimizer = _build_optimizer(
                self.model, stage.lr, self.training_cfg.get("weight_decay", 0.05), differential_lr_cfg,
            )
            scheduler = _build_scheduler(
                optimizer, stage.epochs, self.training_cfg["scheduler"].get("warmup_epochs", 1),
                self.training_cfg["scheduler"].get("warmup_start_factor", 0.1),
            )
            # Early stopping tracks the SAME centralized selection metric/mode
            # as checkpoint selection (see SELECTION_METRIC/SELECTION_MODE) --
            # not validation total loss -- so the two never disagree.
            early_stopping = EarlyStopping(
                patience=self.training_cfg.get("early_stopping_patience", 8), mode=self.SELECTION_MODE,
            )
            lr_groups = (
                {
                    "backbone": stage.lr * differential_lr_cfg.get("backbone_lr_multiplier", 0.1) if not stage.freeze_backbone else None,
                    "adapters_heads": stage.lr,
                }
                if differential_lr_cfg.get("enabled", False)
                else {"all": stage.lr}
            )

            for _ in range(stage.epochs):
                start = time.time()
                train_metrics = self._run_batches(self.train_loader, optimizer, global_epoch + 1)
                val_metrics = self._run_batches(self.val_loader, None, global_epoch + 1)
                current_lr = optimizer.param_groups[-1]["lr"]  # the "head" group's LR (== the only group without differential LR)
                if train_metrics["optimizer_stepped"]:
                    scheduler.step()
                elapsed = time.time() - start
                self.epoch_times.append(elapsed)
                global_epoch += 1

                self.history["train_loss"].append(train_metrics["loss"])
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_age_mae"].append(val_metrics["age_mae"])
                self.history["val_age_rmse"].append(val_metrics["age_rmse"])
                self.history["val_gender_accuracy"].append(val_metrics["gender_accuracy"])
                self.history["val_gender_balanced_accuracy"].append(val_metrics["gender_balanced_accuracy"])
                self.history["val_gender_f1"].append(val_metrics["gender_f1"])
                self.history["val_gender_selective_accuracy"].append(val_metrics["gender_selective_accuracy"])
                self.history["val_gender_coverage"].append(val_metrics["gender_coverage"])
                self.history["val_gender_abstention"].append(val_metrics["gender_abstention"])
                self.history["age_loss"].append(train_metrics["age_loss"])
                self.history["gender_loss"].append(train_metrics["gender_loss"])
                self.history["effective_age_weight"].append(train_metrics["effective_age_weight"])
                self.history["effective_gender_weight"].append(train_metrics["effective_gender_weight"])
                self.history["log_var_age"].append(train_metrics["log_var_age"])
                self.history["log_var_gender"].append(train_metrics["log_var_gender"])
                self.history["lr"].append(current_lr)
                self.history["epoch_time_seconds"].append(elapsed)

                balanced = self._balanced_score(val_metrics["age_mae"], val_metrics["gender_accuracy"], age_max)
                self._maybe_checkpoint("age_mae", val_metrics["age_mae"], global_epoch, val_metrics)
                self._maybe_checkpoint("gender_accuracy", val_metrics["gender_accuracy"], global_epoch, val_metrics)
                is_best_balanced = self._maybe_checkpoint("balanced_score", balanced, global_epoch, val_metrics)
                balanced_tracker = self.trackers["balanced_score"]

                emit(
                    format_epoch_report(
                        experiment_name=self.experiment_name, seed=seed_display, stage_name=stage.name,
                        epoch=global_epoch, total_epochs=total_epochs_planned,
                        train_metrics=train_metrics, val_metrics=val_metrics, lr_groups=lr_groups,
                        selection_score=balanced, is_best=is_best_balanced,
                        best_score=balanced_tracker.best_value, best_epoch=balanced_tracker.best_epoch,
                        early_stopping_bad_epochs=early_stopping.num_bad_epochs,
                        early_stopping_patience=early_stopping.patience, epoch_seconds=elapsed,
                        checkpoint_path=self._last_checkpoint_path,
                    )
                )

                self._write_incremental_history()
                self._write_status_atomic(stage.name, global_epoch, total_epochs_planned, early_stopping)
                self._write_last_checkpoint(stage.name, global_epoch, val_metrics)

                # Early stopping on the centralized selection score (higher is
                # better), skipping epochs whose score is NaN (a task absent
                # this run) rather than counting them as non-improvements.
                if not (balanced == balanced):
                    continue
                if early_stopping.step(balanced):
                    stop_line = f"[{self.experiment_name} | seed={seed_display}] Early stopping triggered at epoch {global_epoch}"
                    emit(stop_line)
                    break

        best_line = (
            f"[{self.experiment_name} | seed={seed_display}] Training complete | "
            f"best scores: {{'age_mae': {self.trackers['age_mae'].best_value}, "
            f"'gender_accuracy': {self.trackers['gender_accuracy'].best_value}, "
            f"'balanced_score': {self.trackers['balanced_score'].best_value}}} | "
            f"last checkpoint: {self._last_checkpoint_path}"
        )
        emit(best_line)

        return {"history": self.history, "epoch_times": self.epoch_times}

    def _maybe_checkpoint(self, metric_name: str, value: float, epoch: int, metrics: dict) -> bool:
        if value != value:  # NaN, task absent this run
            return False
        tracker = self.trackers[metric_name]
        improved = tracker.update(value, epoch)
        if improved:
            path = self.checkpoint_dir / f"{self.experiment_name}_best_{metric_name}.pt"
            save_checkpoint(path, self.model, None, epoch, metrics, self.config)
            self._last_checkpoint_path = path
        return improved

    def _write_run_manifest(self, stages: list[Stage], seed_display, trainable_params: int, total_params: int) -> None:
        """Written once, at the start of training -- static run metadata
        (never overwritten per-epoch, unlike status.json/history.*), so a
        notebook status-table scan can identify which experiment/seed/config
        a run directory belongs to without waiting for it to finish."""
        manifest = {
            "experiment_name": self.experiment_name,
            "seed": seed_display,
            "device": self.device,
            "train_samples": self.train_dataset_size,
            "val_samples": self.val_dataset_size,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "stages": [
                {"name": s.name, "epochs": s.epochs, "lr": s.lr, "freeze_backbone": s.freeze_backbone}
                for s in stages
            ],
            "total_epochs_planned": sum(s.epochs for s in stages),
            "checkpoint_selection_metric": self.SELECTION_METRIC,
            "git_commit_sha": git_commit_sha(),
            "dependency_versions": dependency_versions(),
            "started_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }
        tmp_path = self.run_manifest_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=str)
        tmp_path.replace(self.run_manifest_path)

    def _write_last_checkpoint(self, stage_name: str, epoch: int, val_metrics: dict) -> None:
        """Atomically overwrite a single ``*_last.pt`` after every epoch --
        a live progress/safety artifact (the most recent model state), kept
        deliberately separate from the per-metric ``_best_*.pt`` files this
        class already writes. This does not carry optimizer/scheduler state
        and is not wired into a resume path -- the core Trainer has no
        epoch-level resume today (restart-safety is at the stage level: a
        notebook re-run skips an experiment/seed whose checkpoint already
        exists, see docs/execution_modes.md "Resume safety"); this exists
        so a notebook status scan always has *some* current-state
        checkpoint to point at while a run is in progress, even before the
        first metric improvement is seen."""
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": None,
            "epoch": epoch,
            "stage": stage_name,
            "metrics": val_metrics,
            "config": self.config,
            "extra": {"family": "core", "experiment_name": self.experiment_name},
        }
        tmp_path = self.last_checkpoint_path.with_suffix(self.last_checkpoint_path.suffix + ".tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(self.last_checkpoint_path)

    def _write_incremental_history(self) -> None:
        """Rewrite history.csv/json after every epoch (not just at the end of
        train()), so a notebook can inspect progress -- or recover a
        partial run's history -- even if the process is interrupted
        mid-training. Rewriting the whole file each time (rather than
        appending) is simplest and cheap at these epoch counts, and avoids
        any header/row mismatch risk."""
        keys = list(self.history.keys())
        n_rows = len(self.history[keys[0]]) if keys else 0
        with open(self.history_csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(keys)
            for i in range(n_rows):
                writer.writerow([self.history[key][i] for key in keys])

        with open(self.history_json_path, "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2)

    def _write_status_atomic(self, stage_name: str, epoch: int, total_epochs_planned: int, early_stopping: EarlyStopping) -> None:
        """Write a live status file via write-temp-then-rename, so a reader
        never observes a half-written file (``Path.replace`` is atomic on
        both POSIX and Windows when source/destination are on the same
        filesystem, which they always are here)."""
        status = {
            "experiment_name": self.experiment_name,
            "stage": stage_name,
            "epoch": epoch,
            "total_epochs_planned": total_epochs_planned,
            "best_scores": {name: tracker.best_value for name, tracker in self.trackers.items()},
            "early_stopping_bad_epochs": early_stopping.num_bad_epochs,
            "early_stopping_patience": early_stopping.patience,
            "updated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        }
        tmp_path = self.status_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(status, fh, indent=2)
        tmp_path.replace(self.status_path)
