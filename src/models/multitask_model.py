"""Assembles the backbone, task adapters, and task heads into the full model.

Supports three architecture modes (selected via ``model.architecture``),
which correspond directly to Experiments A/B/C-D in
``configs/experiments.yaml``:

* ``separate``            -- two independent backbones, one per task
  (Experiment A).
* ``shared_no_adapters``  -- one shared backbone, heads attached directly
  to the shared embedding (Experiment B).
* ``shared_adapters``     -- one shared backbone, each task reads through
  its own residual bottleneck adapter before its head (Experiments C/D).

The backbone itself is pluggable via ``model.backbone.name``
(``src/models/backbone_factory.py``): ``custom_resnet18`` (the project's
main research backbone, default) or ``simple_cnn`` (a conventional
non-residual CNN used only as a controlled baseline, see
``exp_0_simple_cnn_shared_adapters_learned_balance``).

When ``model.loss_balancing.mode == "learned_uncertainty"``, this module
also owns the two trainable task log-variance parameters (see
``src/losses/multitask_loss.py`` for how they combine the per-task losses).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.models.adapters import AgeAdapter, GenderAdapter, IdentityAdapter
from src.models.backbone_factory import build_backbone
from src.models.heads import AgeQuantileHead, GenderClassificationHead

VALID_ARCHITECTURES = ("separate", "shared_no_adapters", "shared_adapters")


@dataclass
class ParameterBreakdown:
    """Parameter counts by component, used for architecture-comparison reports.

    ``as_dict()`` keys match what ``src/evaluation/comparison.py`` and the
    generated architecture report expect, including the plain-CNN-vs-ResNet
    backbone comparison section.
    """

    backbone_name: str
    backbone: int
    adapters: int
    age_head: int
    gender_head: int
    log_variance: int = 0
    total: int = field(init=False)

    def __post_init__(self) -> None:
        self.total = self.backbone + self.adapters + self.age_head + self.gender_head + self.log_variance

    def as_dict(self) -> dict[str, int | str]:
        return {
            "backbone_name": self.backbone_name,
            "backbone_parameters": self.backbone,
            "adapter_parameters": self.adapters,
            "age_head_parameters": self.age_head,
            "gender_head_parameters": self.gender_head,
            "log_variance_parameters": self.log_variance,
            "total_parameters": self.total,
        }


class MultiTaskFaceModel(nn.Module):
    """Multi-task age + dataset gender-label model with a pluggable backbone."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        model_cfg = config["model"]
        architecture = model_cfg.get("architecture", "shared_adapters")
        if architecture not in VALID_ARCHITECTURES:
            raise ValueError(f"Unknown architecture '{architecture}', expected one of {VALID_ARCHITECTURES}")
        self.architecture = architecture

        backbone_cfg = model_cfg["backbone"]
        embedding_dim = backbone_cfg.get("embedding_dim", 512)
        self.backbone_name: str = backbone_cfg.get("name", "custom_resnet18")
        adapters_cfg = model_cfg.get("adapters", {})
        adapters_enabled = adapters_cfg.get("enabled", True) and architecture == "shared_adapters"
        bottleneck_dim = adapters_cfg.get("bottleneck_dim", 128)
        adapter_dropout = adapters_cfg.get("dropout", 0.1)

        age_head_cfg = model_cfg.get("age_head", {})
        gender_head_cfg = model_cfg.get("gender_head", {})

        if architecture == "separate":
            self.age_backbone: nn.Module = build_backbone(backbone_cfg)
            self.gender_backbone: nn.Module = build_backbone(backbone_cfg)
            self.backbone = None
        else:
            self.backbone = build_backbone(backbone_cfg)
            self.age_backbone = self.backbone
            self.gender_backbone = self.backbone

        if adapters_enabled:
            self.age_adapter: nn.Module = AgeAdapter(embedding_dim, bottleneck_dim, adapter_dropout)
            self.gender_adapter: nn.Module = GenderAdapter(embedding_dim, bottleneck_dim, adapter_dropout)
        else:
            self.age_adapter = IdentityAdapter()
            self.gender_adapter = IdentityAdapter()
        self.adapters_enabled = adapters_enabled

        self.age_head = AgeQuantileHead(
            input_dim=embedding_dim,
            hidden_dim=age_head_cfg.get("hidden_dim", 128),
            dropout=age_head_cfg.get("dropout", 0.1),
            age_min=age_head_cfg.get("age_min", 0),
            age_max=age_head_cfg.get("age_max", 120),
        )
        self.gender_head = GenderClassificationHead(
            input_dim=embedding_dim,
            hidden_dim=gender_head_cfg.get("hidden_dim", 128),
            dropout=gender_head_cfg.get("dropout", 0.1),
            num_classes=gender_head_cfg.get("num_classes", 2),
        )

        loss_balancing_cfg = model_cfg.get("loss_balancing", {})
        self.loss_balancing_mode = loss_balancing_cfg.get("mode", "fixed")
        if self.loss_balancing_mode == "learned_uncertainty":
            init_cfg = loss_balancing_cfg.get("learned_uncertainty", {})
            self.log_var_age = nn.Parameter(
                torch.tensor(float(init_cfg.get("init_log_var_age", 0.0)))
            )
            self.log_var_gender = nn.Parameter(
                torch.tensor(float(init_cfg.get("init_log_var_gender", 0.0)))
            )
        else:
            self.log_var_age = None
            self.log_var_gender = None

    def encode(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return shared/task embeddings before task heads (used by kNN + representation analysis)."""
        if self.architecture == "separate":
            z_age_shared = self.age_backbone(images)
            z_gender_shared = self.gender_backbone(images)
        else:
            z_shared = self.backbone(images)
            z_age_shared = z_shared
            z_gender_shared = z_shared

        z_age = self.age_adapter(z_age_shared)
        z_gender = self.gender_adapter(z_gender_shared)
        return {
            "shared_embedding": z_age_shared if self.architecture != "separate" else None,
            "age_embedding": z_age,
            "gender_embedding": z_gender,
        }

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = self.encode(images)
        age_output = self.age_head(embeddings["age_embedding"])
        gender_logits = self.gender_head(embeddings["gender_embedding"])
        return {
            **embeddings,
            "age_output": age_output,
            "gender_logits": gender_logits,
        }

    def set_stage_trainable(self, freeze_backbone: bool, unfreeze_layers: list[str]) -> None:
        """Apply Stage A/B/C progressive-freezing rules to the backbone(s).

        Adapters and heads are always trainable. When ``freeze_backbone`` is
        True, only the layers named in ``unfreeze_layers`` (e.g. ``["layer4"]``)
        stay trainable; everything else in the backbone is frozen.
        """
        backbones = (
            [self.age_backbone, self.gender_backbone]
            if self.architecture == "separate"
            else [self.backbone]
        )
        for backbone in backbones:
            for name, param in backbone.named_parameters():
                if not freeze_backbone:
                    param.requires_grad = True
                    continue
                layer_name = name.split(".")[0]
                param.requires_grad = layer_name in unfreeze_layers

        for module in (self.age_adapter, self.gender_adapter, self.age_head, self.gender_head):
            for param in module.parameters():
                param.requires_grad = True

    def backbone_parameters(self):
        """Yield every backbone parameter exactly once, regardless of architecture mode.

        Used by the trainer's differential-learning-rate optimizer setup
        (a lower LR for the backbone than for adapters/heads) so that logic
        doesn't need its own copy of the shared-vs-separate dispatch that
        :meth:`set_stage_trainable` already implements.
        """
        backbones = (
            [self.age_backbone, self.gender_backbone]
            if self.architecture == "separate"
            else [self.backbone]
        )
        seen_ids: set[int] = set()
        for backbone in backbones:
            for param in backbone.parameters():
                if id(param) not in seen_ids:
                    seen_ids.add(id(param))
                    yield param

    def parameter_breakdown(self) -> ParameterBreakdown:
        if self.architecture == "separate":
            backbone_params = sum(p.numel() for p in self.age_backbone.parameters()) + sum(
                p.numel() for p in self.gender_backbone.parameters()
            )
        else:
            backbone_params = sum(p.numel() for p in self.backbone.parameters())

        adapter_params = 0
        if isinstance(self.age_adapter, nn.Module) and hasattr(self.age_adapter, "num_parameters"):
            adapter_params += self.age_adapter.num_parameters()
        if isinstance(self.gender_adapter, nn.Module) and hasattr(self.gender_adapter, "num_parameters"):
            adapter_params += self.gender_adapter.num_parameters()

        age_head_params = sum(p.numel() for p in self.age_head.parameters())
        gender_head_params = sum(p.numel() for p in self.gender_head.parameters())
        log_var_params = 0
        if self.log_var_age is not None:
            log_var_params = self.log_var_age.numel() + self.log_var_gender.numel()

        return ParameterBreakdown(
            backbone_name=self.backbone_name,
            backbone=backbone_params, adapters=adapter_params,
            age_head=age_head_params, gender_head=gender_head_params,
            log_variance=log_var_params,
        )


def build_multitask_model(config: dict) -> MultiTaskFaceModel:
    """Factory that builds the model and optionally loads a repo-produced checkpoint.

    Only checkpoints produced by this repository's own training/pretraining
    scripts (or a compatible local checkpoint explicitly pointed to by the
    user) are ever loaded. No weights are downloaded automatically.
    """
    model = MultiTaskFaceModel(config)
    checkpoint_path = config["model"].get("pretrained_checkpoint")
    if checkpoint_path:
        load_backbone_checkpoint(model, checkpoint_path)
    return model


def load_backbone_checkpoint(model: MultiTaskFaceModel, checkpoint_path: str) -> None:
    """Load backbone weights from a local checkpoint file (never downloaded)."""
    import os

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Pretrained checkpoint '{checkpoint_path}' not found. This repository never "
            "downloads pretrained weights automatically -- point 'model.pretrained_checkpoint' "
            "at a checkpoint produced by scripts/pretrain.py or scripts/train.py."
        )
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("encoder_state_dict", state)
    backbones = (
        [model.age_backbone, model.gender_backbone]
        if model.architecture == "separate"
        else [model.backbone]
    )
    for backbone in backbones:
        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            import logging

            logging.getLogger(__name__).warning(
                "Partial checkpoint load: missing=%s unexpected=%s", missing, unexpected
            )
