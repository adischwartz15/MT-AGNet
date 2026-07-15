"""An ImageNet-pretrained torchvision ResNet-18/50 backbone, wrapped with
this project's own adapters (``src/models/adapters.py``), heads
(``src/models/heads.py``), and loss balancing
(``src/losses/multitask_loss.py``).

Today it's mainly used as a frozen feature extractor for the
non-parametric ``frozen_backbone`` baseline (adapters off -- see
``src/evaluation/nonparametric/features.py``), which only calls
``encode()``/``forward()``. The freeze/unfreeze/parameter-group methods
below are for fine-tuning and aren't used by that path.

``torchvision`` is optional (see ``requirements-transfer.txt``) and is
only imported inside functions here, so core experiments still work
without it installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.adapters import AgeAdapter, GenderAdapter, IdentityAdapter
from src.models.heads import AgeQuantileHead, GenderClassificationHead

DEFAULT_MODEL_ID = "resnet18"

# torchvision model_id -> (builder function name, weights enum name, layer4 out-features).
_SUPPORTED_MODELS = {
    "resnet18": {"weights_enum": "ResNet18_Weights", "builder": "resnet18"},
    "resnet50": {"weights_enum": "ResNet50_Weights", "builder": "resnet50"},
}

# Canonical pretrained-source tags this project will ever accept for this
# extension -- ImageNet only (never a source that could leak the test set).
ALLOWED_PRETRAINED_SOURCES = frozenset({"imagenet1k_v1", "imagenet1k_v2"})


class MissingTorchvisionError(ImportError):
    """Raised when the pretrained-ResNet extension is selected but ``torchvision`` is not installed."""


class PretrainedSourceNotAllowedError(ValueError):
    """Raised when ``model.pretrained_resnet.pretrained_source`` is outside the ImageNet-only allow-list."""


class UnsupportedResNetModelError(ValueError):
    """Raised for a ``model_id`` other than the supported torchvision ResNet variants."""


class InvalidStageTransitionError(RuntimeError):
    """Raised for a stage transition that doesn't correspond to a real training phase."""


def validate_pretrained_source(source: str) -> None:
    if source not in ALLOWED_PRETRAINED_SOURCES:
        raise PretrainedSourceNotAllowedError(
            f"model.pretrained_resnet.pretrained_source='{source}' is not in the allow-list "
            f"{sorted(ALLOWED_PRETRAINED_SOURCES)}. This project only ever uses ImageNet-pretrained "
            "weights for this extension -- any other source risks leaking the test set."
        )


def _require_torchvision():
    try:
        import torchvision
    except ImportError as exc:
        raise MissingTorchvisionError(
            "The pretrained-ResNet transfer-learning extension requires torchvision. Install it with "
            "`pip install -r requirements-transfer.txt`."
        ) from exc
    if not hasattr(torchvision, "models"):
        raise MissingTorchvisionError(
            "A 'torchvision' module was importable but does not look like a real torchvision "
            "install (missing torchvision.models). Reinstall it with "
            "`pip install -r requirements-transfer.txt`."
        )
    return torchvision


@dataclass
class PretrainedResNetParameterBreakdown:
    """Parameter counts by component."""

    backbone_name: str
    backbone: int
    backbone_trainable: int
    adapters: int
    age_head: int
    gender_head: int
    log_variance: int
    total: int
    trainable_total: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "backbone_name": self.backbone_name,
            "backbone_parameters": self.backbone,
            "backbone_trainable_parameters": self.backbone_trainable,
            "adapter_parameters": self.adapters,
            "age_head_parameters": self.age_head,
            "gender_head_parameters": self.gender_head,
            "log_variance_parameters": self.log_variance,
            "total_parameters": self.total,
            "trainable_parameters": self.trainable_total,
        }


class PretrainedResNetFaceOnlyMultiTask(nn.Module):
    """ImageNet-pretrained torchvision ResNet-18/50 backbone + the project's
    existing adapters/heads/loss balancing -- the pretrained-ResNet bridge
    baseline. See this module's docstring for its role and limitations
    relative to the from-scratch Custom ResNet-18.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()

        model_cfg = config["model"]
        resnet_cfg = model_cfg.get("pretrained_resnet", {})
        self.model_id: str = resnet_cfg.get("model_id", DEFAULT_MODEL_ID)
        if self.model_id not in _SUPPORTED_MODELS:
            raise UnsupportedResNetModelError(
                f"model.pretrained_resnet.model_id='{self.model_id}' is not supported. "
                f"Supported: {sorted(_SUPPORTED_MODELS)}."
            )
        pretrained: bool = resnet_cfg.get("pretrained", True)
        pretrained_source: str = resnet_cfg.get("pretrained_source", "imagenet1k_v1")
        validate_pretrained_source(pretrained_source)
        self.pretrained_source = pretrained_source
        self.pretrained = pretrained

        torchvision = _require_torchvision()
        model_info = _SUPPORTED_MODELS[self.model_id]
        weights_enum = getattr(torchvision.models, model_info["weights_enum"])
        weight_tag = "IMAGENET1K_V2" if pretrained_source == "imagenet1k_v2" else "IMAGENET1K_V1"
        if not hasattr(weights_enum, weight_tag):
            weight_tag = "IMAGENET1K_V1"  # some architectures (e.g. resnet18) never shipped a V2 weight
        weights = getattr(weights_enum, weight_tag) if pretrained else None
        builder = getattr(torchvision.models, model_info["builder"])

        # No try/except: if a pretrained-weight download fails (offline,
        # network error, revoked URL), this must raise -- never silently
        # fall back to random initialization while still labeled "pretrained".
        self.backbone = builder(weights=weights)

        # The official weight-specific preprocessing metadata -- resolved
        # once here, reused for build_transforms() and never re-derived
        # elsewhere (see src/data/transforms.py::resolve_eval_transform).
        official_transform = (weights or weights_enum.IMAGENET1K_V1).transforms()
        self.input_size: int = official_transform.crop_size[0]
        resize_size = official_transform.resize_size[0]
        self.crop_pct: float = self.input_size / resize_size if resize_size else 1.0
        self.mean: tuple[float, ...] = tuple(official_transform.mean)
        self.std: tuple[float, ...] = tuple(official_transform.std)
        self._interpolation_mode = official_transform.interpolation

        self.embedding_dim = self._discover_and_verify_embedding_dim()
        # Replace the ImageNet classifier head with identity -- forward(x)
        # then returns the pooled feature vector directly.
        self.backbone.fc = nn.Identity()

        adapters_cfg = model_cfg.get("adapters", {})
        bottleneck_ratio = adapters_cfg.get("bottleneck_ratio", 4)
        bottleneck_dim = adapters_cfg.get("bottleneck_dim") or max(1, round(self.embedding_dim / bottleneck_ratio))
        self.bottleneck_ratio = bottleneck_ratio
        self.bottleneck_dim = bottleneck_dim
        adapter_dropout = adapters_cfg.get("dropout", 0.1)
        adapters_enabled = adapters_cfg.get("enabled", True)

        if adapters_enabled:
            self.age_adapter: nn.Module = AgeAdapter(self.embedding_dim, bottleneck_dim, adapter_dropout)
            self.gender_adapter: nn.Module = GenderAdapter(self.embedding_dim, bottleneck_dim, adapter_dropout)
        else:
            self.age_adapter = IdentityAdapter()
            self.gender_adapter = IdentityAdapter()
        self.adapters_enabled = adapters_enabled

        age_head_cfg = model_cfg.get("age_head", {})
        gender_head_cfg = model_cfg.get("gender_head", {})
        self.age_head = AgeQuantileHead(
            input_dim=self.embedding_dim,
            hidden_dim=age_head_cfg.get("hidden_dim", 128),
            dropout=age_head_cfg.get("dropout", 0.1),
            age_min=age_head_cfg.get("age_min", 0),
            age_max=age_head_cfg.get("age_max", 120),
        )
        self.gender_head = GenderClassificationHead(
            input_dim=self.embedding_dim,
            hidden_dim=gender_head_cfg.get("hidden_dim", 128),
            dropout=gender_head_cfg.get("dropout", 0.1),
            num_classes=gender_head_cfg.get("num_classes", 2),
        )

        loss_balancing_cfg = model_cfg.get("loss_balancing", {})
        self.loss_balancing_mode = loss_balancing_cfg.get("mode", "learned_uncertainty")
        if self.loss_balancing_mode == "learned_uncertainty":
            init_cfg = loss_balancing_cfg.get("learned_uncertainty", {})
            self.log_var_age = nn.Parameter(torch.tensor(float(init_cfg.get("init_log_var_age", 0.0))))
            self.log_var_gender = nn.Parameter(torch.tensor(float(init_cfg.get("init_log_var_gender", 0.0))))
        else:
            self.log_var_age = None
            self.log_var_gender = None

        # Starts fully trainable; a caller doing staged fine-tuning calls
        # freeze_backbone() explicitly before training begins.

    def _discover_and_verify_embedding_dim(self) -> int:
        """Dry-run forward pass that cross-checks torchvision's declared
        ``fc.in_features`` against the actual pooled output shape before
        wiring up adapters/heads."""
        declared_dim = getattr(self.backbone.fc, "in_features", None)
        if declared_dim is None:
            raise ValueError(f"torchvision model '{self.model_id}' has no fc.in_features to read.")

        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(2, 3, self.input_size, self.input_size)
            # forward() through the ORIGINAL (not-yet-replaced) fc still
            # returns 1000-d logits at this point; use avgpool+flatten
            # directly to get the pooled embedding this dry run verifies.
            features = self.backbone.avgpool(
                self.backbone.layer4(
                    self.backbone.layer3(
                        self.backbone.layer2(
                            self.backbone.layer1(
                                self.backbone.maxpool(
                                    self.backbone.relu(self.backbone.bn1(self.backbone.conv1(dummy)))
                                )
                            )
                        )
                    )
                )
            )
            pooled = torch.flatten(features, 1)
            if pooled.ndim != 2:
                raise ValueError(
                    f"torchvision model '{self.model_id}' pooled output has shape {tuple(pooled.shape)}, "
                    "expected [batch, embedding_dim]."
                )
        if pooled.shape[1] != declared_dim:
            raise ValueError(
                f"Embedding dimension mismatch for '{self.model_id}': fc.in_features={declared_dim}, "
                f"actual pooled output shape[1]={pooled.shape[1]}."
            )
        self.backbone.train()
        return int(declared_dim)

    # -- encode/forward: same output-dict contract as MultiTaskFaceModel ------

    def encode(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        z_shared = self.backbone(images)
        assert z_shared.ndim == 2 and z_shared.shape[1] == self.embedding_dim, (
            f"ResNet backbone output shape {tuple(z_shared.shape)} does not match the "
            f"discovered embedding_dim={self.embedding_dim}."
        )
        z_age = self.age_adapter(z_shared)
        z_gender = self.gender_adapter(z_shared)
        return {"shared_embedding": z_shared, "age_embedding": z_age, "gender_embedding": z_gender}

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        embeddings = self.encode(images)
        age_output = self.age_head(embeddings["age_embedding"])
        gender_logits = self.gender_head(embeddings["gender_embedding"])
        return {**embeddings, "age_output": age_output, "gender_logits": gender_logits}

    # -- freeze/unfreeze + parameter groups -----------------

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_stages(self, n: int) -> None:
        """Unfreeze only the last ``n`` of torchvision ResNet's
        ``[layer1, layer2, layer3, layer4]`` stages (plus nothing further
        downstream needs unfreezing -- torchvision's ResNet has no
        post-layer4 normalization before the now-identity ``fc``)."""
        if n < 1:
            raise InvalidStageTransitionError(f"unfreeze_last_stages(n={n}) requires n >= 1.")
        stages = [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]
        if n > len(stages):
            raise InvalidStageTransitionError(
                f"unfreeze_last_stages(n={n}) requested more stages than exist ({len(stages)})."
            )
        self.freeze_backbone()
        for stage in stages[-n:]:
            for param in stage.parameters():
                param.requires_grad = True

    def get_parameter_groups(
        self, backbone_lr: float, adapter_lr: float, head_lr: float, balance_lr: float, weight_decay: float,
    ) -> list[dict]:
        """Per-component learning rates, with zero weight decay on biases/
        normalization/scalar loss-balancing parameters (see
        src/training/optim.py::build_param_groups)."""
        from src.training.optim import build_param_groups

        adapter_ids = {id(p) for p in self.age_adapter.parameters()} | {id(p) for p in self.gender_adapter.parameters()}
        head_ids = {id(p) for p in self.age_head.parameters()} | {id(p) for p in self.gender_head.parameters()}
        balance_ids = set()
        if self.log_var_age is not None:
            balance_ids = {id(self.log_var_age), id(self.log_var_gender)}

        def lr_for(_name, param):
            pid = id(param)
            if pid in adapter_ids:
                return adapter_lr
            if pid in head_ids:
                return head_lr
            if pid in balance_ids:
                return balance_lr
            return backbone_lr

        try:
            return build_param_groups(self.named_parameters(), lr_for, weight_decay)
        except ValueError as exc:
            raise InvalidStageTransitionError(str(exc)) from exc

    # -- introspection ------------------------------------------------------------

    def parameter_breakdown(self) -> PretrainedResNetParameterBreakdown:
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        adapter_params = 0
        if hasattr(self.age_adapter, "num_parameters"):
            adapter_params += self.age_adapter.num_parameters()
        if hasattr(self.gender_adapter, "num_parameters"):
            adapter_params += self.gender_adapter.num_parameters()
        age_head_params = sum(p.numel() for p in self.age_head.parameters())
        gender_head_params = sum(p.numel() for p in self.gender_head.parameters())
        log_var_params = 0
        if self.log_var_age is not None:
            log_var_params = self.log_var_age.numel() + self.log_var_gender.numel()

        total = backbone_params + adapter_params + age_head_params + gender_head_params + log_var_params
        trainable_total = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return PretrainedResNetParameterBreakdown(
            backbone_name=self.model_id,
            backbone=backbone_params,
            backbone_trainable=backbone_trainable,
            adapters=adapter_params,
            age_head=age_head_params,
            gender_head=gender_head_params,
            log_variance=log_var_params,
            total=total,
            trainable_total=trainable_total,
        )

    def build_transforms(self):
        """Build (TrainTransform, EvalTransform) using this backbone's own
        OFFICIAL torchvision weight-specific preprocessing (input size,
        mean/std, interpolation, crop_pct) -- never this project's 128px/
        existing normalization defaults. See
        src/data/transforms.py::resolve_eval_transform, the single place
        every evaluation/calibration/robustness/prediction-export path
        resolves this from."""
        from PIL import Image
        from torchvision.transforms import InterpolationMode

        from src.data.transforms import EvalTransform, TrainTransform

        _TV_TO_PIL = {
            InterpolationMode.BILINEAR: Image.BILINEAR, InterpolationMode.BICUBIC: Image.BICUBIC,
            InterpolationMode.NEAREST: Image.NEAREST,
        }
        interpolation = _TV_TO_PIL.get(self._interpolation_mode, Image.BILINEAR)

        train_transform = TrainTransform(self.input_size, mean=self.mean, std=self.std, interpolation=interpolation, crop_pct=self.crop_pct)
        eval_transform = EvalTransform(self.input_size, mean=self.mean, std=self.std, interpolation=interpolation, crop_pct=self.crop_pct)
        return train_transform, eval_transform


def build_pretrained_resnet_model(config: dict) -> PretrainedResNetFaceOnlyMultiTask:
    """Factory for :class:`PretrainedResNetFaceOnlyMultiTask`."""
    return PretrainedResNetFaceOnlyMultiTask(config)
