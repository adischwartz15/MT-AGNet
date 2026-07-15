"""Progressive fine-tuning stage planning.

Freezing a randomly initialized backbone is not scientifically meaningful
(there is nothing pretrained worth "preserving" by freezing it), so
staged freeze/unfreeze (Stage A -> B -> C) is only used when the backbone
was initialized from a checkpoint produced by this repository's
self-supervised pretraining, or a compatible user-supplied local
checkpoint. Otherwise this module returns a single supervised warm-up
stage and the caller must surface the accompanying warning.

Stage names are matched against each backbone's actual top-level submodule
names (see ``MultiTaskFaceModel.set_stage_trainable``), so this plan works
unchanged for either backbone (``custom_resnet18`` or ``simple_cnn``, see
``src/models/backbone_factory.py``): both expose ``layer1``-``layer4``.
``simple_cnn`` has no separate ``stem`` submodule (``layer1`` fulfills
that role), so a ``"stem"`` entry in ``unfreeze_layers`` simply matches no
parameter for that backbone -- a harmless no-op, not an error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

NO_PRETRAINED_WARNING = (
    "No pretrained checkpoint was supplied (model.pretrained_checkpoint is null). "
    "Progressive freezing of a randomly initialized backbone is not scientifically "
    "meaningful, so training will run a single supervised warm-up stage over the "
    "full model instead of Stage A/B/C freezing. To use staged fine-tuning, first "
    "run scripts/pretrain.py (self-supervised) or supply a compatible local "
    "checkpoint via model.pretrained_checkpoint."
)


@dataclass
class Stage:
    name: str
    epochs: int
    lr: float
    freeze_backbone: bool
    unfreeze_layers: list[str]


def build_stage_plan(training_cfg: dict, has_pretrained_checkpoint: bool) -> list[Stage]:
    """Return the ordered list of training stages to run."""
    if not has_pretrained_checkpoint:
        logger.warning(NO_PRETRAINED_WARNING)
        warm = training_cfg.get("warm_up_from_scratch", {"epochs": 12, "lr": 1.0e-3})
        return [
            Stage(
                name="Warm-up (no pretrained backbone): full model, no freezing",
                epochs=warm.get("epochs", 12),
                lr=warm.get("lr", 1.0e-3),
                freeze_backbone=False,
                unfreeze_layers=["stem", "layer1", "layer2", "layer3", "layer4"],
            )
        ]

    stages_cfg = training_cfg["stages"]
    stages = []
    for key in ("stage_a", "stage_b", "stage_c"):
        cfg = stages_cfg[key]
        stages.append(
            Stage(
                name=cfg["name"],
                epochs=cfg["epochs"],
                lr=cfg["lr"],
                freeze_backbone=cfg["freeze_backbone"],
                unfreeze_layers=list(cfg.get("unfreeze_layers", [])),
            )
        )
    return stages
