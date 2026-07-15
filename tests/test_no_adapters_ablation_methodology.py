"""Tests for T3 (final-run hardening) -- the no-adapters ablation
(exp_b_shared_no_adapters) must be a genuinely controlled ablation:

    shared TRAINABLE backbone, same task heads, same loss design, same
    optimizer/training protocol, NO task-specific adapters

not "remove adapters + freeze the backbone + train only heads" (which
would change two variables at once: adapters AND backbone trainability).

Audit finding: this repository's exp_b_shared_no_adapters was already
correctly defined this way (see configs/experiments.yaml + src/models/
multitask_model.py's IdentityAdapter pass-through + src/training/stages.py,
which never freezes a from-scratch/no-pretrained-checkpoint backbone at
all). These tests lock that in, per the mission's explicit requirement to
"add a test proving that the no-adapters experiment: contains no adapter
parameters; still has trainable backbone parameters; uses the intended
heads and loss-balancing mode."
"""

from __future__ import annotations

from src.models.adapters import IdentityAdapter
from src.models.multitask_model import build_multitask_model
from src.training.stages import build_stage_plan
from src.utils.config import CONFIG_DIR, load_config


def _exp_b_config():
    """The real exp_b_shared_no_adapters config, merged exactly the way
    scripts/run_experiments.py merges it (default+data+model+training,
    then the experiment's own overrides from configs/experiments.yaml) --
    not a hand-rolled test dict, so a drift in the real config would fail
    this test."""
    experiments_cfg = load_config(CONFIG_DIR / "experiments.yaml")
    overrides = experiments_cfg["experiments"]["exp_b_shared_no_adapters"]["overrides"]
    return load_config(CONFIG_DIR / "data.yaml", CONFIG_DIR / "model.yaml", CONFIG_DIR / "training.yaml", overrides=overrides)


def test_exp_b_config_has_no_adapters_and_fixed_loss():
    config = _exp_b_config()
    assert config["model"]["architecture"] == "shared_no_adapters"
    assert config["model"]["adapters"]["enabled"] is False
    assert config["model"]["loss_balancing"]["mode"] == "fixed"


def test_exp_b_model_contains_no_adapter_parameters():
    config = _exp_b_config()
    config["dataset"]["image_size"] = 32  # fast to construct
    model = build_multitask_model(config)
    assert isinstance(model.age_adapter, IdentityAdapter)
    assert isinstance(model.gender_adapter, IdentityAdapter)
    # IdentityAdapter is a pass-through with zero learnable parameters.
    assert sum(p.numel() for p in model.age_adapter.parameters()) == 0
    assert sum(p.numel() for p in model.gender_adapter.parameters()) == 0


def test_exp_b_backbone_remains_trainable_across_the_full_schedule():
    """The critical controlled-ablation property: the backbone must be
    trainable according to the SAME schedule a corresponding adapter
    experiment (e.g. exp_c/exp_d) uses -- never frozen throughout training
    just because adapters are absent. Neither exp_b nor exp_c/exp_d sets
    model.pretrained_checkpoint, so both get the identical from-scratch
    stage plan (a single non-frozen warm-up stage -- see
    src/training/stages.py::build_stage_plan, which never freezes a
    randomly initialized backbone)."""
    config = _exp_b_config()
    assert not config["model"].get("pretrained_checkpoint")
    stages = build_stage_plan(config["training"], has_pretrained_checkpoint=False)
    assert len(stages) == 1
    assert stages[0].freeze_backbone is False

    config["dataset"]["image_size"] = 32
    model = build_multitask_model(config)
    backbone_params = list(model.backbone_parameters())
    assert len(backbone_params) > 0
    assert all(p.requires_grad for p in backbone_params)


def test_exp_b_uses_the_same_stage_plan_as_exp_d():
    """Same optimizer/training protocol as the corresponding adapter
    experiment -- both from-scratch, both get the identical stage plan."""
    experiments_cfg = load_config(CONFIG_DIR / "experiments.yaml")
    exp_b_overrides = experiments_cfg["experiments"]["exp_b_shared_no_adapters"]["overrides"]
    exp_d_overrides = experiments_cfg["experiments"]["exp_d_shared_adapters_learned_balance"]["overrides"]
    config_b = load_config(CONFIG_DIR / "data.yaml", CONFIG_DIR / "model.yaml", CONFIG_DIR / "training.yaml", overrides=exp_b_overrides)
    config_d = load_config(CONFIG_DIR / "data.yaml", CONFIG_DIR / "model.yaml", CONFIG_DIR / "training.yaml", overrides=exp_d_overrides)

    stages_b = build_stage_plan(config_b["training"], has_pretrained_checkpoint=False)
    stages_d = build_stage_plan(config_d["training"], has_pretrained_checkpoint=False)
    assert [(s.epochs, s.lr, s.freeze_backbone) for s in stages_b] == [(s.epochs, s.lr, s.freeze_backbone) for s in stages_d]
    # Same backbone family, same heads config, same optimizer/scheduler config -- only
    # adapters.enabled and loss_balancing.mode differ between the two experiments.
    assert config_b["model"]["backbone"]["name"] == config_d["model"]["backbone"]["name"]
    assert config_b["model"]["age_head"] == config_d["model"]["age_head"]
    assert config_b["model"]["gender_head"] == config_d["model"]["gender_head"]
    assert config_b["training"]["scheduler"] == config_d["training"]["scheduler"]


def test_exp_b_uses_intended_heads():
    from src.models.heads import AgeQuantileHead, GenderClassificationHead

    config = _exp_b_config()
    config["dataset"]["image_size"] = 32
    model = build_multitask_model(config)
    assert isinstance(model.age_head, AgeQuantileHead)
    assert isinstance(model.gender_head, GenderClassificationHead)
