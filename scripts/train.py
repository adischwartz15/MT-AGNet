#!/usr/bin/env python
"""CLI: train the multi-task face model (single configuration, not the full ablation suite).

For the full architecture ablation suite (Experiments A-F) use
scripts/run_experiments.py instead, which calls this same training path
once per experiment config.

Usage:
    python scripts/train.py [--experiment-name multitask] [--set model.architecture=shared_adapters]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import build_datasets
from src.data.transforms import EvalTransform, TrainTransform
from src.models.multitask_model import build_multitask_model
from src.training.trainer import Trainer
from src.utils.config import REPO_ROOT, load_full_config, parse_cli_overrides, resolve_device
from src.utils.io import save_json
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed
from src.utils.visualization import plot_loss_balancing, plot_training_curves

logger = get_logger("scripts.train")


def run_training(config: dict, experiment_name: str) -> dict:
    set_global_seed(config["training"].get("seed", config["seed"]))
    device = resolve_device(config["device"])

    splits_path = REPO_ROOT / config["paths"]["splits_dir"] / "full_metadata_with_splits.csv"
    if not splits_path.exists():
        raise FileNotFoundError(f"No prepared split found at {splits_path}. Run 'make prepare-data' first.")

    df = pd.read_csv(splits_path)
    image_size = config["dataset"]["image_size"]
    datasets = build_datasets(df, TrainTransform(image_size), EvalTransform(image_size))

    gender_class_weights = None
    weights_cfg = config["model"]["gender_head"].get("class_weights")
    if weights_cfg:
        import torch

        gender_class_weights = torch.tensor(weights_cfg, dtype=torch.float32)

    model = build_multitask_model(config)
    checkpoint_dir = REPO_ROOT / config["paths"]["checkpoint_dir"]
    output_dir = REPO_ROOT / config["paths"]["output_dir"]
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model, config, datasets["train"], datasets["validation"], device=device,
        checkpoint_dir=checkpoint_dir, experiment_name=experiment_name,
        gender_class_weights=gender_class_weights, output_dir=output_dir,
    )
    result = trainer.train()
    # trainer.train() already wrote metrics/{experiment_name}_history.{csv,json}
    # incrementally after every epoch (see src/training/trainer.py) -- no
    # need to write history.json again here.
    save_json(
        {"mean_epoch_time_seconds": float(np.mean(result["epoch_times"])) if result["epoch_times"] else None},
        metrics_dir / f"{experiment_name}_timing.json",
    )
    plot_training_curves(result["history"], plots_dir / f"{experiment_name}_training_curves.png")
    plot_loss_balancing(result["history"], plots_dir / f"{experiment_name}_loss_balancing.png")

    breakdown = model.parameter_breakdown().as_dict()
    save_json(breakdown, metrics_dir / f"{experiment_name}_parameter_breakdown.json")
    logger.info("Parameter breakdown for %s: %s", experiment_name, breakdown)

    return {"history": result["history"], "parameter_breakdown": breakdown}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", default="multitask")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    config = load_full_config(overrides=parse_cli_overrides(args.overrides))
    try:
        run_training(config, args.experiment_name)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
