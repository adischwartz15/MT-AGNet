#!/usr/bin/env python
"""CLI: optional SimCLR-style self-supervised pretraining of the Custom ResNet-18 encoder.

Requires data/splits/full_metadata_with_splits.csv (run scripts/prepare_data.py first).
Compute note: contrastive pretraining is significantly more compute-hungry
than supervised fine-tuning; see docs/reproducibility.md.

Usage:
    python scripts/pretrain.py [--set pretrain.epochs=5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import SimCLRPretrainDataset
from src.data.transforms import SimCLRTransform
from src.training.pretrain import pretrain_simclr
from src.utils.config import REPO_ROOT, load_full_config, parse_cli_overrides, resolve_device
from src.utils.io import save_json
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger("scripts.pretrain")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    config = load_full_config(overrides=parse_cli_overrides(args.overrides))
    set_global_seed(config["seed"])
    device = resolve_device(config["device"])

    splits_path = REPO_ROOT / config["paths"]["splits_dir"] / "full_metadata_with_splits.csv"
    if not splits_path.exists():
        logger.error("No prepared split found at %s. Run 'make prepare-data' first.", splits_path)
        return 1

    df = pd.read_csv(splits_path)
    train_df = df[df["split"] == "train"]

    image_size = config["dataset"]["image_size"]
    dataset = SimCLRPretrainDataset(train_df, SimCLRTransform(image_size))

    checkpoint_dir = REPO_ROOT / config["paths"]["checkpoint_dir"]
    result = pretrain_simclr(config["model"]["backbone"], config["pretrain"], dataset, device, checkpoint_dir)

    output_dir = REPO_ROOT / config["paths"]["output_dir"] / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(result["history"], output_dir / "pretrain_history.json")

    print(f"Pretraining complete. Encoder checkpoint: {result['checkpoint_path']}")
    print(f"Set model.pretrained_checkpoint: {result['checkpoint_path']} to use it in training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
