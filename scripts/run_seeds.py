#!/usr/bin/env python
"""CLI: train and evaluate one experiment across multiple seeds, for mean +/- std reporting.

A single training run cannot distinguish "this architecture is better"
from "this particular random initialization got lucky". This script
reuses configs/experiments.yaml's overrides for a named experiment,
trains it once per seed, and for each seed:

    1. writes the checkpoint into an isolated
       ``experiments/<experiment>/seed_<seed>/checkpoints`` directory
       (never a single shared ``checkpoints/`` directory two seeds could
       collide in beyond their filename suffix),
    2. fits a conformal calibration artifact for *that exact checkpoint*
       into that seed's own isolated ``.../calibration`` directory
       (never a shared global ``outputs/calibration``), and
    3. evaluates the checkpoint with that calibration applied, saving
       metrics/plots under that seed's own isolated ``.../metrics`` and
       ``.../plots``.

scripts/evaluate.py validates the calibration artifact's recorded
checkpoint/split provenance against what's actually being evaluated and
fails loudly on any mismatch (see
src/evaluation/calibration.py:validate_calibration_artifact) -- this is
what makes cross-seed calibration contamination impossible even if a
future change to this script's isolation logic has a bug.

Nothing here computes or renders the cross-seed aggregate itself -- see
src/evaluation/comparison.py:aggregate_seed_metrics and
scripts/generate_final_report.py.

Usage:
    python scripts/run_seeds.py --experiment exp_c_shared_adapters --seeds 42,123,2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibrate import calibrate_checkpoint  # noqa: E402
from evaluate import evaluate_checkpoint  # noqa: E402
from train import run_training  # noqa: E402

from src.training.progress import emit, format_multi_seed_preflight  # noqa: E402
from src.utils.config import REPO_ROOT, load_config, load_full_config  # noqa: E402
from src.utils.experiment_paths import experiment_paths  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("scripts.run_seeds")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Experiment name from configs/experiments.yaml")
    parser.add_argument("--seeds", required=True, help="Comma-separated seeds, e.g. 42,123,2026 (>=2 needed for a real std)")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    if len(seeds) < 2:
        logger.warning(
            "Only %d seed(s) requested; mean +/- std across seeds needs at least 2 to be meaningful.", len(seeds)
        )

    experiments_cfg = load_config(REPO_ROOT / "configs" / "experiments.yaml")["experiments"]
    if args.experiment not in experiments_cfg:
        logger.error("Unknown experiment '%s'. See configs/experiments.yaml.", args.experiment)
        return 1
    base_overrides = experiments_cfg[args.experiment].get("overrides", {})

    # This script has no resume/skip-completed logic of its own (unlike a
    # PersistentArtifactManager-backed resumable run) -- every requested
    # seed is always (re)trained from scratch.
    # "already has a checkpoint" here is informational only, not a "will be
    # reused" signal, so it is never reported as "completed (reused)".
    already_has_checkpoint = [
        s for s in seeds
        if (experiment_paths(args.experiment, s)["checkpoint_dir"] / f"{args.experiment}_seed{s}_best_balanced_score.pt").exists()
    ]
    emit(
        format_multi_seed_preflight(
            args.experiment, requested_seeds=seeds, completed_seeds=[],
            incomplete_resumable_seeds=[], missing_seeds=[s for s in seeds if s not in already_has_checkpoint],
            will_run_now_seeds=seeds,
        )
    )
    if already_has_checkpoint:
        emit(
            f"[{args.experiment}] Note: seed(s) {already_has_checkpoint} already have a checkpoint from a "
            "previous run -- this script has no skip/resume logic, so they will be retrained from scratch "
            "and that checkpoint overwritten."
        )

    for seed in seeds:
        run_name = f"{args.experiment}_seed{seed}"
        paths = experiment_paths(args.experiment, seed)
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)

        overrides = {
            **base_overrides,
            "seed": seed,
            "training": {**base_overrides.get("training", {}), "seed": seed},
            "paths": {"checkpoint_dir": str(paths["checkpoint_dir"]), "output_dir": str(paths["base"])},
            "calibration": {"output_dir": str(paths["calibration_dir"])},
        }
        logger.info("=== Training %s (seed=%d) ===", args.experiment, seed)
        config = load_full_config(overrides=overrides)
        try:
            run_training(config, experiment_name=run_name)
        except FileNotFoundError as exc:
            logger.error(str(exc))
            return 1

        checkpoint_path = paths["checkpoint_dir"] / f"{run_name}_best_balanced_score.pt"
        if not checkpoint_path.exists():
            logger.warning("Checkpoint '%s' not found after training; skipping calibration/evaluation.", checkpoint_path)
            continue

        logger.info("=== Calibrating %s (seed=%d) ===", args.experiment, seed)
        calibration_artifact = calibrate_checkpoint(
            str(checkpoint_path), calibration_dir=str(paths["calibration_dir"]),
            experiment_name=run_name, seed=seed,
        )
        if calibration_artifact is None:
            logger.warning(
                "Calibration failed for '%s' (seed=%d); evaluating with raw (uncalibrated) intervals only.",
                args.experiment, seed,
            )

        evaluate_checkpoint(
            str(checkpoint_path), output_name=f"{run_name}_test_metrics",
            calibration_dir=str(paths["calibration_dir"]),
        )

    logger.info(
        "Finished %d seed run(s) for '%s'. Run scripts/generate_final_report.py to aggregate mean +/- std.",
        len(seeds), args.experiment,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
