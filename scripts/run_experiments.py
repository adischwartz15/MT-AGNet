#!/usr/bin/env python
"""CLI: run the full config-driven architecture ablation suite (Experiments 0, A-F).

See configs/experiments.yaml for what each experiment tests. Experiment 0
is a controlled plain-CNN-vs-Custom-ResNet-18 backbone baseline, and
Experiment 0b is the depth/width-matched plain (no-skip) backbone --
compare each against Experiment D. Experiment 0c is the same backbone as
Experiment D with ``zero_init_residual: false`` (see
docs/experiment_plan.md for what each of these three isolates).
Experiment E (parametric vs kNN) does not train a new model -- run
scripts/build_knn_index.py and scripts/evaluate.py --compare-knn against
Experiment D's checkpoint instead. Experiment F (pretrained vs scratch) is
skipped automatically with a clear message if no self-supervised
checkpoint exists yet (run scripts/pretrain.py first).

Each experiment's checkpoint, calibration artifact, and evaluation
metrics/plots are written into an isolated
``experiments/<experiment>/seed_<seed>/{checkpoints,calibration,metrics,plots}``
directory (never a shared global ``checkpoints/`` or ``outputs/calibration``)
-- see src/utils/experiment_paths.py. Every checkpoint is calibrated
(scripts/calibrate.py) before being evaluated with calibration applied;
scripts/evaluate.py independently validates the calibration artifact's
recorded checkpoint/split provenance and fails loudly on any mismatch.

Usage:
    python scripts/run_experiments.py [--only exp_a_separate,exp_c_shared_adapters]
    python scripts/run_experiments.py --only exp_0_simple_cnn_shared_adapters_learned_balance,exp_d_shared_adapters_learned_balance
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

from src.utils.config import REPO_ROOT, load_config, load_full_config  # noqa: E402
from src.utils.experiment_paths import experiment_paths  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger("scripts.run_experiments")

NO_TRAINING_EXPERIMENTS = {"exp_e_parametric_vs_knn"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default=None, help="Comma-separated experiment names to run (default: all)")
    parser.add_argument("--seed", type=int, default=None, help="Default: configs/default.yaml's top-level 'seed'")
    args = parser.parse_args()

    experiments_cfg = load_config(REPO_ROOT / "configs" / "experiments.yaml")["experiments"]
    run_order = load_config(REPO_ROOT / "configs" / "experiments.yaml")["run_order"]
    if args.only:
        run_order = [name for name in run_order if name in set(args.only.split(","))]
    default_seed = load_config(REPO_ROOT / "configs" / "default.yaml")["seed"]

    results = {}
    for name in run_order:
        spec = experiments_cfg[name]
        if name in NO_TRAINING_EXPERIMENTS:
            logger.info("Skipping '%s' (no training step: %s)", name, spec["description"].strip())
            continue

        base_experiment = spec.get("base_experiment")
        if base_experiment:
            logger.info("Experiment '%s' reuses '%s' as its base checkpoint; skipping separate training.", name, base_experiment)
            continue

        overrides = spec.get("overrides", {})
        if name == "exp_f_pretrained_vs_scratch":
            checkpoint_path = REPO_ROOT / overrides.get("model", {}).get("pretrained_checkpoint", "")
            if not checkpoint_path.exists():
                logger.warning(
                    "Skipping '%s': pretrained checkpoint '%s' not found. Run 'make pretrain' first.",
                    name, checkpoint_path,
                )
                continue

        seed = args.seed if args.seed is not None else overrides.get("seed", default_seed)
        paths = experiment_paths(name, seed)
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)

        run_overrides = {
            **overrides,
            "paths": {"checkpoint_dir": str(paths["checkpoint_dir"]), "output_dir": str(paths["base"])},
            "calibration": {"output_dir": str(paths["calibration_dir"])},
        }

        logger.info("=== Running %s ===\n%s", name, spec["description"].strip())
        config = load_full_config(overrides=run_overrides)
        try:
            result = run_training(config, experiment_name=name)
            results[name] = result
        except FileNotFoundError as exc:
            logger.error(str(exc))
            return 1

        # Immediately calibrate and evaluate this experiment's best
        # checkpoint on the test split, so
        # scripts/generate_architecture_report.py's ablation table has real
        # performance numbers (not just parameter counts/timing) per row,
        # and so age intervals are calibrated with an offset that was
        # actually fit for *this* checkpoint (never a shared/global or
        # stale artifact -- see src/evaluation/calibration.py).
        checkpoint_path = paths["checkpoint_dir"] / f"{name}_best_balanced_score.pt"
        if checkpoint_path.exists():
            logger.info("=== Calibrating %s ===", name)
            calibration_artifact = calibrate_checkpoint(
                str(checkpoint_path), calibration_dir=str(paths["calibration_dir"]),
                experiment_name=name, seed=seed,
            )
            if calibration_artifact is None:
                logger.warning(
                    "Calibration failed for '%s'; evaluating with raw (uncalibrated) intervals only.", name
                )

            test_metrics = evaluate_checkpoint(
                str(checkpoint_path), output_name=f"{name}_test_metrics",
                calibration_dir=str(paths["calibration_dir"]),
            )
            if test_metrics is not None:
                results[name]["test_metrics"] = test_metrics
        else:
            logger.warning("Checkpoint '%s' not found after training; skipping its test-set evaluation.", checkpoint_path)

    output_dir = REPO_ROOT / "outputs" / "architecture_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {name: r["parameter_breakdown"] for name, r in results.items()}, output_dir / "parameter_comparison.json"
    )
    logger.info("Finished %d experiment(s).", len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
