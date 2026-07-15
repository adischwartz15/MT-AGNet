"""Isolated per-experiment/per-seed artifact directory layout.

Mirrors the layout the Colab/Kaggle notebooks already use for their
train -> calibrate -> [k-NN] -> evaluate pipeline
(``experiments/<experiment>/seed_<seed>/...``, see the notebooks' "Training
helpers" cell), so ``scripts/run_seeds.py`` / ``scripts/run_experiments.py``
never leave a trained checkpoint's calibration, metrics, plots, robustness,
or k-NN artifacts in a single shared ``outputs/`` directory that a second
experiment or seed could silently overwrite or get calibrated against.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.config import REPO_ROOT


def experiment_run_dir(experiment_name: str, seed: int, root: str | Path | None = None) -> Path:
    """The base directory for one experiment/seed's isolated artifact tree."""
    root = Path(root) if root is not None else REPO_ROOT / "experiments"
    return root / experiment_name / f"seed_{seed}"


def experiment_paths(experiment_name: str, seed: int, root: str | Path | None = None) -> dict[str, Path]:
    """Return the isolated {checkpoints,calibration,metrics,plots,robustness,knn} tree for one run.

    ``base`` doubles as the ``paths.output_dir`` config value, since
    ``scripts/evaluate.py`` / ``scripts/run_robustness.py`` write their
    ``metrics``/``plots``/``robustness`` subdirectories directly under
    whatever ``output_dir`` they're given.
    """
    base = experiment_run_dir(experiment_name, seed, root)
    return {
        "base": base,
        "checkpoint_dir": base / "checkpoints",
        "calibration_dir": base / "calibration",
        "metrics_dir": base / "metrics",
        "plots_dir": base / "plots",
        "robustness_dir": base / "robustness",
        "knn_dir": base / "knn",
    }
