#!/usr/bin/env python
"""CLI: assemble the final, cross-cutting results report from real saved artifacts.

Combines whatever's already on disk under outputs/ (from run_experiments.py,
run_seeds.py, evaluate.py, and run_robustness.py) into one Markdown document:
the architecture ablation table, plain-CNN-vs-ResNet comparison, mean +/- std
across seeds, per-age-bucket uncertainty metrics (raw and calibrated),
robustness degradation, and parameter-count/latency comparison plots. Any
section whose backing artifact is missing renders an explicit "not yet
generated" message rather than a fabricated number -- this script computes no
new metrics itself (except a couple of small comparison plots derived
directly from already-saved numbers).

Usage:
    python scripts/generate_final_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.final_report import save_final_results_report
from src.utils.config import REPO_ROOT
from src.utils.logging import get_logger

logger = get_logger("scripts.generate_final_report")


def main() -> int:
    report_path = save_final_results_report(REPO_ROOT / "outputs", REPO_ROOT / "docs")
    logger.info("Saved final results report to %s", report_path)
    print(f"Saved final results report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
