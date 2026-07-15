#!/usr/bin/env python
"""CLI: (re)generate docs/architecture_analysis_generated.md from current outputs/ artifacts.

This is a thin convenience wrapper around src.evaluation.reports -- useful
after re-running only some pipeline stages (e.g. just 'make robustness')
when you want the report refreshed without recomputing gradient
interference / representation similarity via
scripts/generate_architecture_report.py.

Usage:
    python scripts/export_report.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.reports import save_report
from src.utils.config import REPO_ROOT
from src.utils.logging import get_logger

logger = get_logger("scripts.export_report")


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    report_path = save_report(REPO_ROOT / "outputs", REPO_ROOT / "docs")
    logger.info("Report written to %s", report_path)
    print(f"Report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
