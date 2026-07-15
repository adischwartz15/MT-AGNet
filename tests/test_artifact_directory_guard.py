"""Proves the conftest.py `_guard_real_artifact_directories` autouse fixture
actually catches a test that writes into a real, repo-level artifact
directory (data/splits, checkpoints, outputs, results, logs) instead of an
isolated tmp_path -- not just that the existing suite happens to pass under
it. Regression test for a real incident: an earlier draft of
tests/test_lock_split.py didn't override paths.splits_dir via --set (the
only override tier that beats this repo's own .env DATA_DIR setting), so
scripts/lock_split.py silently wrote synthetic pytest-fixture data into the
real data/splits/ directory. The guard now fails any test that does this,
loudly, instead of leaving silent residue.

Runs a deliberately "bad" test file as a real pytest subprocess (with this
repo's real tests/conftest.py in effect) and asserts it is caught -- this
cannot be proven by importing the fixture directly, since the guarantee is
specifically about pytest's own autouse-fixture wiring.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_BAD_TEST_BODY = textwrap.dedent(
    """
    from src.utils.config import REPO_ROOT

    def test_writes_into_real_data_splits_directory():
        target_dir = REPO_ROOT / "data" / "splits"
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "should_never_land_here.csv").write_text("bad", encoding="utf-8")
    """
)


def test_guard_fails_a_test_that_writes_into_real_data_splits_directory(tmp_path):
    bad_test_path = REPO_ROOT / "tests" / "_tmp_guard_negative_control_test.py"
    real_target = REPO_ROOT / "data" / "splits" / "should_never_land_here.csv"
    assert not real_target.exists(), "negative-control target must not pre-exist"

    bad_test_path.write_text(_BAD_TEST_BODY, encoding="utf-8")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(bad_test_path), "-q"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
        combined = result.stdout + result.stderr
        assert result.returncode != 0, (
            "the guard fixture failed to catch a test that wrote into the real "
            f"data/splits/ directory:\n{combined}"
        )
        assert "real, repo-level artifact directory" in combined
    finally:
        bad_test_path.unlink(missing_ok=True)
        # Belt-and-suspenders cleanup: the guard fixture is expected to have
        # already detected the write (the assertion above proves that), but
        # this test must never itself leave residue in the real directory
        # regardless of the subprocess's outcome.
        real_target.unlink(missing_ok=True)
