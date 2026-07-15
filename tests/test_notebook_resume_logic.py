"""Regression test for the notebooks' stage-level resume/restart-safety logic.

Both notebooks (notebooks/train_evaluate_{colab,kaggle}.ipynb) share an
identical "Training helpers" code cell defining experiment_paths,
train_one_experiment, calibrate_one_experiment, build_knn_one_experiment,
evaluate_one_experiment, and run_experiment_pipeline. Since notebook cells
aren't natively importable/unit-testable, this test extracts that cell's
*actual* source directly from the shipped .ipynb (not a reimplementation
that could silently drift from what's really there) and executes it in a
controlled namespace with a mocked run_command, to verify: with
FORCE_RERUN=False, a stage whose artifact already exists is skipped
(run_command not called for it), and a later stage whose artifact is
missing still runs -- i.e. an evaluation failure never causes training to
be redone once a checkpoint exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def _extract_training_helpers_source(notebook_path: Path) -> str:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "def run_experiment_pipeline" in source:
            return source
    raise AssertionError(f"No cell defining run_experiment_pipeline found in {notebook_path}")


def _extract_helper_library_source(notebook_path: Path) -> str:
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "def run_command" in source:
            return source
    raise AssertionError(f"No cell defining run_command found in {notebook_path}")


@pytest.fixture(params=["train_evaluate_colab.ipynb", "train_evaluate_kaggle.ipynb"])
def training_helpers_namespace(request, tmp_path):
    """Exec the real notebook cell in a namespace with mocked I/O, returning
    (namespace, run_command_calls, run_dir) for assertions."""
    source = _extract_training_helpers_source(NOTEBOOKS_DIR / request.param)

    run_dir = tmp_path / "run"
    calls = []

    def fake_run_command(cmd, cwd=None, log_path=None, check=True, env=None):
        import torch

        calls.append(cmd)
        # Simulate scripts/train.py: creates the checkpoint the caller expects.
        # A real torch.save is needed since train_one_experiment loads it
        # back (torch.load(...)["config"]) to extract the resolved config.
        # Checks the whole command (not a fixed index) since real commands
        # now insert "-u" (unbuffered output) between sys.executable and the
        # script path.
        cmd_str = " ".join(str(c) for c in cmd)
        if "train.py" in cmd_str:
            experiment_name = cmd[cmd.index("--experiment-name") + 1]
            checkpoint_dir = run_dir / "experiments" / experiment_name / "seed_42" / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"config": {}}, checkpoint_dir / f"{experiment_name}_best_balanced_score.pt")
        if "calibrate.py" in " ".join(str(c) for c in cmd):
            experiment_name = "fake_exp"
            calibration_dir = run_dir / "experiments" / experiment_name / "seed_42" / "calibration"
            calibration_dir.mkdir(parents=True, exist_ok=True)
            (calibration_dir / "conformal_calibration.json").write_text("{}", encoding="utf-8")
        if "evaluate.py" in " ".join(str(c) for c in cmd):
            experiment_name = "fake_exp"
            metrics_dir = run_dir / "experiments" / experiment_name / "seed_42" / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            (metrics_dir / f"{experiment_name}_test_metrics.json").write_text('{"age_mae": 5.0}', encoding="utf-8")
        return 0, ""

    def fake_write_manifest(path, data):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, default=str), encoding="utf-8")
        return Path(path)

    def fake_load_json(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def fake_validate_required_artifacts(paths):
        missing = [str(p) for p in paths if not Path(p).exists()]
        if missing:
            raise RuntimeError(f"missing: {missing}")
        return True

    def fake_flatten_overrides(obj, prefix=""):
        return []

    namespace = {
        "RUN_DIR": run_dir,
        "REPO_DIR": tmp_path,
        "FORCE_RERUN": False,
        "MAX_EPOCHS": 1,
        "EARLY_STOPPING_PATIENCE": 1,
        "MAX_BATCHES_PER_EPOCH": None,
        "DIFFERENTIAL_LR_ENABLED": True,
        "BACKBONE_LR_MULTIPLIER": 0.1,
        "ADAPTER_BOTTLENECK_DIM": 256,
        "LOSS_BALANCING_WARMUP_EPOCHS": 3,
        "experiments_cfg": {"fake_exp": {"overrides": {}}},
        "run_command": fake_run_command,
        "write_manifest": fake_write_manifest,
        "load_json": fake_load_json,
        "validate_required_artifacts": fake_validate_required_artifacts,
        "flatten_overrides": fake_flatten_overrides,
        "sys": __import__("sys"),
        "print": lambda *a, **k: None,  # silence the stage-plan/status prints
    }
    exec(compile(source, str(NOTEBOOKS_DIR / request.param), "exec"), namespace)
    return namespace, calls, run_dir


def test_first_run_executes_every_stage(training_helpers_namespace):
    namespace, calls, run_dir = training_helpers_namespace
    paths, metrics = namespace["run_experiment_pipeline"]("fake_exp", 42, include_knn=False)

    assert metrics == {"age_mae": 5.0}
    joined_calls = [" ".join(str(c) for c in call) for call in calls]
    assert any("train.py" in c for c in joined_calls)
    assert any("calibrate.py" in c for c in joined_calls)
    assert any("evaluate.py" in c for c in joined_calls)


def test_resume_skips_training_when_checkpoint_exists_but_reruns_missing_evaluation(training_helpers_namespace):
    """The core restart-safety requirement: pre-create a checkpoint (as if
    training previously succeeded) but no calibration/metrics (as if a
    later stage previously failed) -- re-running the pipeline must skip
    training (no train.py call) while still running calibration and
    evaluation."""
    namespace, calls, run_dir = training_helpers_namespace
    checkpoint_dir = run_dir / "experiments" / "fake_exp" / "seed_42" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "fake_exp_best_balanced_score.pt").write_bytes(b"already-trained")

    paths, metrics = namespace["run_experiment_pipeline"]("fake_exp", 42, include_knn=False)

    joined_calls = [" ".join(str(c) for c in call) for call in calls]
    assert not any("train.py" in c for c in joined_calls), "training must be skipped when checkpoint already exists"
    assert any("calibrate.py" in c for c in joined_calls), "calibration must still run since its artifact was missing"
    assert any("evaluate.py" in c for c in joined_calls), "evaluation must still run since its artifact was missing"
    assert metrics == {"age_mae": 5.0}


def test_resume_skips_every_stage_when_all_artifacts_already_exist(training_helpers_namespace):
    namespace, calls, run_dir = training_helpers_namespace
    base = run_dir / "experiments" / "fake_exp" / "seed_42"
    (base / "checkpoints").mkdir(parents=True, exist_ok=True)
    (base / "checkpoints" / "fake_exp_best_balanced_score.pt").write_bytes(b"done")
    (base / "calibration").mkdir(parents=True, exist_ok=True)
    (base / "calibration" / "conformal_calibration.json").write_text("{}", encoding="utf-8")
    (base / "metrics").mkdir(parents=True, exist_ok=True)
    (base / "metrics" / "fake_exp_test_metrics.json").write_text('{"age_mae": 3.3}', encoding="utf-8")

    paths, metrics = namespace["run_experiment_pipeline"]("fake_exp", 42, include_knn=False)

    assert calls == [], "no stage should re-run when every artifact already exists"
    assert metrics == {"age_mae": 3.3}


@pytest.fixture(params=["train_evaluate_colab.ipynb", "train_evaluate_kaggle.ipynb"])
def helper_library_namespace(request, monkeypatch):
    """Exec the real notebook "Helper library" cell (defines run_command,
    copy_tree_merge, safe_copy2, ...) in a bare namespace.

    The cell imports ``IPython.display`` (only available inside an actual
    Jupyter/Colab/Kaggle runtime, not a project dependency here), so a
    minimal stand-in module is injected into ``sys.modules`` purely to
    satisfy that import -- nothing under test touches it.
    """
    import sys
    import types

    fake_display = types.ModuleType("IPython.display")
    fake_display.Image = object
    fake_display.Markdown = object
    fake_display.display = lambda *a, **k: None
    fake_ipython = types.ModuleType("IPython")
    fake_ipython.display = fake_display
    monkeypatch.setitem(sys.modules, "IPython", fake_ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", fake_display)

    source = _extract_helper_library_source(NOTEBOOKS_DIR / request.param)
    namespace = {"__name__": "helper_library_under_test"}
    exec(compile(source, str(NOTEBOOKS_DIR / request.param), "exec"), namespace)
    return namespace


def test_safe_copy2_skips_instead_of_raising_when_source_and_destination_are_identical(
    helper_library_namespace, tmp_path,
):
    """Regression test for a Colab resume crash:

        SameFileError: RUN_DIR/logs/npm_build.log and RUN_DIR/logs/npm_build.log
        are the same file.

    This happens whenever RUN_DIR already points at the persistent Drive run
    directory (e.g. a resumed run) and a later "sync to persistent storage"
    step tries to copy a file onto itself. safe_copy2 must detect that via
    resolved-path equality and skip instead of raising.
    """
    safe_copy2 = helper_library_namespace["safe_copy2"]
    same_file = tmp_path / "logs" / "npm_build.log"
    same_file.parent.mkdir(parents=True, exist_ok=True)
    same_file.write_text("build output", encoding="utf-8")

    result = safe_copy2(same_file, same_file)

    assert Path(result).resolve() == same_file.resolve()
    assert same_file.read_text(encoding="utf-8") == "build output"


def test_safe_copy2_still_copies_when_source_and_destination_differ(helper_library_namespace, tmp_path):
    safe_copy2 = helper_library_namespace["safe_copy2"]
    src = tmp_path / "src.log"
    src.write_text("hello", encoding="utf-8")
    dst = tmp_path / "dst.log"

    safe_copy2(src, dst)

    assert dst.read_text(encoding="utf-8") == "hello"


def test_copy_tree_merge_does_not_raise_when_src_and_dst_are_the_same_directory(
    helper_library_namespace, tmp_path,
):
    """End-to-end version of the reported crash: mirroring RUN_DIR onto
    itself (a resumed run whose RUN_DIR already *is* the persistent Drive
    directory) must not raise SameFileError for any file underneath it."""
    copy_tree_merge = helper_library_namespace["copy_tree_merge"]
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "logs" / "npm_build.log").write_text("build output", encoding="utf-8")

    copied = copy_tree_merge(run_dir, run_dir)

    assert (run_dir / "logs" / "npm_build.log").read_text(encoding="utf-8") == "build output"
    assert len(copied) == 1
