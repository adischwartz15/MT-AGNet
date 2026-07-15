"""Tests for scripts/run_seeds.py's multi-seed preflight summary --
printed before any training starts, so a Colab/Kaggle cell's output makes
clear from the top which of the requested seeds already have a checkpoint
(informational only: this script has no skip/resume logic, so every
requested seed is always retrained)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = str(REPO_ROOT / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


def test_main_prints_preflight_before_any_training(monkeypatch, tmp_path, capsys):
    import run_seeds as rs

    monkeypatch.setattr(
        rs, "load_config",
        lambda *_a, **_k: {"experiments": {"exp_x": {"overrides": {}}}},
    )

    def _fake_experiment_paths(experiment, seed):
        base = tmp_path / experiment / f"seed_{seed}"
        return {
            "base": base, "checkpoint_dir": base / "checkpoints", "calibration_dir": base / "calibration",
        }

    monkeypatch.setattr(rs, "experiment_paths", _fake_experiment_paths)

    def _fail_if_called(*_a, **_k):
        raise AssertionError("run_training must not be called by this test")

    monkeypatch.setattr(rs, "run_training", _fail_if_called)

    monkeypatch.setattr(sys, "argv", ["run_seeds.py", "--experiment", "exp_x", "--seeds", "42,123"])
    # run_training raising isn't reached: FileNotFoundError is the only
    # caught exception in main()'s loop, so a bare AssertionError from our
    # monkeypatch propagates and fails the test loudly if reached.
    try:
        rs.main()
    except AssertionError:
        pass

    captured = capsys.readouterr()
    assert "Multi-seed run plan:" in captured.out
    assert "requested seeds:            [42, 123]" in captured.out
    assert "will run now:               [42, 123]" in captured.out


def test_preflight_notes_seeds_with_existing_checkpoint_as_informational_only(monkeypatch, tmp_path, capsys):
    import run_seeds as rs

    monkeypatch.setattr(rs, "load_config", lambda *_a, **_k: {"experiments": {"exp_x": {"overrides": {}}}})

    def _fake_experiment_paths(experiment, seed):
        base = tmp_path / experiment / f"seed_{seed}"
        return {
            "base": base, "checkpoint_dir": base / "checkpoints", "calibration_dir": base / "calibration",
        }

    monkeypatch.setattr(rs, "experiment_paths", _fake_experiment_paths)
    existing_ckpt_dir = tmp_path / "exp_x" / "seed_42" / "checkpoints"
    existing_ckpt_dir.mkdir(parents=True)
    (existing_ckpt_dir / "exp_x_seed42_best_balanced_score.pt").write_bytes(b"fake")

    monkeypatch.setattr(rs, "run_training", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("stop after preflight")))
    monkeypatch.setattr(sys, "argv", ["run_seeds.py", "--experiment", "exp_x", "--seeds", "42,123"])
    try:
        rs.main()
    except AssertionError:
        pass

    captured = capsys.readouterr()
    assert "missing (will start fresh): [123]" in captured.out
    assert "already have a checkpoint from a" in captured.out
    assert "will be retrained from scratch" in captured.out
