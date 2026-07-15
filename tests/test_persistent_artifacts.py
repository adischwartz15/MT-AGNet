"""Tests for the platform-agnostic persistence layer (src/training/persistent_artifacts.py).

No Google Drive, no Kaggle, no network -- everything here operates on
``tmp_path`` subdirectories standing in for "local working dir" and
"persistent mirror dir".
"""

from __future__ import annotations

import json

import pytest
import torch

from src.training.persistent_artifacts import (
    CorruptedCheckpointError,
    PersistentArtifactManager,
    build_summary_archive,
    capture_rng_state,
    format_status_line,
    restore_rng_state,
    scan_artifact_root,
    seed_status_report,
)


def _tiny_payload(epoch=1, stage="Stage 1: frozen backbone"):
    return {
        "model_state_dict": {"w": torch.tensor([1.0, 2.0])},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "epoch": epoch,
        "training_stage": stage,
        "config": {"model": {"family": "pretrained_resnet"}},
    }


def test_atomic_checkpoint_write_is_loadable(tmp_path):
    manager = PersistentArtifactManager("exp", seed=42, local_root=tmp_path / "local")
    path = manager.save_last_checkpoint(_tiny_payload())
    assert path.exists()
    assert not path.with_suffix(".pt.tmp").exists()
    loaded = torch.load(path, map_location="cpu")
    assert loaded["epoch"] == 1


def test_checksum_recorded_and_verified(tmp_path):
    manager = PersistentArtifactManager("exp", seed=42, local_root=tmp_path / "local")
    manager.save_last_checkpoint(_tiny_payload())
    checksums = json.loads(manager.checksums_path.read_text())
    assert "last.pt" in checksums
    assert manager._verify_checksum(manager.checkpoints_dir / "last.pt")


def test_last_checkpoint_rotates_to_previous(tmp_path):
    manager = PersistentArtifactManager("exp", seed=42, local_root=tmp_path / "local")
    manager.save_last_checkpoint(_tiny_payload(epoch=1))
    manager.save_last_checkpoint(_tiny_payload(epoch=2))
    assert (manager.checkpoints_dir / "previous_last.pt").exists()
    previous = torch.load(manager.checkpoints_dir / "previous_last.pt", map_location="cpu")
    last = torch.load(manager.checkpoints_dir / "last.pt", map_location="cpu")
    assert previous["epoch"] == 1
    assert last["epoch"] == 2


def test_find_latest_valid_checkpoint_returns_none_when_fresh(tmp_path):
    manager = PersistentArtifactManager("exp", seed=42, local_root=tmp_path / "local")
    assert manager.find_latest_valid_checkpoint() is None


def test_find_latest_valid_checkpoint_falls_back_to_previous_when_last_corrupted(tmp_path):
    manager = PersistentArtifactManager("exp", seed=42, local_root=tmp_path / "local")
    manager.save_last_checkpoint(_tiny_payload(epoch=1))
    manager.save_last_checkpoint(_tiny_payload(epoch=2))
    # Corrupt last.pt in place (truncate it) -- checksum will no longer match.
    (manager.checkpoints_dir / "last.pt").write_bytes(b"not a real checkpoint")

    recovered = manager.find_latest_valid_checkpoint()
    assert recovered is not None
    assert recovered["epoch"] == 1  # previous_last.pt


def test_find_latest_valid_checkpoint_raises_when_both_corrupted(tmp_path):
    manager = PersistentArtifactManager("exp", seed=42, local_root=tmp_path / "local")
    manager.save_last_checkpoint(_tiny_payload(epoch=1))
    manager.save_last_checkpoint(_tiny_payload(epoch=2))
    (manager.checkpoints_dir / "last.pt").write_bytes(b"garbage")
    (manager.checkpoints_dir / "previous_last.pt").write_bytes(b"garbage")

    with pytest.raises(CorruptedCheckpointError):
        manager.find_latest_valid_checkpoint()


def test_sync_and_restore_mirror_without_deleting_persistent_copy(tmp_path):
    local_root = tmp_path / "local"
    persistent_root = tmp_path / "persistent"
    manager = PersistentArtifactManager("exp", seed=42, local_root=local_root, persistent_root=persistent_root)
    manager.save_last_checkpoint(_tiny_payload())
    manager.sync_seed()

    mirrored = persistent_root / "exp" / "seed_42" / "checkpoints" / "last.pt"
    assert mirrored.exists()

    # Simulate a fresh runtime: new local dir, restore from persistent.
    fresh_local_root = tmp_path / "local_fresh"
    fresh_manager = PersistentArtifactManager(
        "exp", seed=42, local_root=fresh_local_root, persistent_root=persistent_root,
    )
    restored = fresh_manager.restore_seed()
    assert restored
    assert (fresh_manager.checkpoints_dir / "last.pt").exists()
    # Persistent copy must still exist after restore.
    assert mirrored.exists()


def test_mark_seed_complete_and_is_seed_complete(tmp_path):
    manager = PersistentArtifactManager("exp", seed=42, local_root=tmp_path / "local")
    checkpoint_path = manager.save_best_checkpoint(_tiny_payload())
    from src.training.persistent_artifacts import sha256_file

    manager.mark_seed_complete({
        "seed": 42, "status": "complete", "best_checkpoint": str(checkpoint_path),
        "test_metrics": {"age_mae": 5.0}, "completed_at": "2026-01-01T00:00:00Z",
        "split_sha256": "abc123", "checkpoint_sha256": sha256_file(checkpoint_path),
        "model_id": "resnet18_224", "pretrained_source": "imagenet1k_v1",
    })

    assert manager.is_seed_complete(expected_split_sha256="abc123", expected_model_id="resnet18_224")
    assert not manager.is_seed_complete(expected_split_sha256="different-hash")
    assert not manager.is_seed_complete(expected_model_id="some_other_model")


def test_is_seed_complete_false_when_only_directory_exists(tmp_path):
    """A seed directory existing (even with files in it) must never, by
    itself, be treated as complete -- only an explicit, validated
    completion.json does."""
    manager = PersistentArtifactManager("exp", seed=42, local_root=tmp_path / "local")
    manager.save_last_checkpoint(_tiny_payload())
    assert not manager.is_seed_complete()


def test_is_seed_complete_false_when_checkpoint_checksum_tampered(tmp_path):
    manager = PersistentArtifactManager("exp", seed=42, local_root=tmp_path / "local")
    from src.training.persistent_artifacts import sha256_file

    checkpoint_path = manager.save_best_checkpoint(_tiny_payload())
    manager.mark_seed_complete({
        "seed": 42, "status": "complete", "best_checkpoint": str(checkpoint_path),
        "test_metrics": {"age_mae": 5.0}, "completed_at": "now",
        "checkpoint_sha256": sha256_file(checkpoint_path),
    })
    assert manager.is_seed_complete()

    checkpoint_path.write_bytes(b"tampered")
    assert not manager.is_seed_complete()


def test_seed_status_report_states(tmp_path):
    local_root = tmp_path / "local"

    not_started = seed_status_report("exp", 2026, local_root, None)
    assert not_started["status"] == "NOT STARTED"

    manager = PersistentArtifactManager("exp", seed=123, local_root=local_root)
    manager.save_last_checkpoint(_tiny_payload(epoch=5, stage="Stage 2: fine-tune"))
    incomplete = seed_status_report("exp", 123, local_root, None)
    assert incomplete["status"] == "INCOMPLETE"
    assert "Stage 2" in incomplete["detail"]
    assert "epoch 5" in incomplete["detail"]
    assert "Seed 123" in format_status_line(incomplete)


def test_summary_archive_excludes_checkpoints_by_default(tmp_path):
    import zipfile

    experiment_root = tmp_path / "exp"
    manager = PersistentArtifactManager("exp", seed=42, local_root=tmp_path)
    manager.save_last_checkpoint(_tiny_payload())
    manager.save_metrics("test_metrics", {"age_mae": 4.0})
    manager.save_run_manifest({"seed": 42})

    archive_path = build_summary_archive(experiment_root, tmp_path / "summary.zip")
    assert archive_path.exists()
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
    assert any(name.endswith("test_metrics.json") for name in names)
    assert any(name.endswith("run_manifest.json") for name in names)
    assert not any(name.endswith(".pt") for name in names)


def test_summary_archive_can_include_best_and_last_but_never_previous_last(tmp_path):
    import zipfile

    experiment_root = tmp_path / "exp"
    manager = PersistentArtifactManager("exp", seed=42, local_root=tmp_path)
    manager.save_last_checkpoint(_tiny_payload(epoch=1))
    manager.save_last_checkpoint(_tiny_payload(epoch=2))  # creates previous_last.pt
    manager.save_best_checkpoint(_tiny_payload(epoch=2))

    archive_path = build_summary_archive(
        experiment_root, tmp_path / "full_summary.zip", include_best_and_last_checkpoints=True,
    )
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
    assert any(name.endswith("best.pt") for name in names)
    assert any(name.endswith("last.pt") and "previous_last" not in name for name in names)
    assert not any("previous_last.pt" in name for name in names)


def test_summary_archive_excludes_predictions_dir_and_credential_shaped_names(tmp_path):
    import zipfile

    experiment_root = tmp_path / "exp"
    (experiment_root / "seed_42" / "predictions").mkdir(parents=True)
    (experiment_root / "seed_42" / "predictions" / "preds.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (experiment_root / "seed_42" / "state").mkdir(parents=True)
    (experiment_root / "seed_42" / "state" / "service_account.json").write_text("{}", encoding="utf-8")
    (experiment_root / "seed_42" / "state" / "run_manifest.json").write_text("{}", encoding="utf-8")

    archive_path = build_summary_archive(experiment_root, tmp_path / "summary.zip")
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
    assert not any("preds.csv" in name for name in names)
    assert not any("service_account" in name for name in names)
    assert any("run_manifest.json" in name for name in names)


def test_summary_archive_includes_extra_files(tmp_path):
    import zipfile

    extra = tmp_path / "table_b.csv"
    extra.write_text("model,age_mae\n", encoding="utf-8")
    archive_path = build_summary_archive(tmp_path / "exp", tmp_path / "summary.zip", extra_files=[extra])
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
    assert "table_b.csv" in names


def test_rng_state_roundtrip_reproduces_next_draw(tmp_path):
    import random

    random.seed(0)
    state = capture_rng_state()
    first_draw = random.random()

    random.seed(999)  # perturb
    restore_rng_state(state)
    second_draw = random.random()
    assert first_draw == second_draw


# -- scan_artifact_root ---------------------------------------------------------------


def test_scan_artifact_root_empty_root_returns_no_rows(tmp_path):
    assert scan_artifact_root(tmp_path / "does_not_exist") == []


def test_scan_artifact_root_reports_not_started_when_no_state_files(tmp_path):
    manager = PersistentArtifactManager("exp_a", seed=42, local_root=tmp_path)
    rows = scan_artifact_root(tmp_path)
    assert len(rows) == 1
    assert rows[0]["experiment"] == "exp_a"
    assert rows[0]["seed"] == 42
    assert rows[0]["status"] == "NOT STARTED"
    assert rows[0]["checkpoint"] is None
    assert manager  # manager's mkdir side effect is what created the seed_42 dir scan_artifact_root finds


def test_scan_artifact_root_reports_incomplete_with_stage_epoch_and_checkpoint(tmp_path):
    manager = PersistentArtifactManager("exp_b", seed=123, local_root=tmp_path)
    manager.save_last_checkpoint(_tiny_payload(epoch=5, stage="Stage 2: fine-tune"))
    manager.save_training_state(
        {"training_stage": "Stage 2: fine-tune", "global_epoch": 5, "best_validation_metric": 0.42}
    )
    rows = scan_artifact_root(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "INCOMPLETE"
    assert row["stage"] == "Stage 2: fine-tune"
    assert row["epoch"] == 5
    assert row["best_score"] == 0.42
    assert row["checkpoint"].endswith("last.pt")
    assert row["last_update"] is not None


def test_scan_artifact_root_reports_complete_and_prefers_best_checkpoint(tmp_path):
    manager = PersistentArtifactManager("exp_c", seed=42, local_root=tmp_path)
    manager.save_last_checkpoint(_tiny_payload())
    manager.save_best_checkpoint(_tiny_payload())
    manager.mark_seed_complete(
        {"seed": 42, "status": "complete", "best_checkpoint": str(manager.checkpoints_dir / "best.pt"),
         "test_metrics": {"balanced_score": 0.9}, "completed_at": "2026-01-01T00:00:00Z"}
    )
    rows = scan_artifact_root(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "COMPLETE"
    assert row["best_score"] == 0.9
    assert row["checkpoint"].endswith("best.pt")


def test_scan_artifact_root_covers_multiple_experiments_and_seeds(tmp_path):
    PersistentArtifactManager("exp_a", seed=42, local_root=tmp_path)
    PersistentArtifactManager("exp_a", seed=123, local_root=tmp_path)
    PersistentArtifactManager("exp_b", seed=42, local_root=tmp_path)
    rows = scan_artifact_root(tmp_path)
    pairs = {(r["experiment"], r["seed"]) for r in rows}
    assert pairs == {("exp_a", 42), ("exp_a", 123), ("exp_b", 42)}


def test_scan_artifact_root_is_read_only(tmp_path):
    manager = PersistentArtifactManager("exp_a", seed=42, local_root=tmp_path)
    manager.save_last_checkpoint(_tiny_payload())
    before = {p: p.stat().st_mtime for p in tmp_path.rglob("*") if p.is_file()}
    scan_artifact_root(tmp_path)
    after = {p: p.stat().st_mtime for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
