"""Reusable, platform-agnostic persistence layer for long-running, resumable
training runs.

Why this exists: a Colab/Kaggle runtime can disconnect at any point during a
multi-hour, multi-seed run. Without a persistence layer, checkpoints and
metrics written only to the ephemeral runtime filesystem are lost, forcing a
full retrain. This module centralizes atomic checkpoint writes, checksums,
seed-completion tracking, and local<->persistent-storage synchronization
behind one class, so that:

- A trainer only ever calls a handful of ``on_*`` hook methods -- it has no
  idea whether "persistent" means a mounted Google Drive folder,
  ``/kaggle/working``, or a plain local directory (a unit test uses two temp
  directories).
- No ``shutil.copytree``/Drive-mount/Kaggle-specific code is scattered
  through model or trainer code.

Torch is a hard dependency of this module (checkpoints are ``torch.save``
payloads) -- that's fine, since torch is a base requirement of the whole
repository.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

CHECKPOINT_NAMES = ("last.pt", "previous_last.pt", "best.pt")


class CorruptedCheckpointError(RuntimeError):
    """Raised when every candidate checkpoint file exists but fails validation.

    Deliberately distinct from "no checkpoint exists yet" (which is a normal
    fresh-start condition, not an error) -- see
    :meth:`PersistentArtifactManager.find_latest_valid_checkpoint`.
    """


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def _atomic_write_json(path: Path, data: Any) -> None:
    _atomic_write_bytes(path, json.dumps(data, indent=2, default=str).encode("utf-8"))


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    import hashlib

    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    """Snapshot Python/NumPy/PyTorch (CPU + CUDA) RNG state for exact resume."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(tuple(state["python"]) if not isinstance(state["python"], tuple) else state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch_state = state["torch_cpu"]
        if not torch.is_tensor(torch_state):
            torch_state = torch.ByteTensor(torch_state)
        torch.set_rng_state(torch_state.type(torch.ByteTensor))
    if "torch_cuda" in state and torch.cuda.is_available():
        cuda_states = [s if torch.is_tensor(s) else torch.ByteTensor(s) for s in state["torch_cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


@dataclass
class SeedCompletionInfo:
    seed: int
    status: str
    best_checkpoint: str
    test_metrics: dict
    completed_at: str
    split_sha256: str | None = None
    git_commit_sha: str | None = None
    checkpoint_sha256: str | None = None
    model_id: str | None = None
    pretrained_source: str | None = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "status": self.status,
            "best_checkpoint": self.best_checkpoint,
            "test_metrics": self.test_metrics,
            "completed_at": self.completed_at,
            "split_sha256": self.split_sha256,
            "git_commit_sha": self.git_commit_sha,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_id": self.model_id,
            "pretrained_source": self.pretrained_source,
            **self.extra,
        }


class PersistentArtifactManager:
    """Owns one experiment/seed's isolated, resumable artifact tree.

    Directory layout under ``local_root`` (and mirrored, file-for-file,
    under ``persistent_root`` when one is given)::

        seed_<seed>/
        |-- checkpoints/{last.pt, previous_last.pt, best.pt}
        |-- state/{trainer_state.json, run_manifest.json, completion.json,
        |          checkpoint_checksums.json}
        |-- metrics/{validation_history.json, test_metrics.json, subgroup_metrics.json}
        |-- predictions/
        |-- plots/
        `-- logs/

    ``local_root``/``persistent_root`` are both plain filesystem paths --
    that's what makes this class platform-agnostic. A Colab caller passes
    ``persistent_root=Path("/content/drive/MyDrive/AgeGender/transfer_learning")``
    (a mounted Drive folder looks like any other directory once mounted); a
    Kaggle caller passes ``persistent_root=Path("/kaggle/working/...")`` (or
    ``None`` and relies on the optional Drive-API backup helper instead --
    see ``src/utils/kaggle_drive_backup.py``); a unit test passes two
    ``tmp_path`` subdirectories. Nothing in this class imports
    ``google.colab`` or the Kaggle API.
    """

    def __init__(
        self,
        experiment_name: str,
        seed: int,
        local_root: str | Path,
        persistent_root: str | Path | None = None,
        sync_after_epoch: bool = True,
    ) -> None:
        self.experiment_name = experiment_name
        self.seed = seed
        self.sync_after_epoch = sync_after_epoch

        self.local_seed_dir = Path(local_root) / experiment_name / f"seed_{seed}"
        self.persistent_seed_dir = (
            Path(persistent_root) / experiment_name / f"seed_{seed}" if persistent_root is not None else None
        )
        for sub in ("checkpoints", "state", "metrics", "predictions", "plots", "logs"):
            (self.local_seed_dir / sub).mkdir(parents=True, exist_ok=True)
        # Set by find_latest_valid_checkpoint() to whichever file (last.pt or
        # previous_last.pt) actually resolved -- lets a caller report a
        # precise resume announcement (path + checksum) without duplicating
        # that method's fallback resolution logic.
        self.last_resolved_checkpoint_path: Path | None = None

    # -- path helpers ---------------------------------------------------------------

    @property
    def checkpoints_dir(self) -> Path:
        return self.local_seed_dir / "checkpoints"

    @property
    def state_dir(self) -> Path:
        return self.local_seed_dir / "state"

    @property
    def metrics_dir(self) -> Path:
        return self.local_seed_dir / "metrics"

    @property
    def predictions_dir(self) -> Path:
        return self.local_seed_dir / "predictions"

    @property
    def plots_dir(self) -> Path:
        return self.local_seed_dir / "plots"

    @property
    def logs_dir(self) -> Path:
        return self.local_seed_dir / "logs"

    @property
    def checksums_path(self) -> Path:
        return self.state_dir / "checkpoint_checksums.json"

    @property
    def completion_path(self) -> Path:
        return self.state_dir / "completion.json"

    # -- checksums --------------------------------------------------------------

    def _load_checksums(self) -> dict:
        if self.checksums_path.exists():
            with open(self.checksums_path, encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _record_checksum(self, filename: str, digest: str) -> None:
        checksums = self._load_checksums()
        checksums[filename] = digest
        _atomic_write_json(self.checksums_path, checksums)

    def _verify_checksum(self, path: Path) -> bool:
        checksums = self._load_checksums()
        expected = checksums.get(path.name)
        if expected is None:
            # No recorded checksum (e.g. checkpoint written before checksum
            # tracking was enabled) -- not proof of corruption by itself,
            # callers still attempt torch.load() to validate the file.
            return True
        try:
            return sha256_file(path) == expected
        except OSError:
            return False

    # -- checkpoint writes --------------------------------------------------------

    def _save_named_checkpoint(self, filename: str, payload: dict) -> Path:
        path = self.checkpoints_dir / filename
        _atomic_torch_save(path, payload)
        self._record_checksum(filename, sha256_file(path))
        return path

    def save_last_checkpoint(self, payload: dict) -> Path:
        """Atomically write ``last.pt``, first rotating any existing valid
        ``last.pt`` into ``previous_last.pt`` so a crash mid-write never
        leaves the seed without *some* resumable checkpoint."""
        last_path = self.checkpoints_dir / "last.pt"
        if last_path.exists():
            previous_path = self.checkpoints_dir / "previous_last.pt"
            os.replace(last_path, previous_path)
            checksums = self._load_checksums()
            if "last.pt" in checksums:
                checksums["previous_last.pt"] = checksums["last.pt"]
                _atomic_write_json(self.checksums_path, checksums)
        return self._save_named_checkpoint("last.pt", payload)

    def save_best_checkpoint(self, payload: dict) -> Path:
        return self._save_named_checkpoint("best.pt", payload)

    def save_checkpoint(self, name: str, payload: dict) -> Path:
        """Generic named checkpoint (for callers needing an ad hoc snapshot)."""
        filename = name if name.endswith(".pt") else f"{name}.pt"
        return self._save_named_checkpoint(filename, payload)

    # -- checkpoint reads/validation ------------------------------------------------

    def _try_load_checkpoint(self, filename: str) -> dict | None:
        path = self.checkpoints_dir / filename
        if not path.exists():
            return None
        if not self._verify_checksum(path):
            logger.warning("Checksum mismatch for %s -- treating as corrupted.", path)
            return None
        try:
            # weights_only=False: these checkpoints intentionally carry
            # non-tensor Python state (NumPy RNG state, plain config dicts,
            # scheduler/early-stopping state) -- only ever checkpoints this
            # same codebase wrote (see module docstring), never an
            # untrusted download.
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            logger.warning("Failed to load %s -- treating as corrupted.", path, exc_info=True)
            return None

    def find_latest_valid_checkpoint(self) -> dict | None:
        """Load ``last.pt``, validating checksum + torch.load; on any
        corruption, fall back to ``previous_last.pt`` (with a loud warning,
        never silently). Returns ``None`` only if neither file exists (a
        genuine fresh start). Raises :class:`CorruptedCheckpointError` if at
        least one checkpoint file exists but none loads validly -- that is
        data loss, not a fresh start, and must never be silently treated as one.
        """
        self.last_resolved_checkpoint_path = None
        last_exists = (self.checkpoints_dir / "last.pt").exists()
        previous_exists = (self.checkpoints_dir / "previous_last.pt").exists()
        if not last_exists and not previous_exists:
            return None

        payload = self._try_load_checkpoint("last.pt")
        if payload is not None:
            self.last_resolved_checkpoint_path = self.checkpoints_dir / "last.pt"
            return payload

        if last_exists:
            logger.warning(
                "seed=%d: last.pt exists but is corrupted/invalid -- falling back to previous_last.pt.",
                self.seed,
            )
        payload = self._try_load_checkpoint("previous_last.pt")
        if payload is not None:
            self.last_resolved_checkpoint_path = self.checkpoints_dir / "previous_last.pt"
            return payload

        raise CorruptedCheckpointError(
            f"seed={self.seed}: both last.pt and previous_last.pt exist but neither loaded "
            "validly (checksum mismatch or torch.load failure). Refusing to silently restart "
            "from scratch -- inspect "
            f"{self.checkpoints_dir} manually before proceeding."
        )

    def last_resolved_checkpoint_sha256(self) -> str | None:
        """SHA-256 of whichever file :meth:`find_latest_valid_checkpoint`
        most recently resolved to (``None`` if it hasn't been called, found
        nothing, or the file has since disappeared) -- powers a resume
        announcement's "checkpoint sha256" field without re-deriving which
        of last.pt/previous_last.pt was actually used."""
        if self.last_resolved_checkpoint_path is None or not self.last_resolved_checkpoint_path.exists():
            return None
        return sha256_file(self.last_resolved_checkpoint_path)

    def load_best_checkpoint(self) -> dict | None:
        return self._try_load_checkpoint("best.pt")

    # -- state / history / metrics --------------------------------------------------

    def save_training_state(self, state: dict) -> Path:
        path = self.state_dir / "trainer_state.json"
        _atomic_write_json(path, state)
        return path

    def save_run_manifest(self, manifest: dict) -> Path:
        path = self.state_dir / "run_manifest.json"
        _atomic_write_json(path, manifest)
        return path

    def save_history(self, history: dict) -> Path:
        path = self.metrics_dir / "validation_history.json"
        _atomic_write_json(path, history)
        return path

    def save_metrics(self, name: str, metrics: dict) -> Path:
        filename = name if name.endswith(".json") else f"{name}.json"
        path = self.metrics_dir / filename
        _atomic_write_json(path, metrics)
        return path

    # -- seed completion --------------------------------------------------------

    def mark_seed_complete(self, info: SeedCompletionInfo | dict) -> Path:
        data = info.as_dict() if isinstance(info, SeedCompletionInfo) else info
        path = self.completion_path
        _atomic_write_json(path, data)
        return path

    def load_completion(self) -> dict | None:
        if not self.completion_path.exists():
            return None
        with open(self.completion_path, encoding="utf-8") as fh:
            return json.load(fh)

    def is_seed_complete(
        self,
        expected_split_sha256: str | None = None,
        expected_model_id: str | None = None,
        expected_pretrained_source: str | None = None,
    ) -> bool:
        """A seed is complete only if every one of these independently holds
        (never inferred from directory existence alone):

        1. ``completion.json`` exists and ``status == "complete"``.
        2. The best checkpoint it references exists on disk.
        3. That checkpoint's checksum matches the recorded one.
        4. Test metrics are present (non-empty).
        5. The split fingerprint matches the *current* split (if provided).
        6. The model identifier / pretrained source match the *current*
           config (if provided) -- prevents reusing a seed trained under a
           different backbone/config as if it were compatible.
        """
        completion = self.load_completion()
        if completion is None or completion.get("status") != "complete":
            return False

        best_checkpoint = completion.get("best_checkpoint")
        if not best_checkpoint or not Path(best_checkpoint).exists():
            return False

        recorded_checksum = completion.get("checkpoint_sha256")
        if recorded_checksum:
            try:
                if sha256_file(best_checkpoint) != recorded_checksum:
                    logger.warning("seed=%d: best checkpoint checksum mismatch -- not complete.", self.seed)
                    return False
            except OSError:
                return False

        if not completion.get("test_metrics"):
            return False

        if expected_split_sha256 is not None and completion.get("split_sha256") not in (None, expected_split_sha256):
            logger.warning(
                "seed=%d: split fingerprint mismatch (recorded=%s, current=%s) -- not complete.",
                self.seed, completion.get("split_sha256"), expected_split_sha256,
            )
            return False

        if expected_model_id is not None and completion.get("model_id") not in (None, expected_model_id):
            logger.warning("seed=%d: model identifier mismatch -- not complete.", self.seed)
            return False

        if expected_pretrained_source is not None and completion.get("pretrained_source") not in (
            None, expected_pretrained_source,
        ):
            logger.warning("seed=%d: pretrained source mismatch -- not complete.", self.seed)
            return False

        return True

    # -- synchronization (local <-> persistent mirror) -------------------------------

    def _copy_tree_merge(self, src: Path, dst: Path) -> list[Path]:
        if not src.exists():
            return []
        dst.mkdir(parents=True, exist_ok=True)
        copied = []
        for path in src.rglob("*"):
            if not path.is_file():
                continue
            target = dst / path.relative_to(src)
            if path.resolve() == target.resolve():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp_target = target.with_suffix(target.suffix + ".tmp")
            shutil.copy2(path, tmp_target)
            os.replace(tmp_target, target)
            copied.append(target)
        return copied

    def sync_seed(self) -> list[Path]:
        """Mirror the local seed directory to the persistent root (if configured).

        A failure here (Drive unmounted, disk full, network hiccup) must
        never destroy or corrupt the local copy -- ``_copy_tree_merge``
        only ever reads from local and writes new/temp files on the
        persistent side, so raising mid-copy leaves both sides in a valid
        (if incompletely synced) state.
        """
        if self.persistent_seed_dir is None:
            return []
        try:
            copied = self._copy_tree_merge(self.local_seed_dir, self.persistent_seed_dir)
            logger.info("seed=%d: synced %d file(s) to %s", self.seed, len(copied), self.persistent_seed_dir)
            return copied
        except OSError as exc:
            logger.warning(
                "seed=%d: sync to persistent storage failed (%s) -- local checkpoints remain intact at %s.",
                self.seed, exc, self.local_seed_dir,
            )
            return []

    def restore_seed(self) -> list[Path]:
        """Mirror the persistent seed directory back into the local working
        directory (e.g. after a fresh runtime start). Never deletes the
        persistent copy. No-op if no persistent root is configured or
        nothing has been synced there yet."""
        if self.persistent_seed_dir is None or not self.persistent_seed_dir.exists():
            return []
        restored = self._copy_tree_merge(self.persistent_seed_dir, self.local_seed_dir)
        logger.info("seed=%d: restored %d file(s) from %s", self.seed, len(restored), self.persistent_seed_dir)
        return restored

    # -- hooks (called by a resumable trainer / CLI) -------------------------------------

    def on_epoch_end(self, checkpoint_payload: dict, history: dict, trainer_state: dict, is_best: bool) -> None:
        self.save_last_checkpoint(checkpoint_payload)
        self.save_history(history)
        self.save_training_state(trainer_state)
        if is_best:
            self.save_best_checkpoint(checkpoint_payload)
        if self.sync_after_epoch:
            self.sync_seed()

    def on_new_best(self, checkpoint_payload: dict) -> None:
        self.save_best_checkpoint(checkpoint_payload)
        if self.sync_after_epoch:
            self.sync_seed()

    def on_stage_transition(self, checkpoint_payload: dict, trainer_state: dict) -> None:
        self.save_last_checkpoint(checkpoint_payload)
        self.save_training_state(trainer_state)
        self.sync_seed()

    def on_seed_complete(self, info: SeedCompletionInfo | dict) -> None:
        self.mark_seed_complete(info)
        self.sync_seed()


def seed_status_report(
    experiment_name: str, seed: int, local_root: str | Path, persistent_root: str | Path | None,
    expected_split_sha256: str | None = None,
) -> dict:
    """Human-readable status for one seed -- powers the notebooks' "Seed NN:
    COMPLETE / INCOMPLETE / NOT STARTED" display and the CLI's startup log."""
    manager = PersistentArtifactManager(experiment_name, seed, local_root, persistent_root, sync_after_epoch=False)
    if manager.is_seed_complete(expected_split_sha256=expected_split_sha256):
        completion = manager.load_completion()
        return {"seed": seed, "status": "COMPLETE", "detail": f"reusing {completion.get('best_checkpoint')}"}
    try:
        checkpoint = manager.find_latest_valid_checkpoint()
    except CorruptedCheckpointError as exc:
        return {"seed": seed, "status": "CORRUPTED", "detail": str(exc)}
    if checkpoint is not None:
        stage = checkpoint.get("training_stage", "unknown")
        epoch = checkpoint.get("epoch", "unknown")
        return {"seed": seed, "status": "INCOMPLETE", "detail": f"resuming from epoch {epoch}, {stage}"}
    return {"seed": seed, "status": "NOT STARTED", "detail": "no checkpoint found"}


def format_status_line(status: dict) -> str:
    return f"Seed {status['seed']:<5}: {status['status']:<11} -- {status['detail']}"


def _read_json_or_none(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def scan_artifact_root(root: str | Path) -> list[dict]:
    """Scan a :class:`PersistentArtifactManager`-style root directory
    (``<root>/<experiment_name>/seed_<seed>/...``) and return one summary
    row per experiment/seed directory found -- read-only, never mutates
    anything under ``root``. Powers a notebook's live status-table display
    without duplicating that layout knowledge in notebook code.

    Each row: ``experiment``, ``seed``, ``status`` (``COMPLETE`` /
    ``INCOMPLETE`` / ``NOT STARTED`` / ``CORRUPTED``), ``stage``, ``epoch``,
    ``best_score``, ``last_update`` (UTC ISO timestamp of the most recently
    modified state file, or ``None``), ``checkpoint`` (best.pt if present,
    else last.pt, else ``None``).
    """
    root = Path(root)
    rows: list[dict] = []
    if not root.exists():
        return rows

    for experiment_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for seed_dir in sorted(experiment_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            try:
                seed = int(seed_dir.name[len("seed_"):])
            except ValueError:
                continue

            state_dir = seed_dir / "state"
            trainer_state = _read_json_or_none(state_dir / "trainer_state.json")
            completion = _read_json_or_none(state_dir / "completion.json")

            if completion is not None and completion.get("status") == "complete":
                status = "COMPLETE"
            elif trainer_state is not None:
                status = "INCOMPLETE"
            else:
                status = "NOT STARTED"

            stage = None
            epoch = None
            best_score = None
            if trainer_state is not None:
                stage = trainer_state.get("training_stage")
                epoch = trainer_state.get("global_epoch")
                best_score = trainer_state.get("best_validation_metric")
            if completion is not None:
                stage = stage or completion.get("training_stage")
                if best_score is None:
                    test_metrics = completion.get("test_metrics") or {}
                    best_score = test_metrics.get("balanced_score")

            checkpoint = None
            for name in ("best.pt", "last.pt"):
                candidate = seed_dir / "checkpoints" / name
                if candidate.exists():
                    checkpoint = str(candidate)
                    break

            last_update = None
            state_file = state_dir / "trainer_state.json"
            if state_file.exists():
                last_update = datetime.datetime.fromtimestamp(
                    state_file.stat().st_mtime, tz=datetime.timezone.utc
                ).isoformat()

            rows.append(
                {
                    "experiment": experiment_dir.name, "seed": seed, "status": status, "stage": stage,
                    "epoch": epoch, "best_score": best_score, "last_update": last_update, "checkpoint": checkpoint,
                }
            )
    return rows


_ARCHIVE_EXCLUDED_SUBDIRS = {"predictions"}
_ARCHIVE_CREDENTIAL_NAME_HINTS = ("secret", "credential", "service_account", "token", "api_key")


def build_summary_archive(
    experiment_root: str | Path,
    output_path: str | Path,
    extra_files: list[str | Path] | None = None,
    include_best_and_last_checkpoints: bool = False,
) -> Path:
    """Build (atomically) a zip summarizing every seed under
    ``experiment_root`` -- manifests, metrics, plots, completion markers,
    config snapshots, logs -- plus any ``extra_files`` (e.g.
    ``table_b.csv``). Never includes raw dataset images, cache/temp files,
    or anything credential-shaped (checked by filename).

    ``include_best_and_last_checkpoints=False`` (the default, used for the
    Colab/CLI "lightweight" ``transfer_learning_summary.zip`` -- see
    ``docs/transfer_learning.md`` "Archive contents") excludes every
    checkpoint. ``True`` (used for the Kaggle notebook's fuller
    ``agegender_transfer_learning_artifacts.zip``) includes ``best.pt``/
    ``last.pt`` but always excludes ``previous_last.pt`` (a duplicate of an
    already-superseded checkpoint) and ``predictions/`` either way.

    Called only at seed-completion and full-run-completion boundaries --
    never per-epoch (a large-checkpoint zip on every epoch would be slow
    and wasteful; per-epoch persistence is the atomic checkpoint writes
    themselves, not this archive).
    """
    experiment_root = Path(experiment_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    import zipfile

    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if experiment_root.exists():
            for path in experiment_root.rglob("*"):
                if not path.is_file():
                    continue
                rel_parts = path.relative_to(experiment_root).parts
                if any(part in _ARCHIVE_EXCLUDED_SUBDIRS for part in rel_parts):
                    continue
                if "checkpoints" in rel_parts:
                    if not include_best_and_last_checkpoints:
                        continue
                    if path.name not in ("best.pt", "last.pt"):
                        continue  # excludes previous_last.pt ("duplicate checkpoint") and any .tmp leftovers
                lower_name = path.name.lower()
                if any(hint in lower_name for hint in _ARCHIVE_CREDENTIAL_NAME_HINTS):
                    continue
                zf.write(path, arcname=str(Path(experiment_root.name) / Path(*rel_parts)))
        for extra in extra_files or []:
            extra = Path(extra)
            if extra.exists():
                zf.write(extra, arcname=extra.name)

    os.replace(tmp_path, output_path)
    return output_path
