"""YAML configuration loading, deep-merging, and validation.

All scripts load configuration through :func:`load_config`, which:

1. Loads ``.env`` into ``os.environ`` (see :func:`load_env_file`).
2. Merges ``configs/default.yaml`` with any number of additional YAML
   files (later files win on key conflicts).
3. Merges in overrides derived from recognized environment variables
   (see :func:`env_config_overrides` / ``.env.example``).
4. Merges in an explicit dotted-key override dict, highest priority,
   typically parsed from a CLI ``--set model.adapters.bottleneck_dim=64``.

So the precedence, lowest to highest, is: YAML defaults < .env < --set.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


class ConfigError(ValueError):
    """Raised when a configuration file is missing required keys."""


def load_env_file(env_path: str | os.PathLike | None = None) -> None:
    """Load ``.env`` into ``os.environ`` if present (no-op if missing).

    Values already set in the real environment are never overwritten
    (``override=False``), so an explicit ``export`` / Colab
    ``os.environ[...] = ...`` always wins over a stale ``.env`` file.
    Call this once near the start of any entry point (CLI script) that
    reads Kaggle credentials, GENDER_LABEL_0/1, etc. from the
    environment -- nothing in this repo reads ``.env`` automatically
    otherwise.
    """
    from dotenv import load_dotenv

    path = Path(env_path) if env_path else REPO_ROOT / ".env"
    if path.exists():
        load_dotenv(path, override=False)


# Maps .env.example variables to the config key they override. Only
# affects config *defaults*: a loaded checkpoint always uses the config
# snapshot saved at its own training time (see src/inference/artifacts.py),
# so changing these after a model is trained cannot desync a checkpoint's
# saved age range / confidence threshold from what it actually trained
# with -- these only matter for a *new* training run.
_ENV_OVERRIDE_MAP: dict[str, str] = {
    "DATASET_SOURCE": "dataset.source",
    "GENDER_CONFIDENCE_THRESHOLD": "model.gender_head.confidence_threshold",
    "AGE_MIN": "model.age_head.age_min",
    "AGE_MAX": "model.age_head.age_max",
    "CHECKPOINT_DIR": "paths.checkpoint_dir",
    "OUTPUT_DIR": "paths.output_dir",
}


def env_config_overrides() -> dict[str, Any]:
    """Build a config-override dict from recognized environment variables.

    Applied with lower priority than an explicit ``overrides`` dict (e.g.
    CLI ``--set``) but higher priority than ``configs/*.yaml`` -- ``.env``
    lets you change common settings without editing YAML, while ``--set``
    still wins for a one-off run.

    ``DATA_DIR`` additionally re-derives ``paths.raw_dir`` /
    ``paths.processed_dir`` / ``paths.splits_dir`` as subdirectories of it,
    since those are independent keys in ``configs/default.yaml`` rather
    than being computed from ``paths.data_dir`` at load time.
    """
    overrides: dict[str, Any] = {}
    for env_key, dotted_path in _ENV_OVERRIDE_MAP.items():
        raw_value = os.environ.get(env_key)
        if raw_value:
            set_by_dotted_key(overrides, dotted_path, _coerce_scalar(raw_value))

    data_dir = os.environ.get("DATA_DIR")
    if data_dir:
        set_by_dotted_key(overrides, "paths.data_dir", data_dir)
        set_by_dotted_key(overrides, "paths.raw_dir", f"{data_dir}/raw")
        set_by_dotted_key(overrides, "paths.processed_dir", f"{data_dir}/processed")
        set_by_dotted_key(overrides, "paths.splits_dir", f"{data_dir}/splits")

    return overrides


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a top-level mapping")
    return data


def set_by_dotted_key(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``config[a][b][c] = value`` given ``dotted_key == "a.b.c"``."""
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _coerce_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in raw or "e" in lowered:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def parse_cli_overrides(overrides: list[str] | None) -> dict[str, Any]:
    """Parse ``["a.b.c=1", "x.y=foo"]`` CLI overrides into a nested dict."""
    result: dict[str, Any] = {}
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"Invalid override '{item}', expected key=value")
        key, raw_value = item.split("=", 1)
        set_by_dotted_key(result, key.strip(), _coerce_scalar(raw_value.strip()))
    return result


def load_config(
    *extra_files: str | Path,
    base: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and deep-merge one or more YAML config files.

    Parameters
    ----------
    extra_files:
        Additional YAML files merged on top of ``base``, in order.
    base:
        Base config file, defaults to ``configs/default.yaml``.
    overrides:
        A nested dict merged on top of everything else (highest priority),
        typically produced by :func:`parse_cli_overrides`.
    """
    load_env_file()  # populate os.environ from .env before reading env var overrides below

    base_path = Path(base) if base is not None else CONFIG_DIR / "default.yaml"
    merged = _load_yaml(base_path)
    for extra in extra_files:
        merged = _deep_merge(merged, _load_yaml(extra))
    merged = _deep_merge(merged, env_config_overrides())
    if overrides:
        merged = _deep_merge(merged, overrides)
    return merged


def load_full_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Convenience loader that merges all standard config files.

    Merges default -> data -> model -> training, which is the combination
    most training/eval scripts need.
    """
    return load_config(
        CONFIG_DIR / "data.yaml",
        CONFIG_DIR / "model.yaml",
        CONFIG_DIR / "training.yaml",
        overrides=overrides,
    )


def resolve_path(relative_or_absolute: str | os.PathLike) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(relative_or_absolute)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def resolve_device(device_setting: str) -> str:
    """Resolve the "auto" device setting to "cuda", "mps", or "cpu".

    Checks CUDA first (NVIDIA GPUs), then Apple Silicon's Metal backend
    (``torch.backends.mps``), falling back to CPU when neither is
    available or torch itself isn't importable. An explicit
    ``device_setting`` (not "auto") is always returned unchanged -- this
    only resolves the "pick the best available device" case.
    """
    if device_setting != "auto":
        return device_setting
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"
