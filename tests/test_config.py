"""Tests for .env loading (GENDER_LABEL_0/1, KAGGLE_* etc. are otherwise inert)."""

from __future__ import annotations

import os

from src.utils.config import _ENV_OVERRIDE_MAP, env_config_overrides, load_env_file

_ALL_ENV_KEYS = list(_ENV_OVERRIDE_MAP) + ["DATA_DIR"]


def _clear_all(monkeypatch):
    for key in _ALL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_load_env_file_sets_new_variable(tmp_path, monkeypatch):
    monkeypatch.delenv("SOME_TEST_VAR", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("SOME_TEST_VAR=hello\n")

    load_env_file(env_path)
    assert os.environ["SOME_TEST_VAR"] == "hello"
    del os.environ["SOME_TEST_VAR"]


def test_load_env_file_does_not_override_existing_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_TEST_VAR", "from_shell")
    env_path = tmp_path / ".env"
    env_path.write_text("SOME_TEST_VAR=from_dotenv\n")

    load_env_file(env_path)
    assert os.environ["SOME_TEST_VAR"] == "from_shell"


def test_load_env_file_is_a_noop_when_file_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.env"
    load_env_file(missing_path)  # must not raise


def test_env_config_overrides_empty_when_nothing_set(monkeypatch):
    _clear_all(monkeypatch)
    assert env_config_overrides() == {}


def test_env_config_overrides_coerces_numeric_types(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("AGE_MIN", "0")
    monkeypatch.setenv("AGE_MAX", "90")
    monkeypatch.setenv("GENDER_CONFIDENCE_THRESHOLD", "0.75")

    overrides = env_config_overrides()
    assert overrides["model"]["age_head"]["age_min"] == 0
    assert overrides["model"]["age_head"]["age_max"] == 90
    assert isinstance(overrides["model"]["age_head"]["age_max"], int)
    assert overrides["model"]["gender_head"]["confidence_threshold"] == 0.75


def test_env_config_overrides_dataset_source_and_checkpoint_dir(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("DATASET_SOURCE", "csv")
    monkeypatch.setenv("CHECKPOINT_DIR", "/tmp/my_checkpoints")
    monkeypatch.setenv("OUTPUT_DIR", "/tmp/my_outputs")

    overrides = env_config_overrides()
    assert overrides["dataset"]["source"] == "csv"
    assert overrides["paths"]["checkpoint_dir"] == "/tmp/my_checkpoints"
    assert overrides["paths"]["output_dir"] == "/tmp/my_outputs"


def test_env_config_overrides_data_dir_cascades_to_subdirs(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("DATA_DIR", "/mnt/mydata")

    overrides = env_config_overrides()
    assert overrides["paths"]["data_dir"] == "/mnt/mydata"
    assert overrides["paths"]["raw_dir"] == "/mnt/mydata/raw"
    assert overrides["paths"]["processed_dir"] == "/mnt/mydata/processed"
    assert overrides["paths"]["splits_dir"] == "/mnt/mydata/splits"


def test_env_config_overrides_ignores_blank_values(monkeypatch):
    _clear_all(monkeypatch)
    monkeypatch.setenv("AGE_MIN", "")
    assert env_config_overrides() == {}


def test_load_config_precedence_yaml_lt_env_lt_explicit_overrides(monkeypatch):
    """YAML default < .env override < explicit --set-style overrides.

    A real local .env (e.g. for running the actual app) may itself set
    every one of these variables, so the real ``load_env_file`` is
    stubbed out here -- this test is about override *precedence*, not
    about what any particular developer's .env happens to contain.
    """
    import src.utils.config as config_module

    _clear_all(monkeypatch)
    monkeypatch.setattr(config_module, "load_env_file", lambda *args, **kwargs: None)

    yaml_only = config_module.load_full_config()
    assert yaml_only["model"]["gender_head"]["confidence_threshold"] == 0.80  # configs/model.yaml default

    monkeypatch.setenv("GENDER_CONFIDENCE_THRESHOLD", "0.6")
    env_applied = config_module.load_full_config()
    assert env_applied["model"]["gender_head"]["confidence_threshold"] == 0.6

    explicit_applied = config_module.load_full_config(
        overrides={"model": {"gender_head": {"confidence_threshold": 0.99}}}
    )
    assert explicit_applied["model"]["gender_head"]["confidence_threshold"] == 0.99
