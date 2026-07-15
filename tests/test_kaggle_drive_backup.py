"""Tests for src/utils/kaggle_drive_backup.py.

No Kaggle, no Google Drive, no network -- ``kaggle_secrets``/``google.oauth2``/
``googleapiclient`` are not installed in this environment, and every test
here relies on that (proving the module degrades gracefully rather than
crashing) instead of mocking them in.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from src.utils import kaggle_drive_backup as backup


def test_module_never_imports_platform_packages_at_module_scope():
    """Static guard: kaggle_secrets/google.oauth2/googleapiclient must
    only ever be imported inside a function body, never at module scope --
    this is what makes it safe to import this module from Colab, local, or
    a Kaggle run with Drive backup disabled."""
    source = Path(backup.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned = {"kaggle_secrets", "google", "google.oauth2", "googleapiclient"}
    for node in tree.body:  # module-level statements only, not nested inside functions
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in {"kaggle_secrets", "google", "googleapiclient"}
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in {"kaggle_secrets", "google", "googleapiclient"}


def test_is_configured_false_without_kaggle_secrets_module():
    assert backup.is_configured() is False


def test_upload_file_returns_false_without_crashing_when_secrets_missing(tmp_path, caplog):
    target = tmp_path / "artifact.txt"
    target.write_text("hello", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        result = backup.upload_file(target)

    assert result is False
    assert any("Kaggle Secrets are not both set" in record.message for record in caplog.records)


def test_upload_file_never_logs_secret_value(tmp_path, monkeypatch, caplog):
    """Even if a secret happens to be readable (simulated here), its value
    must never appear in a log message."""
    target = tmp_path / "artifact.txt"
    target.write_text("hello", encoding="utf-8")

    secret_value = "super-secret-service-account-json-payload"
    monkeypatch.setattr(
        backup, "_read_kaggle_secret",
        lambda name: secret_value if name == backup.SECRET_SERVICE_ACCOUNT_JSON else "folder-id-123",
    )

    with caplog.at_level(logging.WARNING):
        result = backup.upload_file(target)

    # No google-api-python-client installed -> fails soft, but must still
    # never have logged the secret payload itself.
    assert result is False
    assert not any(secret_value in record.message for record in caplog.records)


def test_upload_file_returns_false_for_missing_local_file(monkeypatch, tmp_path):
    monkeypatch.setattr(backup, "_read_kaggle_secret", lambda name: "configured")
    assert backup.upload_file(tmp_path / "does_not_exist.txt") is False


def test_upload_paths_isolates_failures(tmp_path):
    ok_path = tmp_path / "a.txt"
    ok_path.write_text("x", encoding="utf-8")
    missing_path = tmp_path / "missing.txt"

    results = backup.upload_paths([ok_path, missing_path])
    assert results[str(ok_path)] is False  # no secrets configured in this test environment
    assert results[str(missing_path)] is False
    assert set(results) == {str(ok_path), str(missing_path)}


def test_download_file_returns_false_without_crashing_when_secrets_missing(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        result = backup.download_file(tmp_path / "restored.zip")
    assert result is False
    assert any("Kaggle Secrets are not both set" in record.message for record in caplog.records)


def test_download_file_never_writes_partial_file_on_missing_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(backup, "_read_kaggle_secret", lambda name: "configured")
    target = tmp_path / "restored.zip"
    result = backup.download_file(target)
    assert result is False
    assert not target.exists()


def test_credentials_never_written_to_disk_on_upload_attempt(tmp_path, monkeypatch):
    """A generic regression guard: after an upload_file() call (success or
    failure), no new file appears anywhere under tmp_path except the
    artifact itself -- i.e. no stray credential file was written."""
    target = tmp_path / "artifact.txt"
    target.write_text("hello", encoding="utf-8")
    monkeypatch.setattr(backup, "_read_kaggle_secret", lambda name: "configured-but-fake")

    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    backup.upload_file(target)
    after = {p for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
