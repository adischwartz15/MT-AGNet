"""Optional, secure Google Drive backup for Kaggle notebook runs.

Kaggle notebook outputs already persist under ``/kaggle/working`` (saved as
a notebook-version's output dataset) -- this module exists only for the
*additional*, opt-in case where a user wants a second, off-Kaggle copy of
the run's artifacts. It is never required for the Kaggle notebook's own
resume/persistence behavior (see
``src/training/persistent_artifacts.py::PersistentArtifactManager``, which
works entirely off ``/kaggle/working`` and an optional attached
prior-output dataset).

Security contract:

- Credentials are read **only** from Kaggle Secrets
  (``GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON``, ``GOOGLE_DRIVE_FOLDER_ID``) via
  ``kaggle_secrets.UserSecretsClient`` -- never from a committed file, a
  notebook cell literal, an environment variable set in the notebook, or
  any other source.
- The service-account JSON is held only in memory (parsed straight into an
  in-memory ``google.oauth2.service_account.Credentials`` object) and is
  **never written to disk** -- not to a temp file, not to a log, not to any
  archive this module or its callers produce.
- The secret value is never printed or logged, in success or failure paths.
- A missing secret, a missing optional dependency
  (``google-api-python-client``/``google-auth``), a network failure, or any
  Drive API error is handled by logging a clear warning and returning
  ``False`` -- this module never raises out of :func:`upload_file`, so a
  Drive hiccup can never fail or interrupt a training run. Callers must
  keep saving to ``/kaggle/working`` regardless of this module's outcome.

Every import of ``kaggle_secrets``, ``google.oauth2``, and
``googleapiclient`` is lazy (inside the function that needs it) -- neither
this module nor anything that imports it requires those packages to be
installed for Colab or local (non-Kaggle) execution, or even for a Kaggle
run that leaves Drive backup disabled.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SECRET_SERVICE_ACCOUNT_JSON = "GOOGLE_DRIVE_SERVICE_ACCOUNT_JSON"
SECRET_FOLDER_ID = "GOOGLE_DRIVE_FOLDER_ID"

_DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _read_kaggle_secret(name: str) -> str | None:
    """Read one secret from Kaggle Secrets.

    Returns ``None`` (never raises) if running outside Kaggle, the secret
    is unset, or the Kaggle secrets API is unavailable -- callers treat a
    missing secret as "Drive backup not configured", not an error. Never
    logs the secret's value.
    """
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return None
    try:
        return UserSecretsClient().get_secret(name)
    except Exception:
        logger.warning("Kaggle secret '%s' is not set or not accessible.", name)
        return None


def is_configured() -> bool:
    """True only if both required Kaggle Secrets are present.

    A cheap, no-network check -- does not attempt to build a Drive client
    or verify the credentials/folder are actually valid (that only happens
    inside :func:`upload_file`, which fails soft on any error).
    """
    return bool(_read_kaggle_secret(SECRET_SERVICE_ACCOUNT_JSON)) and bool(_read_kaggle_secret(SECRET_FOLDER_ID))


def _build_drive_service(service_account_json: str):
    """Construct an in-memory Drive API client from service-account JSON text.

    The JSON is parsed directly into an in-memory credentials object and
    never touches disk.
    """
    import json as _json

    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    info = _json.loads(service_account_json)
    credentials = service_account.Credentials.from_service_account_info(info, scopes=_DRIVE_SCOPES)
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def _find_existing_file_id(service, filename: str, folder_id: str) -> str | None:
    query = f"name = '{filename}' and '{folder_id}' in parents and trashed = false"
    response = service.files().list(q=query, spaces="drive", fields="files(id)").execute()
    files = response.get("files", [])
    return files[0]["id"] if files else None


def upload_file(local_path: str | Path, drive_filename: str | None = None) -> bool:
    """Upload (or update, if a same-named file already exists) one file to
    the configured Drive folder.

    Returns ``True`` on success, ``False`` on *any* failure -- missing
    secrets, a missing optional dependency, no network, or a Drive API
    error. Never raises: a Drive backup failure must never fail or
    interrupt the training run that called it (the file is always still
    safe under ``/kaggle/working`` regardless of this function's outcome).
    """
    local_path = Path(local_path)
    service_account_json = _read_kaggle_secret(SECRET_SERVICE_ACCOUNT_JSON)
    folder_id = _read_kaggle_secret(SECRET_FOLDER_ID)
    if not service_account_json or not folder_id:
        logger.warning(
            "Google Drive backup is enabled but '%s'/'%s' Kaggle Secrets are not both set -- "
            "skipping Drive upload for '%s'. Artifacts under /kaggle/working are unaffected.",
            SECRET_SERVICE_ACCOUNT_JSON, SECRET_FOLDER_ID, local_path.name,
        )
        return False
    if not local_path.exists():
        logger.warning("Cannot upload '%s' to Drive -- file does not exist.", local_path)
        return False

    try:
        from googleapiclient.http import MediaFileUpload
    except ImportError:
        logger.warning(
            "Google Drive backup is enabled but the optional 'google-api-python-client'/"
            "'google-auth' packages are not installed -- skipping Drive upload. Install "
            "them with 'pip install -r requirements-kaggle-drive.txt'.",
        )
        return False

    try:
        service = _build_drive_service(service_account_json)
        filename = drive_filename or local_path.name
        media = MediaFileUpload(str(local_path), resumable=True)
        existing_id = _find_existing_file_id(service, filename, folder_id)
        if existing_id:
            service.files().update(fileId=existing_id, media_body=media).execute()
        else:
            service.files().create(body={"name": filename, "parents": [folder_id]}, media_body=media).execute()
        logger.info("Uploaded '%s' to Google Drive (folder id %s).", filename, folder_id)
        return True
    except Exception as exc:
        logger.warning(
            "Google Drive upload of '%s' failed (%s) -- artifacts under /kaggle/working are unaffected.",
            local_path.name, exc,
        )
        return False
    finally:
        # Drop the in-memory reference as soon as this function is done
        # with it -- belt-and-braces alongside "never written to disk".
        service_account_json = None  # noqa: F841


def upload_paths(paths: list[str | Path]) -> dict[str, bool]:
    """Upload multiple files; a failure on one never stops the rest.

    Returns ``{str(path): success}`` so a caller can report exactly which
    artifacts made it to the Drive mirror and which didn't.
    """
    return {str(path): upload_file(path) for path in paths}


def download_file(local_path: str | Path, drive_filename: str | None = None) -> bool:
    """Download one file (by name) from the configured Drive folder to
    ``local_path``. Returns ``True`` on success, ``False`` on any failure --
    same fail-soft contract as :func:`upload_file`.

    Scope note: this restores one named file (e.g. a lightweight summary
    archive), not an entire seed's checkpoint tree -- the primary Kaggle
    restore path for full checkpoints is attaching a previous run's
    notebook output as an input dataset (a plain read-only mounted
    directory, ``PersistentArtifactManager.restore_seed()`` handles that
    natively).
    """
    local_path = Path(local_path)
    service_account_json = _read_kaggle_secret(SECRET_SERVICE_ACCOUNT_JSON)
    folder_id = _read_kaggle_secret(SECRET_FOLDER_ID)
    if not service_account_json or not folder_id:
        logger.warning(
            "Google Drive restore is enabled but '%s'/'%s' Kaggle Secrets are not both set -- "
            "skipping Drive download for '%s'.",
            SECRET_SERVICE_ACCOUNT_JSON, SECRET_FOLDER_ID, drive_filename or local_path.name,
        )
        return False

    try:
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError:
        logger.warning(
            "Google Drive restore is enabled but the optional 'google-api-python-client'/"
            "'google-auth' packages are not installed -- skipping Drive download.",
        )
        return False

    filename = drive_filename or local_path.name
    try:
        import os

        service = _build_drive_service(service_account_json)
        file_id = _find_existing_file_id(service, filename, folder_id)
        if not file_id:
            logger.warning("No file named '%s' found in the configured Drive folder.", filename)
            return False

        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
        request = service.files().get_media(fileId=file_id)
        with open(tmp_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        os.replace(tmp_path, local_path)
        logger.info("Downloaded '%s' from Google Drive to '%s'.", filename, local_path)
        return True
    except Exception as exc:
        logger.warning("Google Drive download of '%s' failed (%s).", filename, exc)
        return False
    finally:
        service_account_json = None  # noqa: F841
