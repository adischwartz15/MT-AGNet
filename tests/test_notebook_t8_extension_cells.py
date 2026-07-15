"""Notebook-level checks for the optional Non-Parametric Baselines
section present in both notebooks.

Statically parses and validates the real, shipped .ipynb files rather
than reimplementing their logic (which could drift from what actually
ships). Parametrized over both notebooks since the section exists in
both.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COLAB_PATH = REPO_ROOT / "notebooks" / "train_evaluate_colab.ipynb"
KAGGLE_PATH = REPO_ROOT / "notebooks" / "train_evaluate_kaggle.ipynb"
NOTEBOOK_PATHS = [COLAB_PATH, KAGGLE_PATH]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_headers(nb: dict) -> list[tuple[int, str]]:
    out = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "markdown":
            continue
        text = "".join(c["source"]).strip()
        if text.startswith("#"):
            out.append((i, text.splitlines()[0]))
    return out


def _code_cell_containing(nb: dict, needle: str) -> str:
    return next(
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code" and needle in "".join(c["source"])
    )


# Anchor unique to the Non-Parametric Baselines *code* cell -- unlike
# "RUN_NONPARAMETRIC_BASELINES" (which also appears in the earlier USER
# CONFIGURATION cell that sets its default), this string only appears in
# the cell that actually runs the baselines.
_NONPARAM_CODE_ANCHOR = "tune_nonparametric_baselines.py"


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=lambda p: p.name)
class TestNonParametricBaselinesCell:
    def test_section_exists_and_before_archive(self, path):
        nb = _load(path)
        headers = _markdown_headers(nb)
        matches = [(i, h) for i, h in headers if h.startswith("## 21. Non-Parametric Baselines")]
        assert len(matches) == 1
        section_idx = matches[0][0]
        archive_idx = next(i for i, h in headers if h.startswith("## 22."))
        summary_idx = next(i for i, h in headers if h.startswith("## 20."))
        assert summary_idx < section_idx < archive_idx

    def test_toggle_defaults_on(self, path):
        nb = _load(path)
        config_code = _code_cell_containing(nb, "USER CONFIGURATION")
        assert "RUN_NONPARAMETRIC_BASELINES = True" in config_code

    def test_reuses_repository_scripts_not_duplicated_logic(self, path):
        nb = _load(path)
        code = _code_cell_containing(nb, _NONPARAM_CODE_ANCHOR)
        assert "tune_nonparametric_baselines.py" in code
        assert "evaluate_nonparametric_baselines.py" in code

    def test_code_is_syntactically_valid(self, path):
        nb = _load(path)
        code = _code_cell_containing(nb, _NONPARAM_CODE_ANCHOR)
        ast.parse(code)


def test_colab_new_cells_sync_to_drive_after_each_extension():
    """Only Colab has sync_after_phase (Drive persistence) -- Kaggle
    persists automatically via /kaggle/working, see test below."""
    nb = _load(COLAB_PATH)
    nonparam_code = _code_cell_containing(nb, _NONPARAM_CODE_ANCHOR)
    assert "sync_after_phase(" in nonparam_code


def test_kaggle_new_cells_do_not_reference_colab_only_drive_sync():
    """sync_after_phase is a Colab-only symbol (defined in a Colab-only
    cell that checks IN_COLAB) -- referencing it in the Kaggle notebook
    would raise NameError."""
    nb = _load(KAGGLE_PATH)
    nonparam_code = _code_cell_containing(nb, _NONPARAM_CODE_ANCHOR)
    assert "sync_after_phase(" not in nonparam_code


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_notebook_remains_valid_nbformat(path):
    import warnings

    nbformat = pytest.importorskip("nbformat")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
