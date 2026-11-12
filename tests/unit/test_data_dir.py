"""Commands run outside a Slurm job use the caller's own data folder."""

import getpass
from pathlib import Path

import pytest

from askiisc.common.config import data_dir


def test_askiisc_data_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASKIISC_DATA", "/somewhere/else")
    assert data_dir() == Path("/somewhere/else")


def test_default_is_the_callers_folder_not_the_authors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ASKIISC_DATA", raising=False)
    monkeypatch.setattr(getpass, "getuser", lambda: "redteamer")
    assert data_dir() == Path("/data4/redteamer/askiisc")
    monkeypatch.setenv("ASKIISC_DATA", "")
    assert data_dir() == Path("/data4/redteamer/askiisc")
