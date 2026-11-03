"""Live databases on local disk, restored from and backed up to /data4."""

import sqlite3
from pathlib import Path

import pytest

from askiisc.web.dbfiles import backup, db_dir, restore


def _db(path: Path, rows: int) -> None:
    with sqlite3.connect(path) as db:
        db.execute("create table if not exists t(x)")
        db.executemany("insert into t values (?)", [(i,) for i in range(rows)])
    db.close()


def _count(path: Path) -> int:
    with sqlite3.connect(path) as db:
        n = int(db.execute("select count(*) from t").fetchone()[0])
    db.close()
    return n


def test_where_the_live_files_are(monkeypatch: pytest.MonkeyPatch) -> None:
    app = Path("/data4/x/private/app")
    monkeypatch.delenv("ASKIISC_DB_DIR", raising=False)
    monkeypatch.delenv("ASKIISC_SCRATCH", raising=False)
    assert db_dir(app) == app
    monkeypatch.setenv("ASKIISC_SCRATCH", "/data2/scratch/u/askiisc")
    assert db_dir(app) == Path("/data2/scratch/u/askiisc/app")
    monkeypatch.setenv("ASKIISC_DB_DIR", "/tmp/db")
    assert db_dir(app) == Path("/tmp/db")


def test_restore_then_backup(tmp_path: Path) -> None:
    app, live = tmp_path / "app", tmp_path / "live"
    app.mkdir()
    _db(app / "app.db", 3)
    assert restore(app, live) == ["app.db"]  # requests.db does not exist
    assert _count(live / "app.db") == 3
    assert (live.stat().st_mode & 0o777) == 0o700
    _db(live / "app.db", 2)
    assert restore(app, live) == []  # a live copy is never overwritten
    assert backup(live, app) == ["app.db"]
    assert _count(app / "app.db") == 5
    assert (app / "app.db").stat().st_mode & 0o777 == 0o600
    assert not list(app.glob("*.tmp"))
    assert restore(app, app) == [] and backup(app, app) == []
