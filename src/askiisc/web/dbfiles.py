"""Where the app's SQLite files live: local disk, backed up to /data4.

SQLite on the NFS drive (/data4) takes about 120 ms per commit, against
4 ms on the server's local disk, and every request commits several times
through one lock; that capped the chat app at about 5 requests/s. So the
live files are on local disk ($ASKIISC_SCRATCH/app, mode 700, private like
/data4), and a consistent copy (SQLite's backup API) goes back to the
private folder on /data4 every few minutes and at shutdown.

The serving job and the admin CLI use the same `db_dir`, so both always
open the live files.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path

DATABASES = ("app.db", "requests.db")
BACKUP_INTERVAL_S = 600.0


def db_dir(app: Path) -> Path:
    """The folder of the live database files.

    ASKIISC_DB_DIR if set, else $ASKIISC_SCRATCH/app when the Slurm
    environment is loaded, else the private app folder itself.
    """
    if os.environ.get("ASKIISC_DB_DIR"):
        return Path(os.environ["ASKIISC_DB_DIR"])
    scratch = os.environ.get("ASKIISC_SCRATCH")
    return Path(scratch) / "app" if scratch else app


def _copy(source: Path, target: Path) -> None:
    """A consistent copy of a live SQLite file (written, then renamed)."""
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    with sqlite3.connect(source) as src, sqlite3.connect(tmp) as dst:
        src.backup(dst)
    src.close()
    dst.close()
    tmp.chmod(0o600)
    tmp.replace(target)


def restore(
    app: Path, live: Path, names: Iterable[str] = DATABASES
) -> list[str]:
    """Copies backups from `app` to `live` for files `live` lacks.

    Returns:
        The names restored.
    """
    if live == app:
        return []
    live.mkdir(parents=True, exist_ok=True)
    live.chmod(0o700)
    restored = []
    for name in names:
        if not (live / name).exists() and (app / name).exists():
            _copy(app / name, live / name)
            restored.append(name)
    return restored


def backup(
    live: Path, app: Path, names: Iterable[str] = DATABASES
) -> list[str]:
    """Copies the live files to the private folder on /data4.

    Returns:
        The names backed up.
    """
    if live == app:
        return []
    done = []
    for name in names:
        if (live / name).exists():
            _copy(live / name, app / name)
            done.append(name)
    return done
