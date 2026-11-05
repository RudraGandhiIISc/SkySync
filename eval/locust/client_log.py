"""The load generator's own record of every request it sent (FR-12 check).

One JSON line per request: the gateway's request id, the HTTP status, a
question id (never the text), the citations the answer carried, whether a
`warming` event arrived, and the client-side time. eval/audit_check.py
matches these lines against the audit store.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

Cited = tuple[str, int, str]
"""(doc_id, version, chunk_id) of one citation."""


@dataclass(frozen=True)
class ClientRecord:
    """One request as the client saw it."""

    time: float
    request_id: str | None
    status: int
    question_id: str | None
    citations: tuple[Cited, ...] = ()
    warming: bool = False
    seconds: float = 0.0


def citations_of(answer: dict[str, Any] | None) -> tuple[Cited, ...]:
    """The (doc_id, version, chunk_id) triples of an answer event."""
    if not answer:
        return ()
    return tuple(
        (str(c["doc_id"]), int(c["version"]), str(c["chunk_id"]))
        for c in answer.get("citations", [])
    )


class ClientLog:
    """Thread-safe list of ClientRecords for one run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: list[ClientRecord] = []

    def add(self, record: ClientRecord) -> None:
        """Appends a record."""
        with self._lock:
            self._rows.append(record)

    def clear(self) -> None:
        """Forgets every record (a new run started)."""
        with self._lock:
            self._rows = []

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)

    def write(self, path: Path) -> None:
        """Writes every record as JSON lines."""
        with self._lock:
            lines = [json.dumps(asdict(r)) for r in self._rows]
        path.write_text("".join(line + "\n" for line in lines))


def read_client_log(path: Path) -> list[ClientRecord]:
    """Reads a client log back.

    Raises:
        ValueError: On a line that is not a client record.
    """
    out = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            row["citations"] = tuple(tuple(c) for c in row["citations"])
            out.append(ClientRecord(**row))
        except (KeyError, TypeError) as err:
            raise ValueError(f"{path}:{n}: not a client record") from err
    return out
