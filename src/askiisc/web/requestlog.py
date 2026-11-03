"""The admin request log: one row per request, never any text.

Rows name the user (admins need to see who is using the service) but hold
no question, answer or chat text. Writes are queued and flushed by one
background thread, so a request never waits for the disk. Old rows are
dropped after `retention_days`.
"""

from __future__ import annotations

import json
import math
import queue
import sqlite3
import threading
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from askiisc.gateway.app import RequestEvent

RETENTION_DAYS = 30
FLUSH_S = 0.5
FLUSH_ROWS = 500

COLUMNS = (
    "request_id",
    "time",
    "user_id",
    "department",
    "role",
    "source",
    "synthetic",
    "decision",
    "status",
    "latency_ms",
    "ttft_ms",
    "prompt_tokens",
    "completion_tokens",
    "cache_hit",
    "stale",
    "citations",
    "documents",
    "model",
    "policy_version",
    "cold_hold",
)
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS requests ({", ".join(COLUMNS)},
    PRIMARY KEY (request_id));
CREATE INDEX IF NOT EXISTS requests_time ON requests (time);
CREATE INDEX IF NOT EXISTS requests_user ON requests (user_id, time);
"""
FILTERS = ("user_id", "department", "source", "decision", "status")


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile, or None for no values."""
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, max(0, math.ceil(q / 100 * len(s)) - 1))]


def _latency_stats(rows: Iterable[tuple[Any, ...]]) -> dict[str, Any]:
    latency = [r[0] for r in rows if r[0] is not None]
    return {
        f"p{q}_ms": None if (v := percentile(latency, q)) is None else round(v)
        for q in (50, 95, 99)
    }


class RequestLog:
    """SQLite request log with a background writer.

    Args:
        path: The SQLite file (in the private folder).
        role_of: Looks up a user's role in the directory.
        now: Current time (for retention).
        lines: Also append each request as one JSON line here
            (requests.log, shipped to Loki).
    """

    def __init__(
        self,
        path: Path,
        role_of: Callable[[str], str] = lambda user: "",
        now: float | None = None,
        lines: Path | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lines = lines
        self._db = sqlite3.connect(
            str(path), check_same_thread=False, timeout=60.0
        )
        self._lock = threading.Lock()
        self._role_of = role_of
        self._queue: queue.Queue[RequestEvent | None] = queue.Queue()
        with self._lock, self._db:
            self._db.executescript(_SCHEMA)
            if now is not None:
                self._db.execute(
                    "DELETE FROM requests WHERE time < ?",
                    (now - RETENTION_DAYS * 86400,),
                )
        self._thread: threading.Thread | None = None
        self._stopping = False

    def start(self) -> None:
        """Starts the background writer."""
        self._thread = threading.Thread(
            target=self._run, name="request-log", daemon=True
        )
        self._thread.start()

    def record(self, event: RequestEvent) -> None:
        """Queues one request (the gateway's on_request hook)."""
        self._queue.put(event)

    def _row(self, event: RequestEvent) -> tuple[Any, ...]:
        data = asdict(event)
        data["department"] = data.pop("tenant")
        data["role"] = self._role_of(event.user_id)
        data["documents"] = json.dumps(list(event.documents))
        return tuple(data[c] for c in COLUMNS)

    def flush(self) -> int:
        """Writes everything queued; returns the number of rows."""
        rows = []
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break
            if event is None:
                self._stopping = True
            else:
                rows.append(self._row(event))
        if rows:
            marks = ", ".join("?" * len(COLUMNS))
            with self._lock, self._db:
                self._db.executemany(
                    f"INSERT OR REPLACE INTO requests VALUES ({marks})",  # noqa: S608  # nosec B608
                    rows,
                )
            self._write_lines(rows)
        return len(rows)

    def _write_lines(self, rows: list[tuple[Any, ...]]) -> None:
        """Appends one JSON line per request (shipped to Loki).

        user_id and request_id are fields of the line, not Loki labels.
        """
        if self._lines is None:
            return
        with self._lines.open("a") as out:
            for row in rows:
                line = dict(zip(COLUMNS, row, strict=True))
                line["documents"] = json.loads(line["documents"])
                out.write(json.dumps(line) + "\n")

    def _run(self) -> None:  # pragma: no cover - timing loop over flush()
        while not self._stopping:
            try:
                first = self._queue.get(timeout=FLUSH_S)
            except queue.Empty:
                continue
            self._queue.put(first)
            self.flush()

    def close(self) -> None:
        """Flushes, stops the writer and closes the file."""
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=10)
        self.flush()
        self._db.close()

    # -- queries ---------------------------------------------------------

    @staticmethod
    def _where(
        departments: set[str] | None,
        since: float | None = None,
        until: float | None = None,
        filters: dict[str, object] | None = None,
    ) -> tuple[str, list[Any]]:
        filters = filters or {}
        clauses: list[str] = ["1 = 1"]
        args: list[Any] = []
        if departments is not None:
            clauses.append(
                f"department IN ({', '.join('?' * len(departments))})"
            )
            args += sorted(departments)
        if since is not None:
            clauses.append("time >= ?")
            args.append(since)
        if until is not None:
            clauses.append("time < ?")
            args.append(until)
        for key in FILTERS:
            value = filters.get(key)
            if value not in (None, ""):
                clauses.append(f"{key} = ?")
                args.append(value)
        if filters.get("synthetic") is not None:
            clauses.append("synthetic = ?")
            args.append(int(bool(filters["synthetic"])))
        return " AND ".join(clauses), args

    def _query(self, sql: str, args: list[Any]) -> list[tuple[Any, ...]]:
        with self._lock:
            return [tuple(r) for r in self._db.execute(sql, args).fetchall()]

    def rows(
        self,
        departments: set[str] | None,
        limit: int = 100,
        before: float | None = None,
        offset: int = 0,
        **filters: object,
    ) -> list[dict[str, Any]]:
        """The latest requests, newest first, optionally filtered."""
        where, args = self._where(departments, until=before, filters=filters)
        out = self._query(
            f"SELECT {', '.join(COLUMNS)} FROM requests "  # noqa: S608  # nosec B608
            f"WHERE {where} ORDER BY time DESC LIMIT ? OFFSET ?",
            [*args, limit, offset],
        )
        rows = [dict(zip(COLUMNS, r, strict=True)) for r in out]
        for row in rows:
            row["documents"] = json.loads(row["documents"])
            for flag in ("synthetic", "cache_hit", "stale", "cold_hold"):
                row[flag] = bool(row[flag])
        return rows

    def count(self, departments: set[str] | None, **filters: object) -> int:
        """How many requests match (for the request log's pages)."""
        where, args = self._where(departments, filters=filters)
        return int(
            self._query(f"SELECT count(*) FROM requests WHERE {where}", args)[  # noqa: S608  # nosec B608
                0
            ][0]
        )

    def summary(
        self,
        departments: set[str] | None,
        since: float,
        now: float,
        **filters: object,
    ) -> dict[str, Any]:
        """Totals and latency percentiles since a time."""
        where, args = self._where(departments, since, filters=filters)
        rows = self._query(
            "SELECT latency_ms, ttft_ms, synthetic, decision, status, "  # noqa: S608  # nosec B608
            "completion_tokens, cache_hit, user_id, source "
            f"FROM requests WHERE {where}",
            args,
        )
        served = [r for r in rows if r[3] in {"answered", "abstained"}]
        recent = self._query(
            f"SELECT count(*) FROM requests WHERE {where} AND time >= ?",  # noqa: S608  # nosec B608
            [*args, now - 60],
        )[0][0]

        def count(index: int) -> dict[str, int]:
            out: dict[str, int] = {}
            for r in rows:
                out[str(r[index])] = out.get(str(r[index]), 0) + 1
            return out

        ttft = [r[1] for r in served if r[1] is not None]
        return {
            "requests": len(rows),
            "real": sum(1 for r in rows if not r[2]),
            "locust": sum(1 for r in rows if r[2]),
            "by_decision": count(3),
            "by_status": count(4),
            "by_source": count(8),
            "errors": sum(1 for r in rows if r[3] == "error"),
            "rejected_429": sum(1 for r in rows if r[4] == 429),  # noqa: PLR2004
            "users_active": len({r[7] for r in rows}),
            "rate_per_s_last_minute": round(recent / 60, 3),
            "output_tokens": sum(r[5] or 0 for r in rows),
            "cache_hit_rate": round(
                sum(1 for r in served if r[6]) / len(served), 3
            )
            if served
            else None,
            "ttft_p95_ms": None
            if (t := percentile(ttft, 95)) is None
            else round(t),
            **_latency_stats(served),
        }

    def series(
        self,
        departments: set[str] | None,
        since: float,
        step_s: float,
        **filters: object,
    ) -> list[dict[str, Any]]:
        """Per-interval counts and latency percentiles, oldest first."""
        where, args = self._where(departments, since, filters=filters)
        rows = self._query(
            "SELECT time, latency_ms, synthetic, decision, status "  # noqa: S608  # nosec B608
            f"FROM requests WHERE {where} ORDER BY time",
            args,
        )
        buckets: dict[int, list[tuple[Any, ...]]] = {}
        for r in rows:
            buckets.setdefault(int((r[0] - since) // step_s), []).append(r)
        out = []
        for index in sorted(buckets):
            group = buckets[index]
            served = [
                (r[1],) for r in group if r[3] in {"answered", "abstained"}
            ]
            out.append(
                {
                    "time": since + index * step_s,
                    "real": sum(1 for r in group if not r[2]),
                    "locust": sum(1 for r in group if r[2]),
                    "errors": sum(
                        1
                        for r in group
                        if r[3] == "error" or r[4] >= 500  # noqa: PLR2004
                    ),
                    "rejected": sum(1 for r in group if r[4] == 429),  # noqa: PLR2004
                    "rate_per_s": round(len(group) / step_s, 3),
                    **_latency_stats(served),
                }
            )
        return out

    def per_user(
        self, departments: set[str] | None, since: float, **filters: object
    ) -> list[dict[str, Any]]:
        """Requests per user since a time, busiest first."""
        where, args = self._where(departments, since, filters=filters)
        rows = self._query(
            "SELECT user_id, department, role, synthetic, count(*), "  # noqa: S608  # nosec B608
            "sum(decision = 'error'), sum(status = 429), "
            "sum(completion_tokens), max(time), avg(latency_ms) "
            f"FROM requests WHERE {where} GROUP BY user_id "
            "ORDER BY count(*) DESC",
            args,
        )
        keys = (
            "user_id",
            "department",
            "role",
            "synthetic",
            "requests",
            "errors",
            "rejected_429",
            "output_tokens",
            "last_request",
            "mean_latency_ms",
        )
        out = [dict(zip(keys, r, strict=True)) for r in rows]
        for row in out:
            row["synthetic"] = bool(row["synthetic"])
            row["mean_latency_ms"] = round(row["mean_latency_ms"] or 0)
        return out
