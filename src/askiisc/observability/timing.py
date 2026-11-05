"""Time at the serving node: responses, event-loop lag, stack samples (FR-6).

FR-6 is judged at the serving node (2026-11-05 decision): from a request
reaching the server to its response being sent. Three instruments:

- `ResponseTimer`, the outermost ASGI middleware, writes one line per
  answered request: milliseconds from the app receiving it to the response
  start and to the last body byte handed to the server.
- `LoopLag` samples how late the event loop runs a timer. A request's
  bytes wait that long before the server parses them, which the middleware
  cannot see, so the run reports both.
- `StackSampler` (ASKIISC_PROFILE=1) counts the main thread's Python
  stacks from a background thread, to find where the loop spends its time.
  It needs no profiler package.

No line holds question text, user ids or paths with ids.
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from types import FrameType
from typing import Any, TextIO

REQUEST_ID = b"x-request-id"


@dataclass(frozen=True)
class ResponseTiming:
    """One answered request, as the serving node saw it."""

    time: float
    kind: str
    status: int
    request_id: str | None
    start_ms: float
    done_ms: float


class JsonLines:
    """Appends JSON objects to a file, one per line."""

    def __init__(self, path: Path) -> None:
        self._file: TextIO = path.open("a", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        """Writes one row and flushes it (readable while the run goes on)."""
        self._file.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._file.flush()

    def close(self) -> None:
        """Flushes and closes the file."""
        self._file.close()


def ask_kind(path: str) -> str | None:
    """Names the question routes; None for every other path.

    Args:
        path: The request path.

    Returns:
        "temporary", "chat" or "api", or None.
    """
    if path == "/api/temporary":
        return "temporary"
    if path.startswith("/api/chats/") and path.endswith("/messages"):
        return "chat"
    if path == "/v1/ask":
        return "api"
    return None


class ResponseTimer:
    """ASGI middleware timing every question request (see module doc)."""

    def __init__(
        self,
        app: Any,
        record: Callable[[ResponseTiming], None],
        monotonic: Callable[[], float],
        wall: Callable[[], float],
        kind: Callable[[str], str | None] = ask_kind,
    ) -> None:
        """Wraps an ASGI app.

        Args:
            app: The app.
            record: Receives each timing.
            monotonic: Seconds, for durations.
            wall: Seconds since the epoch, for the row's time.
            kind: Names the paths to time (None: not timed).
        """
        self._app = app
        self._record = record
        self._mono = monotonic
        self._wall = wall
        self._kind = kind

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Passes the request on, timing its response."""
        kind = (
            self._kind(scope.get("path", ""))
            if scope["type"] == "http" and scope.get("method") == "POST"
            else None
        )
        if kind is None:
            await self._app(scope, receive, send)
            return
        await self._app(scope, receive, self._timed(kind, send))

    def _timed(
        self, kind: str, send: Callable[[Any], Awaitable[None]]
    ) -> Callable[[Any], Awaitable[None]]:
        arrived, wall = self._mono(), self._wall()
        state: dict[str, Any] = {}

        async def timed(message: Any) -> None:
            await send(message)
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                state["start"] = self._mono() - arrived
                state["id"] = _header(message, REQUEST_ID)
            elif message["type"] == "http.response.body" and not message.get(
                "more_body", False
            ):
                self._record(
                    ResponseTiming(
                        wall,
                        kind,
                        state.get("status", 0),
                        state.get("id"),
                        1000 * state.get("start", 0.0),
                        1000 * (self._mono() - arrived),
                    )
                )

        return timed


def _header(message: Any, name: bytes) -> str | None:
    for key, value in message.get("headers", []):
        if key.lower() == name:
            return bytes(value).decode()
    return None


def timing_row(timing: ResponseTiming) -> dict[str, Any]:
    """A timing as a JSON-ready row, durations rounded to 0.01 ms."""
    row = asdict(timing)
    row["start_ms"] = round(timing.start_ms, 2)
    row["done_ms"] = round(timing.done_ms, 2)
    return row


@dataclass(frozen=True)
class LagWindow:
    """Event-loop lag over one window."""

    time: float
    samples: int
    mean_ms: float
    max_ms: float


class LoopLag:
    """Measures how late the event loop wakes a timer."""

    def __init__(
        self,
        sleep: Callable[[float], Awaitable[None]],
        monotonic: Callable[[], float],
        wall: Callable[[], float],
        record: Callable[[LagWindow], None],
        interval_s: float = 0.01,
        window_s: float = 1.0,
    ) -> None:
        """Creates the probe.

        Args:
            sleep: Asynchronous sleep.
            monotonic: Seconds, for durations.
            wall: Seconds since the epoch, for each window's time.
            record: Receives one summary per window.
            interval_s: Time between samples.
            window_s: Length of a window.
        """
        self._sleep = sleep
        self._mono = monotonic
        self._wall = wall
        self._record = record
        self._interval = interval_s
        self._window = window_s

    async def run(self, rounds: int | None = None) -> None:
        """Samples until cancelled (or for `rounds` samples, in tests)."""
        lags: list[float] = []
        opened = self._mono()
        done = 0
        while rounds is None or done < rounds:
            before = self._mono()
            await self._sleep(self._interval)
            after = self._mono()
            lags.append(max(0.0, after - before - self._interval))
            done += 1
            if after - opened >= self._window:
                self._emit(lags)
                lags, opened = [], after
        if lags:
            self._emit(lags)

    def _emit(self, lags: list[float]) -> None:
        self._record(
            LagWindow(
                self._wall(),
                len(lags),
                round(1000 * sum(lags) / len(lags), 3),
                round(1000 * max(lags), 3),
            )
        )


def collapse(frame: FrameType | None) -> str:
    """One stack as "module:function" names, outermost first, ";"-joined."""
    names: list[str] = []
    while frame is not None:
        code = frame.f_code
        module = frame.f_globals.get("__name__", "?")
        names.append(f"{module}:{code.co_name}")
        frame = frame.f_back
    return ";".join(reversed(names))


class StackSampler:
    """Counts one thread's stacks from a background thread."""

    def __init__(
        self,
        frames: Callable[[], dict[int, FrameType]],
        thread_id: int,
        interval_s: float,
        stop: threading.Event,
    ) -> None:
        """Creates the sampler.

        Args:
            frames: Every thread's current frame (`sys._current_frames`).
            thread_id: The thread to sample.
            interval_s: Seconds between samples.
            stop: Set to end `run`.
        """
        self._frames = frames
        self._thread = thread_id
        self._interval = interval_s
        self._stop = stop
        self.counts: Counter[str] = Counter()

    def sample(self) -> None:
        """Takes one sample."""
        frame = self._frames().get(self._thread)
        if frame is not None:
            self.counts[collapse(frame)] += 1

    def run(self) -> None:
        """Samples until `stop` is set."""
        while not self._stop.wait(self._interval):
            self.sample()

    def folded(self) -> str:
        """The counts in folded-stack form ("stack count" per line)."""
        return "".join(
            f"{stack} {n}\n" for stack, n in self.counts.most_common()
        )


class GcPauses:
    """Times Python's garbage collections (a `gc.callbacks` entry).

    A full collection walks every live object while the event loop waits.
    Each collection of at least `min_ms` becomes one row (time, generation,
    milliseconds, objects collected), so loop-lag spikes can be matched to
    collections.
    """

    def __init__(
        self,
        monotonic: Callable[[], float],
        wall: Callable[[], float],
        record: Callable[[dict[str, Any]], None],
        min_ms: float = 1.0,
    ) -> None:
        """Creates the timer.

        Args:
            monotonic: Seconds, for durations.
            wall: Seconds since the epoch, for each row's time.
            record: Receives each row.
            min_ms: Shorter collections are not recorded.
        """
        self._mono = monotonic
        self._wall = wall
        self._record = record
        self._min_ms = min_ms
        self._started: float | None = None

    def __call__(self, phase: str, info: dict[str, Any]) -> None:
        """The gc callback: "start" and "stop" of each collection."""
        if phase == "start":
            self._started = self._mono()
            return
        if self._started is None:
            return
        ms = 1000 * (self._mono() - self._started)
        self._started = None
        if ms >= self._min_ms:
            self._record(
                {
                    "time": self._wall(),
                    "generation": info.get("generation"),
                    "ms": round(ms, 3),
                    "collected": info.get("collected"),
                }
            )
