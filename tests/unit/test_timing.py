"""Serving-node timing: responses, loop lag and stack samples (FR-6)."""

import json
import sys
import threading
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from askiisc.observability.timing import (
    JsonLines,
    LagWindow,
    LoopLag,
    ResponseTimer,
    ResponseTiming,
    StackSampler,
    ask_kind,
    collapse,
    timing_row,
)


class Ticks:
    """A monotonic clock that moves 0.010 s per reading."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        self.t += 0.010
        return self.t


def app_with_timer(rows: list[ResponseTiming]) -> Any:
    app = FastAPI()

    @app.post("/api/temporary")
    async def refuse() -> JSONResponse:
        return JSONResponse(
            {"error": "busy"}, status_code=429, headers={"X-Request-Id": "r1"}
        )

    @app.post("/api/chats/{chat}/messages")
    async def stream(chat: str) -> StreamingResponse:
        async def body() -> Any:
            yield b"event: token\n\n"
            yield b"event: answer\n\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    @app.post("/api/other")
    async def other() -> dict[str, str]:
        return {"ok": "yes"}

    return ResponseTimer(app, rows.append, Ticks(), lambda: 1000.0)


async def test_question_routes_are_timed_and_others_are_not() -> None:
    rows: list[ResponseTiming] = []
    transport = httpx.ASGITransport(app=app_with_timer(rows))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://t"
    ) as client:
        assert (await client.post("/api/temporary")).status_code == 429  # noqa: PLR2004
        await client.post("/api/chats/c1/messages")
        await client.post("/api/other")
        await client.get("/api/temporary")
    assert [(r.kind, r.status, r.request_id) for r in rows] == [
        ("temporary", 429, "r1"),
        ("chat", 200, None),
    ]
    refused = rows[0]
    assert refused.time == 1000.0  # noqa: PLR2004
    assert 0 < refused.start_ms <= refused.done_ms
    streamed = rows[1]
    assert streamed.done_ms > streamed.start_ms


def test_ask_kind_names_only_question_routes() -> None:
    assert ask_kind("/v1/ask") == "api"
    assert ask_kind("/api/temporary") == "temporary"
    assert ask_kind("/api/chats/abc/messages") == "chat"
    assert ask_kind("/api/chats/abc") is None
    assert ask_kind("/api/admin/overview") is None


def test_rows_are_written_as_json_lines(tmp_path: Path) -> None:
    sink = JsonLines(tmp_path / "responses.jsonl")
    sink.write(timing_row(ResponseTiming(1.0, "chat", 429, "r", 0.1234, 2.0)))
    sink.close()
    row = json.loads((tmp_path / "responses.jsonl").read_text())
    assert row == {
        "time": 1.0,
        "kind": "chat",
        "status": 429,
        "request_id": "r",
        "start_ms": 0.12,
        "done_ms": 2.0,
    }


async def test_loop_lag_reports_late_wakeups_per_window() -> None:
    now = [0.0]
    late = [0.0, 0.005, 0.0, 0.020]

    async def sleep(seconds: float) -> None:
        now[0] += seconds + late.pop(0)

    windows: list[LagWindow] = []
    lag = LoopLag(
        sleep,
        lambda: now[0],
        lambda: 7.0,
        windows.append,
        interval_s=0.01,
        window_s=0.025,
    )
    await lag.run(rounds=4)
    assert [(w.samples, w.max_ms) for w in windows] == [(2, 5.0), (2, 20.0)]
    assert windows[0].mean_ms == 2.5  # noqa: PLR2004
    assert windows[0].time == 7.0  # noqa: PLR2004


def leaf() -> str:
    return collapse(sys._getframe())


def middle() -> str:
    return leaf()


def test_collapse_names_the_stack_outermost_first() -> None:
    stack = collapse(sys._getframe()).split(";")
    assert (
        stack[-1] == f"{__name__}:test_collapse_names_the_stack_outermost_first"
    )
    names = middle().split(";")
    assert names[-2:] == [f"{__name__}:middle", f"{__name__}:leaf"]
    assert collapse(None) == ""


def test_sampler_counts_a_threads_stacks() -> None:
    stop = threading.Event()
    me = threading.get_ident()
    sampler = StackSampler(sys._current_frames, me, 0.001, stop)
    sampler.sample()
    sampler.sample()
    sampler.sample = stop.set  # type: ignore[method-assign]
    sampler.run()  # returns once a sample sets stop
    ((stack, count),) = sampler.counts.items()
    assert count == 2  # noqa: PLR2004
    assert "test_sampler_counts_a_threads_stacks" in stack
    assert sampler.folded() == f"{stack} 2\n"
    missing = StackSampler(dict, me, 0.001, stop)
    missing.sample()
    assert not missing.counts


def test_rows_are_readable_before_close(tmp_path: Path) -> None:
    # Regression (run 2123): the FR-6 report read responses.jsonl while the
    # stack was up and saw only the rows already flushed.
    sink = JsonLines(tmp_path / "responses.jsonl")
    for n in range(3):
        sink.write({"n": n})
    lines = (tmp_path / "responses.jsonl").read_text().splitlines()
    assert [json.loads(x)["n"] for x in lines] == [0, 1, 2]
    sink.close()


def test_gc_pauses_record_long_collections_only() -> None:
    from askiisc.observability.timing import GcPauses

    t = [0.0]
    rows: list[dict[str, object]] = []
    pauses = GcPauses(lambda: t[0], lambda: 9.0, rows.append, min_ms=1.0)
    pauses("stop", {})  # a stop without a start is ignored
    pauses("start", {"generation": 0})
    t[0] += 0.0005
    pauses("stop", {"generation": 0, "collected": 1})
    pauses("start", {"generation": 2})
    t[0] += 0.25
    pauses("stop", {"generation": 2, "collected": 7})
    assert rows == [{"time": 9.0, "generation": 2, "ms": 250.0, "collected": 7}]
