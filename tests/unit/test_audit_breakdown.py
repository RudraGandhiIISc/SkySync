"""FR-12: every audit record has workers, cold start and a latency breakdown.

A fake monotonic clock advances only inside the faked stages, so each
stage's milliseconds are known exactly and `other_ms` must be 0.
"""

import asyncio
import typing
from collections.abc import AsyncIterator
from typing import Any

import pytest

from askiisc.common.errors import AccessDenied
from askiisc.gateway.audit import STAGES, AuditRecord, LatencyBreakdown
from tests import corpus
from tests.gateway_fixtures import Stack, build, final

QUESTION = "What are the hostel fees rules?"


class Ticks:
    """Monotonic seconds that move only when a fake stage says so."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _slow(ticks: Ticks, seconds: float, fn: Any) -> Any:
    """Wraps a sync or async callable so it takes `seconds` on `ticks`."""
    if asyncio.iscoroutinefunction(fn):

        async def wrapped_async(*a: Any, **kw: Any) -> Any:
            ticks.now += seconds
            return await fn(*a, **kw)

        return wrapped_async

    def wrapped(*a: Any, **kw: Any) -> Any:
        ticks.now += seconds
        return fn(*a, **kw)

    return wrapped


def _timed_stack(
    monkeypatch: pytest.MonkeyPatch, **kw: Any
) -> tuple[Stack, Ticks]:
    stack = build(**kw)
    ticks = Ticks()
    deps = stack.deps
    deps.monotonic = ticks
    auth = deps.auth
    monkeypatch.setattr(
        auth, "authenticate", _slow(ticks, 0.002, auth.authenticate)
    )
    access = deps.access
    monkeypatch.setattr(access, "filter", _slow(ticks, 0.003, access.filter))
    monkeypatch.setattr(
        access, "authorize", _slow(ticks, 0.001, access.authorize)
    )
    embedder = corpus.EMBEDDER
    monkeypatch.setattr(embedder, "embed", _slow(ticks, 0.005, embedder.embed))
    index = deps.rag._retriever._index  # noqa: SLF001
    monkeypatch.setattr(index, "search", _slow(ticks, 0.007, index.search))
    original = deps.rag.generate

    async def generate(*a: Any, **k: Any) -> AsyncIterator[str]:
        ticks.now += 0.011  # prefill until the first token
        first = True
        async for piece in original(*a, **k):
            if not first:
                ticks.now += 0.001
            first = False
            yield piece

    monkeypatch.setattr(deps.rag, "generate", generate)
    return stack, ticks


def _latency(record: AuditRecord) -> dict[str, float]:
    return record.latency.model_dump()


async def test_success_breakdown_attributes_every_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack, _ = _timed_stack(monkeypatch)
    response = await stack.ask("u_cds_stu", QUESTION)
    assert response.status_code == 200
    answer = final(response)
    record = stack.audit.records[-1]
    assert record.decision == "answered"
    assert record.active_workers == 1 and not record.cold_start
    ms = _latency(record)
    assert ms["auth_ms"] == pytest.approx(2.0)
    assert ms["access_ms"] == pytest.approx(6.0)  # gateway + retriever
    assert ms["embed_ms"] == pytest.approx(5.0)
    assert ms["search_ms"] == pytest.approx(7.0)
    assert ms["first_token_ms"] == pytest.approx(11.0)
    assert ms["generation_ms"] >= 0.0
    cited_docs = {c["doc_id"] for c in answer["citations"]}  # type: ignore[union-attr]
    assert ms["filter_ms"] == pytest.approx(1.0 * len(cited_docs))
    assert ms["cold_hold_ms"] == 0.0 and ms["queue_ms"] == pytest.approx(0.0)
    assert ms["other_ms"] == pytest.approx(0.0, abs=1e-6)
    assert record.latency.sum_ms() == pytest.approx(record.latency_ms)


async def test_cache_hit_has_no_retrieval_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack, _ = _timed_stack(monkeypatch)
    await stack.ask("u_cds_stu", QUESTION)
    await stack.ask("u_cds_stu", QUESTION)
    record = stack.audit.records[-1]
    assert record.cache_hit
    ms = _latency(record)
    assert ms["embed_ms"] == ms["search_ms"] == ms["first_token_ms"] == 0.0
    assert ms["auth_ms"] == pytest.approx(2.0)
    assert ms["access_ms"] == pytest.approx(3.0)
    assert record.latency.sum_ms() == pytest.approx(record.latency_ms)


async def test_401_and_413_records(monkeypatch: pytest.MonkeyPatch) -> None:
    stack, _ = _timed_stack(monkeypatch)
    r = await stack.ask(None, QUESTION)
    assert r.status_code == 401
    r = await stack.ask("u_cds_stu", "word " * 600)
    assert r.status_code == 413
    for record in stack.audit.records:
        assert record.decision == "rejected"
        assert record.active_workers == 1 and not record.cold_start
        assert record.latency.auth_ms == pytest.approx(2.0)
        assert record.latency.sum_ms() == pytest.approx(record.latency_ms)


async def test_429_record(monkeypatch: pytest.MonkeyPatch) -> None:
    stack, _ = _timed_stack(monkeypatch, rate=1.0, burst=1.0)
    await stack.ask("u_cds_stu", QUESTION)
    r = await stack.ask("u_cds_stu", "Another question about fees?")
    assert r.status_code == 429
    record = stack.audit.records[-1]
    assert record.decision == "rejected" and record.active_workers == 1
    assert record.latency.auth_ms == pytest.approx(2.0)
    assert record.latency.access_ms == 0.0  # shed before the access check
    assert record.latency.sum_ms() == pytest.approx(record.latency_ms)


async def test_403_access_down_record(monkeypatch: pytest.MonkeyPatch) -> None:
    stack, ticks = _timed_stack(monkeypatch)

    async def down(token: str, at: float | None = None) -> None:
        ticks.now += 0.004
        raise AccessDenied("down")

    monkeypatch.setattr(stack.deps.access, "filter", down)
    r = await stack.ask("u_cds_stu", QUESTION)
    assert r.status_code == 403
    record = stack.audit.records[-1]
    assert record.decision == "denied"
    assert record.latency.access_ms == pytest.approx(4.0)
    assert record.latency.sum_ms() == pytest.approx(record.latency_ms)


class _Cold:
    """No worker until wait_ready, which takes 30 ms on the fake clock."""

    def __init__(self, ticks: Ticks) -> None:
        self._ready = False
        self._ticks = ticks

    def ready(self) -> bool:
        return self._ready

    async def wait_ready(self, timeout_s: float) -> bool:
        self._ticks.now += 0.030
        self._ready = True
        return True

    def note_request(self) -> None:
        pass


async def test_cold_start_record(monkeypatch: pytest.MonkeyPatch) -> None:
    stack, ticks = _timed_stack(monkeypatch)
    stack.deps.workers = _Cold(ticks)
    r = await stack.ask("u_cds_stu", QUESTION)
    assert r.status_code == 200
    record = stack.audit.records[-1]
    assert record.cold_start and record.active_workers == 0
    assert record.latency.cold_hold_ms == pytest.approx(30.0)
    assert record.latency.other_ms == pytest.approx(0.0, abs=1e-6)


async def test_active_workers_comes_from_the_injected_count() -> None:
    stack = build()
    stack.deps.ready_workers = lambda: 2
    await stack.ask("u_cds_stu", QUESTION)
    assert stack.audit.records[-1].active_workers == 2  # noqa: PLR2004


async def test_every_response_names_its_audit_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stack, _ = _timed_stack(monkeypatch, rate=1.0, burst=2.0)
    responses = [
        await stack.ask("u_cds_stu", QUESTION),  # 200
        await stack.ask(None, QUESTION),  # 401
        await stack.ask("u_cds_stu", "word " * 600),  # 413
        await stack.ask("u_cds_stu", "Second question on fees?"),  # 200
        await stack.ask("u_cds_stu", "Third question on fees?"),  # 429
    ]
    assert [r.status_code for r in responses] == [200, 401, 413, 200, 429]
    ids = [r.headers["X-Request-Id"] for r in responses]
    assert ids == [rec.request_id for rec in stack.audit.records]
    assert len(set(ids)) == len(ids)


def test_breakdown_other_is_the_remainder_and_never_negative() -> None:
    b = LatencyBreakdown.from_stages({"embed": 0.010, "search": 0.020}, 50.0)
    assert b.other_ms == pytest.approx(20.0)
    assert b.sum_ms() == pytest.approx(50.0)
    over = LatencyBreakdown.from_stages({"embed": 0.100}, 50.0)
    assert over.other_ms == 0.0


def test_breakdown_refuses_unknown_stages() -> None:
    with pytest.raises(ValueError, match="unknown"):
        LatencyBreakdown.from_stages({"question": 0.1}, 1.0)


def test_breakdown_has_every_stage_and_only_numbers() -> None:
    fields = LatencyBreakdown.model_fields
    assert set(fields) == {f"{s}_ms" for s in STAGES} | {"other_ms"}
    hints = typing.get_type_hints(LatencyBreakdown)
    assert all(hints[name] is float for name in fields)
    record_hints = typing.get_type_hints(AuditRecord)
    assert record_hints["active_workers"] is int
    assert record_hints["cold_start"] is bool


async def test_start_prompt_finish_and_deliver_are_timed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Driven without HTTP, so the test decides when the stream starts and
    # how long the consumer takes after the final answer event.
    from askiisc.gateway.app import AskRequest, ask

    stack, ticks = _timed_stack(monkeypatch)
    import askiisc.llm.rag as rag_module

    monkeypatch.setattr(
        rag_module,
        "build_messages",
        _slow(ticks, 0.003, rag_module.build_messages),
    )
    rag = stack.deps.rag
    monkeypatch.setattr(rag, "finish", _slow(ticks, 0.002, rag.finish))
    header = f"Bearer {stack.w.token('u_cds_stu')}"
    response = await ask(stack.deps, AskRequest(question=QUESTION), header)
    ticks.now += 0.040  # the event loop gets to the stream late
    async for chunk in response.body_iterator:  # type: ignore[attr-defined]
        if bytes(chunk).startswith(b"event: answer"):
            ticks.now += 0.025  # the chat app saves the answer
    record = stack.audit.records[-1]
    assert record.decision == "answered"
    ms = _latency(record)
    assert ms["start_ms"] == pytest.approx(40.0)
    assert ms["prompt_ms"] == pytest.approx(3.0)
    assert ms["finish_ms"] == pytest.approx(2.0)
    assert ms["deliver_ms"] == pytest.approx(25.0)
    assert ms["other_ms"] == pytest.approx(0.0, abs=1e-6)


async def test_refusals_name_their_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from askiisc.common.errors import AccessTimeout

    stack, _ = _timed_stack(monkeypatch, rate=1.0, burst=1.0)
    await stack.ask(None, QUESTION)
    await stack.ask("u_not_in_directory", QUESTION)
    await stack.ask("u_cds_stu", "word " * 600)
    await stack.ask("u_cds_stu", QUESTION)  # answered; empties the bucket
    await stack.ask("u_cds_stu", "Mess fees?")
    reasons = [(r.decision, r.reason) for r in stack.audit.records]
    assert reasons == [
        ("rejected", "invalid_token"),
        ("denied", "not_in_directory"),
        ("rejected", "question_too_long"),
        ("answered", None),
        ("rejected", "rate_limited"),
    ]

    async def slow(token: str, at: float | None = None) -> None:
        raise AccessTimeout("timed out")

    async def broken(token: str, at: float | None = None) -> None:
        raise AccessDenied("500")

    for fake, expected in (
        (slow, "access_check_timeout"),
        (broken, "access_check_failed"),
    ):
        timed = build()
        monkeypatch.setattr(timed.deps.access, "filter", fake)
        assert (await timed.ask("u_cds_stu", QUESTION)).status_code == 403  # noqa: PLR2004
        assert timed.audit.records[-1].reason == expected


async def test_mid_stream_access_timeout_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from askiisc.common.errors import AccessTimeout

    stack = build()

    async def timeout(*a: object, **k: object) -> None:
        raise AccessTimeout("slow")

    monkeypatch.setattr(stack.deps.rag, "prepare", timeout)
    await stack.ask("u_cds_stu", QUESTION)
    record = stack.audit.records[-1]
    assert (record.decision, record.reason) == (
        "denied",
        "access_check_timeout",
    )


async def test_record_refusal_writes_a_full_record() -> None:
    from askiisc.common.errors import AccessTimeout
    from askiisc.gateway.app import Origin, record_refusal

    stack = build()
    ticks = Ticks()
    stack.deps.monotonic = ticks
    header = f"Bearer {stack.w.token('u_cds_stu')}"
    rid = record_refusal(
        stack.deps, header, QUESTION, Origin("chat"), AccessTimeout("x"), 0.3
    )
    record = stack.audit.records[-1]
    assert record.request_id == rid
    assert (record.decision, record.reason) == (
        "denied",
        "access_check_timeout",
    )
    assert record.tenant != "unknown" and record.active_workers == 1
    assert record.latency.access_ms == pytest.approx(300.0)
    assert record.latency_ms == pytest.approx(300.0)
    anonymous = record_refusal(
        stack.deps,
        "Bearer bad",
        QUESTION,
        Origin("chat"),
        AccessDenied("x"),
        0.0,
    )
    last = stack.audit.records[-1]
    assert last.request_id == anonymous and last.reason == "access_check_failed"
    assert last.tenant == "unknown"
