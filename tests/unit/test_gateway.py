"""The gateway on fakes: FR-1, FR-6, FR-10 hold, FR-12, NFR-9."""

import asyncio
from pathlib import Path

import pytest

from askiisc.common.errors import ConfigError
from askiisc.common.types import Answer, ChunkId, Citation, DocId, Version
from askiisc.gateway.app import GatewayDeps, create_app, sse
from askiisc.gateway.audit import (
    AuditRecord,
    FileAuditSink,
    salted_hash,
)
from askiisc.gateway.cache import (
    CachedAnswer,
    InMemoryAnswerCache,
    RedisAnswerCache,
    cache_key,
    normalise,
)
from askiisc.gateway.output_filter import filter_answer
from tests.gateway_fixtures import SALT, build, events, final

QUESTION = "What are the hostel fees rules?"


async def test_answered_with_valid_citation() -> None:
    stack = build()
    response = await stack.ask("u_cds_stu", QUESTION)
    assert response.status_code == 200
    kinds = [e for e, _ in events(response)]
    assert kinds[0] == "token" and kinds[-1] == "answer"
    answer = final(response)
    assert not answer["abstained"]
    cited = answer["citations"][0]["chunk_id"]  # type: ignore[index]
    retrieved = [c for c, _ in answer["retrieved"]]  # type: ignore[union-attr]
    assert cited == retrieved[0]
    readable = {"d01", "d02", "d05", "d09", "d12", "d14"}
    assert {d for _, d in answer["retrieved"]} <= readable  # type: ignore[union-attr]
    record = stack.audit.records[-1]
    assert record.decision == "answered"
    assert record.chunk_ids == tuple(retrieved)
    assert {d.doc_id for d in record.documents} <= readable


async def test_unknown_user_abstains_and_no_token_is_401() -> None:
    stack = build()
    r = await stack.ask(None, QUESTION)
    assert r.status_code == 401
    r = await stack.ask("u_not_in_directory", QUESTION)
    assert r.status_code == 403
    assert [rec.decision for rec in stack.audit.records] == [
        "rejected",
        "denied",
    ]


async def test_no_readable_chunks_abstains() -> None:
    stack = build()
    stack.w.policies.delete(DocId("d13"))
    r = await stack.ask("u_guest", QUESTION)
    answer = final(r)
    assert answer["abstained"] and answer["citations"] == []
    assert stack.audit.records[-1].decision == "abstained"


async def test_rate_limit_returns_429_with_retry_after() -> None:
    stack = build(rate=1.0, burst=2.0)
    codes = [
        (await stack.ask("u_cds_stu", QUESTION)).status_code for _ in range(3)
    ]
    assert codes == [200, 200, 429]
    r = await stack.ask("u_cds_stu", QUESTION)
    assert r.headers["Retry-After"] == "1"
    assert stack.audit.records[-1].decision == "rejected"
    # Another tenant is unaffected.
    assert (await stack.ask("u_ece_stu", QUESTION)).status_code == 200


async def test_question_too_long_is_413() -> None:
    stack = build()
    r = await stack.ask("u_cds_stu", "word " * 600)
    assert r.status_code == 413


async def test_cache_hit_and_isolation_between_permission_sets() -> None:
    stack = build()
    first = final(await stack.ask("u_cds_stu", QUESTION))
    again = final(
        await stack.ask("u_cds_stu", "  what are the HOSTEL fees rules? ")
    )
    assert not first["cache_hit"] and again["cache_hit"]
    assert again["citations"] == first["citations"]
    assert again["retrieved"] == first["retrieved"]
    other = final(await stack.ask("u_ece_stu", QUESTION))
    assert not other["cache_hit"]
    assert stack.audit.records[1].cache_hit
    assert stack.audit.records[1].chunk_ids == stack.audit.records[0].chunk_ids


async def test_policy_or_corpus_change_invalidates_cache() -> None:
    stack = build()
    await stack.ask("u_cds_stu", QUESTION)
    stack.w.policies.delete(DocId("d15"))  # any policy change
    assert not final(await stack.ask("u_cds_stu", QUESTION))["cache_hit"]
    stack.generation[0] += 1  # a new document version
    assert not final(await stack.ask("u_cds_stu", QUESTION))["cache_hit"]
    assert final(await stack.ask("u_cds_stu", QUESTION))["cache_hit"]


async def test_audit_never_stores_raw_query(tmp_path: Path) -> None:
    stack = build()
    sink = FileAuditSink(tmp_path / "audit.jsonl")
    stack.deps.audit = sink
    secret = "Is the director's salary 12345 rupees?"
    await stack.ask("u_cds_stu", secret)
    await stack.ask("u_banned", secret)
    raw = (tmp_path / "audit.jsonl").read_text()
    assert "salary" not in raw and "12345" not in raw
    assert "u_cds_stu" not in raw and "u_banned" not in raw
    records = sink.read()
    assert records[0].query_hash == salted_hash(SALT, normalise(secret))
    assert records[0].query_hash == records[1].query_hash
    assert records[0].user_hash != records[1].user_hash
    assert FileAuditSink(tmp_path / "missing.jsonl").read() == []


async def test_audit_record_reproduces_answer() -> None:
    """FR-12: replaying with the record's as-of time and versions gives the
    same chunks."""
    stack = build()
    await stack.ask("u_phd", QUESTION)
    record = stack.audit.records[-1]
    replay = await stack.ask("u_phd", QUESTION, as_of=record.as_of)
    replayed = [c for c, _ in final(replay)["retrieved"]]  # type: ignore[union-attr]
    assert tuple(replayed) == record.chunk_ids
    assert record.policy_version == stack.w.policies.version


class _ColdWorkers:
    def __init__(self, becomes_ready: bool) -> None:
        self._ready = False
        self._becomes = becomes_ready
        self.requests = 0

    def ready(self) -> bool:
        return self._ready

    async def wait_ready(self, timeout_s: float) -> bool:
        await asyncio.sleep(0)
        self._ready = self._becomes
        return self._becomes

    def note_request(self) -> None:
        self.requests += 1


async def test_cold_start_holds_request_with_warming_event() -> None:
    workers = _ColdWorkers(becomes_ready=True)
    stack = build(workers=workers)
    r = await stack.ask("u_cds_stu", QUESTION)
    kinds = [e for e, _ in events(r)]
    assert kinds[0] == "warming" and kinds[-1] == "answer"
    assert workers.requests == 1
    health = (await stack.client.get("/healthz")).json()
    assert health["workers_ready"] is True


async def test_cold_start_timeout_is_an_error_event() -> None:
    stack = build(workers=_ColdWorkers(becomes_ready=False))
    r = await stack.ask("u_cds_stu", QUESTION)
    assert [e for e, _ in events(r)] == ["warming", "error"]
    assert stack.audit.records[-1].decision == "error"


async def test_access_service_down_denies() -> None:
    from askiisc.common.errors import AccessDenied

    stack = build()

    class _Down:
        async def filter(self, token: str, at: float | None = None) -> None:
            raise AccessDenied("down")

        async def authorize(self, *a: object) -> bool:
            raise AccessDenied("down")

    stack.deps.access = _Down()  # type: ignore[assignment]
    r = await stack.ask("u_cds_stu", QUESTION)
    assert r.status_code == 403
    assert stack.audit.records[-1].decision == "denied"


async def test_revocation_during_generation_is_withheld() -> None:
    stack = build()
    original = stack.deps.rag.finish

    def finish_then_revoke(prepared):  # type: ignore[no-untyped-def]
        answer = original(prepared)
        for doc in {c.doc_id for c in answer.citations}:
            stack.w.policies.delete(doc)
        return answer

    stack.deps.rag.finish = finish_then_revoke  # type: ignore[method-assign]
    answer = final(await stack.ask("u_cds_stu", QUESTION))
    assert answer["abstained"] and answer["citations"] == []
    stack.deps.rag.finish = original  # type: ignore[method-assign]
    assert not final(await stack.ask("u_cds_stu", QUESTION))["cache_hit"]


async def test_mid_stream_access_failure_is_error_event() -> None:
    from askiisc.common.errors import AccessDenied

    stack = build()

    async def broken(*args: object, **kwargs: object) -> None:
        raise AccessDenied("down mid-request")

    stack.deps.rag.prepare = broken  # type: ignore[method-assign]
    r = await stack.ask("u_cds_stu", QUESTION)
    assert [e for e, _ in events(r)] == ["error"]
    assert stack.audit.records[-1].decision == "denied"


async def test_metrics_endpoint_has_no_query_text() -> None:
    stack = build()
    await stack.ask("u_cds_stu", "unique-marker-question about fees")
    text = (await stack.client.get("/metrics")).text
    assert "askiisc_requests_total" in text
    assert "unique-marker" not in text


def test_gateway_requires_salt() -> None:
    stack = build()
    deps = GatewayDeps(**{**stack.deps.__dict__, "audit_salt": ""})
    with pytest.raises(ConfigError):
        create_app(deps)
    with pytest.raises(ValueError):
        salted_hash("", "x")


def test_sse_format() -> None:
    assert (
        sse("token", {"text": "hi"})
        == b'event: token\ndata: {"text": "hi"}\n\n'
    )


def test_cache_key_has_no_question_text_and_depends_on_everything() -> None:
    base = cache_key(["d1", "d2"], 3, 7, "Hostel fees?")
    assert "Hostel" not in base and "hostel" not in base
    assert base == cache_key(["d2", "d1"], 3, 7, "  hostel   FEES? ")
    assert base != cache_key(["d1"], 3, 7, "Hostel fees?")
    assert base != cache_key(["d1", "d2"], 4, 7, "Hostel fees?")
    assert base != cache_key(["d1", "d2"], 3, 8, "Hostel fees?")


async def test_in_memory_cache_expires() -> None:
    from askiisc.common.clock import FakeClock

    clock = FakeClock()
    cache = InMemoryAnswerCache(clock, ttl_s=10)
    value = CachedAnswer(answer=Answer(text="x"), documents=(), chunk_ids=())
    await cache.set("k", value)
    assert await cache.get("k") == value
    clock.advance(10)
    assert await cache.get("k") is None


class _FakeRedis:
    def __init__(self, fail: bool = False) -> None:
        self.data: dict[str, str] = {}
        self.fail = fail

    async def get(self, key: str) -> str | None:
        from redis.exceptions import ConnectionError as RedisConnectionError

        if self.fail:
            raise RedisConnectionError("down")
        return self.data.get(key)

    async def set(self, key: str, value: str, ex: int) -> None:
        from redis.exceptions import ConnectionError as RedisConnectionError

        if self.fail:
            raise RedisConnectionError("down")
        self.data[key] = value


async def test_redis_cache_round_trip_and_failure_is_miss() -> None:
    value = CachedAnswer(
        answer=Answer(text="x"), documents=(("d", 1),), chunk_ids=("d@v1#0",)
    )
    cache = RedisAnswerCache(_FakeRedis())
    await cache.set("k", value)
    assert await cache.get("k") == value
    assert await cache.get("missing") is None
    broken = RedisAnswerCache(_FakeRedis(fail=True))
    await broken.set("k", value)
    assert await broken.get("k") is None


async def test_output_filter_passes_allowed_citations() -> None:
    stack = build()
    citation = Citation(
        doc_id=DocId("d01"),
        version=Version(1),
        chunk_id=ChunkId("d01@v1#0"),
        page=1,
    )
    answer = Answer(text="ok [chunk:d01@v1#0]", citations=(citation,))
    token = stack.w.token("u_cds_stu")
    result = await filter_answer(stack.deps.access, token, answer)
    assert result.answer == answer and not result.withheld
    result = await filter_answer(stack.deps.access, "bad", answer)
    assert result.withheld


def test_audit_record_has_no_text_fields() -> None:
    fields = set(AuditRecord.model_fields)
    assert not fields & {"query", "question", "answer", "text", "user_id"}
