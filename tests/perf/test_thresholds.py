"""Performance thresholds as tests on the fake stack (Phase 11).

- FR-6: at 3x load the excess gets HTTP 429 within 50 ms (p95);
- NFR-10: access authorize p99 <= 10 ms and filter p99 <= 20 ms at 500
  decisions per second over HTTP;
- NFR-4: search p95 <= 150 ms on a 40k-chunk synthetic corpus.

Run inside Slurm: sbatch --wait slurm/gate.sh test-perf (or slurm/nightly.sh).
"""

import asyncio
import math
import random
import time

import httpx
import numpy as np
import pytest
from eval.probes.real import pct

from askiisc.access.directory import Directory
from askiisc.access.policy import DocPolicy, PolicyStore
from askiisc.access.service import create_app
from askiisc.common.clock import SystemClock
from askiisc.common.targets import (
    EMBEDDING_DIM,
    FR6_OVERLOAD_FACTOR,
    FR6_REJECT_WITHIN_MS,
    NFR4_SEARCH_P95_S,
    NFR10_AUTHORIZE_MS,
    NFR10_DECISIONS_PER_SECOND,
    NFR10_FILTER_MS,
    RETRIEVAL_TOP_K,
)
from askiisc.common.tokens import TokenIssuer
from askiisc.common.types import Chunk, ChunkId, DocId, Version
from askiisc.retrieval.dense import DenseIndex
from tests.gateway_fixtures import build

pytestmark = pytest.mark.perf
QUESTION = "What are the hostel fee rules?"
SEARCH_CHUNKS = 40_000


async def test_fr6_excess_requests_get_429_fast() -> None:
    capacity = 8
    stack = build(rate=0.001, burst=float(capacity))
    load = int(capacity * FR6_OVERLOAD_FACTOR)

    async def one() -> tuple[int, float]:
        started = time.perf_counter()
        r = await stack.ask("u_cds_stu", QUESTION)
        return r.status_code, (time.perf_counter() - started) * 1e3

    results = await asyncio.gather(*(one() for _ in range(load)))
    rejected = [ms for code, ms in results if code == 429]  # noqa: PLR2004
    assert len(rejected) == load - capacity
    assert pct(rejected, 95) <= FR6_REJECT_WITHIN_MS


def _access_world() -> tuple[Directory, PolicyStore, list[str], list[str]]:
    rng = random.Random(10)
    depts = ["cds", "csa", "ece", "accounts"]
    people = []
    for i in range(400):
        dept = depts[i % 4]
        role = ["student", "faculty", "staff"][i % 3]
        people.append(
            {"user_id": f"p{i}", "department": dept, "role": role,
             "groups": [f"{dept}-{role}s"]}
        )  # fmt: skip
    directory = Directory.from_records(people)
    policies = [
        DocPolicy(doc_id=DocId(f"pub{i}"), visibility="public")
        for i in range(20_000)
    ] + [
        DocPolicy(
            doc_id=DocId(f"res{i}"),
            owner=depts[i % 4],
            visibility="restricted",
            groups=[f"{rng.choice(depts)}-faculty"],
        )
        for i in range(200)
    ]  # fmt: skip
    docs = [str(p.doc_id) for p in policies]
    return (
        directory,
        PolicyStore(policies),
        [p["user_id"] for p in people],
        docs,
    )


async def _fixed_rate(
    app: object,
    issuer: TokenIssuer,
    users: list[str],
    docs: list[str],
    path: str,
) -> list[float]:
    """Calls the service at NFR10_DECISIONS_PER_SECOND for 5 s (ms each)."""
    tokens = {u: issuer.issue(u, time.time() + 3600) for u in users}
    rng = random.Random(4)
    lat: list[float] = []
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://a") as c:

        async def one() -> None:
            user = rng.choice(users)
            body = (
                {"doc_id": rng.choice(docs)}
                if path == "/v1/authorize"
                else {"compact": True}
            )
            started = time.perf_counter()
            r = await c.post(
                path,
                json=body,
                headers={"Authorization": f"Bearer {tokens[user]}"},
            )
            assert r.status_code == 200  # noqa: PLR2004
            lat.append((time.perf_counter() - started) * 1e3)

        tasks, start = [], time.monotonic()
        for i in range(5 * NFR10_DECISIONS_PER_SECOND):
            tasks.append(asyncio.create_task(one()))
            delay = start + (i + 1) / NFR10_DECISIONS_PER_SECOND
            await asyncio.sleep(max(0.0, delay - time.monotonic()))
        await asyncio.gather(*tasks)
    return lat


@pytest.mark.parametrize(
    ("path", "limit_ms"),
    [("/v1/authorize", NFR10_AUTHORIZE_MS), ("/v1/filter", NFR10_FILTER_MS)],
)
async def test_nfr10_access_latency_at_500_per_second(
    path: str, limit_ms: float
) -> None:
    """Service time per decision (compact filter, as the gateway asks).

    In process, so no TCP: the live HTTP figure comes from the access
    service's own histogram in a serving job.
    """
    directory, policies, users, docs = _access_world()
    clock, issuer = SystemClock(), TokenIssuer()
    app = create_app(directory, policies, issuer.verifier(clock), clock)
    lat = await _fixed_rate(app, issuer, users, docs, path)
    assert pct(lat, 99) <= limit_ms, (pct(lat, 50), pct(lat, 99))


def test_nfr4_search_on_40k_chunks() -> None:
    rng = np.random.default_rng(4)
    vectors = rng.standard_normal((SEARCH_CHUNKS, EMBEDDING_DIM), np.float32)
    index = DenseIndex(dim=EMBEDDING_DIM)
    words = ["hostel", "fee", "library", "exam", "credit", "thesis", "mess"]
    index.upsert(
        [
            Chunk(
                id=ChunkId(f"d{i // 20}@v1#{i % 20}"),
                doc_id=DocId(f"d{i // 20}"),
                version=Version(1),
                text=" ".join(words[(i + k) % 7] for k in range(30)),
                page=1,
                valid_from=0.0,
                embedding=tuple(vectors[i].tolist()),
            )
            for i in range(SEARCH_CHUNKS)
        ]
    )
    allowed = frozenset(
        DocId(f"d{i}") for i in range(0, SEARCH_CHUNKS // 20, 2)
    )
    times = []
    for _ in range(200):
        q = rng.standard_normal(EMBEDDING_DIM).tolist()
        started = time.perf_counter()
        hits = index.search(q, allowed, time.time(), RETRIEVAL_TOP_K)
        times.append(time.perf_counter() - started)
        assert len(hits) == RETRIEVAL_TOP_K
    times.sort()
    assert times[math.ceil(0.95 * len(times)) - 1] <= NFR4_SEARCH_P95_S
