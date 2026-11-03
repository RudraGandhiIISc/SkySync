"""Access checks on the real index and policies (FR-2, FR-3, NFR-10).

1. probes: N searches of the real Milvus index (539k chunks), each as a
   random user from the users tree, with the permission filter from the
   policy engine inside the search; the oracle (independent code) replays
   every result. Query vectors are random unit vectors: leaks do not depend
   on what is asked.
2. latency: the access service over HTTP at a fixed rate (500 decisions per
   second for authorize, then for filter), recording p50/p95/p99.

Usage: python -m eval.probes.real --state DIR --corpus DIR --users DIR --out DIR
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from pathlib import Path

import httpx
from tests.integration.conftest import ThreadServer

from askiisc.access.directory import Directory
from askiisc.access.policy import PolicyEngine, PolicyStore
from askiisc.access.service import create_app
from askiisc.common.clock import SystemClock
from askiisc.common.tokens import TokenIssuer
from askiisc.common.types import UserId
from askiisc.ingest.corpus import load_corpus
from askiisc.retrieval.store import MilvusIndex
from eval.oracle.oracle import LogEntry, check_run


def pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile."""
    s = sorted(values)
    return s[min(len(s) - 1, math.ceil(q / 100 * len(s)) - 1)]


def probes(
    engine: PolicyEngine,
    index: MilvusIndex,
    users: list[str],
    directory_records: list[dict[str, object]],
    policy_records: list[dict[str, object]],
    n: int,
    rng: random.Random,
) -> dict[str, object]:
    """Runs N filtered searches and replays them through the oracle."""
    now = time.time()
    log, search_ms, filter_ms = [], [], []
    for _ in range(n):
        user = rng.choice(users)
        t0 = time.perf_counter()
        allowed = engine.filter(UserId(user), now).doc_ids
        filter_ms.append((time.perf_counter() - t0) * 1e3)
        vec = [rng.gauss(0, 1) for _ in range(768)]
        norm = math.sqrt(sum(v * v for v in vec))
        t0 = time.perf_counter()
        hits = index.search([v / norm for v in vec], allowed, now, 8)
        search_ms.append((time.perf_counter() - t0) * 1e3)
        log.append(
            LogEntry(
                user, now, tuple((h.chunk.id, h.chunk.doc_id) for h in hits)
            )
        )
    violations = check_run(log, directory_records, policy_records)
    return {
        "probes": n,
        "retrieved_chunks": sum(len(e.retrieved) for e in log),
        "false_allows": sum(v.kind == "false_allow" for v in violations),
        "violations_sample": [v.__dict__ for v in violations[:5]],
        "engine_filter_ms_p50": pct(filter_ms, 50),
        "engine_filter_ms_p99": pct(filter_ms, 99),
        "search_ms_p50": pct(search_ms, 50),
        "search_ms_p95": pct(search_ms, 95),
        "search_ms_p99": pct(search_ms, 99),
    }


async def http_latency(
    app: object,
    issuer: TokenIssuer,
    users: list[str],
    docs: list[str],
    rate: float,
    seconds: float,
    path: str,
) -> dict[str, object]:
    """Fixed-rate calls to the access service; client-side latency."""
    server = ThreadServer(app).start()
    tokens = {u: issuer.issue(u, time.time() + 3600) for u in users}
    rng = random.Random(4)
    lat: list[float] = []
    errors = 0
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=64)) as c:

        async def one() -> None:
            nonlocal errors
            user = rng.choice(users)
            body = (
                {"doc_id": rng.choice(docs)} if path == "/v1/authorize" else {}
            )
            t0 = time.perf_counter()
            try:
                r = await c.post(
                    server.url + path,
                    json=body,
                    timeout=5,
                    headers={"Authorization": f"Bearer {tokens[user]}"},
                )
                if r.status_code != 200:  # noqa: PLR2004
                    errors += 1
            except httpx.HTTPError:
                errors += 1
            lat.append((time.perf_counter() - t0) * 1e3)

        tasks = []
        start = time.monotonic()
        i = 0
        while time.monotonic() - start < seconds:
            tasks.append(asyncio.create_task(one()))
            i += 1
            await asyncio.sleep(max(0.0, start + i / rate - time.monotonic()))
        await asyncio.gather(*tasks)
        took = time.monotonic() - start
    server.stop()
    return {
        "path": path,
        "target_rate": rate,
        "achieved_rate": len(lat) / took,
        "requests": len(lat),
        "errors": errors,
        "ms_p50": pct(lat, 50),
        "ms_p95": pct(lat, 95),
        "ms_p99": pct(lat, 99),
    }


def main() -> None:
    """Writes probes.json and latency.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("--state", "--corpus", "--users", "--out"):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument("--probes", type=int, default=10_000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from pymilvus import MilvusClient  # noqa: PLC0415

    directory = Directory()
    directory.load(args.users)
    users = sorted(str(u) for u in directory._people)
    docs = load_corpus(args.corpus)
    policies = PolicyStore([d.policy for d in docs])
    engine = PolicyEngine(directory, policies)
    directory_records = [
        {
            "user_id": p.user_id,
            "department": p.department,
            "role": p.role,
            "groups": sorted(p.groups),
        }
        for p in directory._people.values()
    ]
    policy_records = [d.policy.model_dump(mode="json") for d in docs]
    index = MilvusIndex(MilvusClient(str(args.state / "milvus.db")))
    result = probes(
        engine,
        index,
        users,
        directory_records,
        policy_records,
        args.probes,
        random.Random(252),
    )
    (args.out / "probes.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    clock, issuer = SystemClock(), TokenIssuer()
    app = create_app(directory, policies, issuer.verifier(clock), clock)
    doc_ids = [str(d.policy.doc_id) for d in docs]
    latency = [
        asyncio.run(
            http_latency(app, issuer, users, doc_ids, 500.0, 60.0, path)
        )
        for path in ("/v1/authorize", "/v1/filter")
    ]
    (args.out / "latency.json").write_text(json.dumps(latency, indent=1))
    print(json.dumps(latency, indent=1))


if __name__ == "__main__":
    main()
