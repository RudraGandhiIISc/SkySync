"""FR-2: 10,000 access probes against Milvus, checked by the oracle.

Zero forbidden chunks may be returned, and recall@8 against an exact
filtered search must be at least 0.95. Milvus Lite indexes as FLAT, so
recall here is exact; the HNSW figure needs Milvus standalone (Phase 12).
"""

import random
from pathlib import Path

from eval.oracle.oracle import LogEntry, check_run

from askiisc.access.directory import Directory
from askiisc.access.policy import DocPolicy, PolicyEngine, PolicyStore
from askiisc.common.targets import (
    FR2_ACCESS_PROBES,
    FR2_MIN_RECALL_AT_K,
    RETRIEVAL_TOP_K,
)
from askiisc.common.types import Chunk, ChunkId, DocId, UserId, Version
from askiisc.retrieval.embedder import FakeEmbedder
from askiisc.retrieval.store import InMemoryIndex, MilvusIndex

GROUPS = [f"g{i}" for i in range(12)]
DEPARTMENTS = ["cds", "ece", "csa", "physics", "biology"]
WORDS = [
    "fees",
    "hostel",
    "exam",
    "attendance",
    "thesis",
    "library",
    "leave",
    "travel",
    "grant",
    "lab",
    "safety",
    "course",
    "registration",
    "stipend",
    "convocation",
    "rules",
    "deadline",
]
AT = 1000.0


def _world(rng: random.Random) -> tuple[list[dict], list[dict]]:
    directory = [
        {
            "user_id": f"user{i}",
            "department": rng.choice(DEPARTMENTS),
            "role": rng.choice(["student", "faculty", "staff"]),
            "groups": rng.sample(GROUPS, rng.randint(0, 3)),
        }
        for i in range(200)
    ]
    policies = []
    for d in range(100):
        policy: dict = {
            "doc_id": f"doc{d:03d}",
            "groups": rng.sample(GROUPS, rng.randint(0, 2)),
        }
        if rng.random() < 0.3:
            policy["departments"] = [rng.choice(DEPARTMENTS)]
        if rng.random() < 0.1:
            policy["deny_users"] = rng.sample(
                [u["user_id"] for u in directory], 5
            )
        if rng.random() < 0.1:
            policy["valid_from"] = AT + 1  # embargoed
        policies.append(policy)
    return directory, policies


def _chunks(rng: random.Random, embedder: FakeEmbedder) -> list[Chunk]:
    chunks = []
    for d in range(100):
        for n in range(20):
            text = " ".join(rng.choices(WORDS, k=12))
            chunks.append(
                Chunk(
                    id=ChunkId(f"doc{d:03d}@v1#{n}"),
                    doc_id=DocId(f"doc{d:03d}"),
                    version=Version(1),
                    text=text,
                    page=1,
                    valid_from=0.0,
                    embedding=embedder.embed_one(text),
                )
            )
    return chunks


def test_ten_thousand_probes_zero_leaks_and_recall(tmp_path: Path) -> None:
    from pymilvus import MilvusClient

    rng = random.Random(252)
    embedder = FakeEmbedder(dim=64)
    directory, policies = _world(rng)
    engine = PolicyEngine(
        Directory.from_records(directory),
        PolicyStore([DocPolicy(**p) for p in policies]),
    )
    chunks = _chunks(rng, embedder)
    client = MilvusClient(str(tmp_path / "probes.db"))
    milvus = MilvusIndex(client, dim=64)
    milvus.upsert(chunks)
    exact = InMemoryIndex()
    exact.upsert(chunks)
    log, hits, wanted = [], 0, 0
    for _ in range(FR2_ACCESS_PROBES):
        user = UserId(rng.choice(directory)["user_id"])
        allowed = engine.filter(user, AT).doc_ids
        vector = embedder.embed_one(" ".join(rng.choices(WORDS, k=4)))
        got = milvus.search(vector, allowed, AT, RETRIEVAL_TOP_K)
        want = exact.search(vector, allowed, AT, RETRIEVAL_TOP_K)
        log.append(
            LogEntry(user, AT, tuple((h.chunk.id, h.chunk.doc_id) for h in got))
        )
        want_ids = {h.chunk.id for h in want}
        hits += len(want_ids & {h.chunk.id for h in got})
        wanted += len(want_ids)
    client.close()
    assert check_run(log, directory, policies) == []
    assert wanted > 0
    assert hits / wanted >= FR2_MIN_RECALL_AT_K
