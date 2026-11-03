from pathlib import Path

import httpx
import pytest

from askiisc.common.types import Chunk, ChunkId, DocId, Version
from askiisc.retrieval.embedder import (
    EmbeddingError,
    FakeEmbedder,
    HttpEmbedder,
)
from askiisc.retrieval.store import (
    InMemoryIndex,
    MilvusIndex,
    check_id,
    milvus_filter,
)

EMB = FakeEmbedder(dim=16)


def _chunk(
    doc: str, version: int, text: str, valid_to: float | None = None
) -> Chunk:
    return Chunk(
        id=ChunkId(f"{doc}@v{version}#0"),
        doc_id=DocId(doc),
        version=Version(version),
        text=text,
        page=1,
        valid_from=10.0,
        valid_to=valid_to,
        acl=frozenset({"public"}),
        embedding=EMB.embed_one(text),
    )


CHUNKS = [
    _chunk("fees", 1, "hostel fees are due in july", valid_to=50.0),
    _chunk("fees", 2, "hostel fees are due in august"),
    _chunk("exam", 1, "attendance must be eighty percent"),
    _chunk("secret", 1, "hostel fees for staff quarters"),
]


def test_fake_embedder_is_deterministic_and_unit() -> None:
    a, b = EMB.embed_one("hostel fees"), EMB.embed_one("hostel fees")
    assert a == b
    assert sum(x * x for x in a) == pytest.approx(1.0)
    assert EMB.embed_one("")[0] == 1.0


def test_in_memory_filters_inside_search() -> None:
    index = InMemoryIndex()
    index.upsert(CHUNKS)
    query = EMB.embed_one("hostel fees")
    allowed = frozenset({DocId("fees"), DocId("exam")})
    now = [h.chunk.id for h in index.search(query, allowed, 60.0, 8)]
    assert now[0] == "fees@v2#0"
    assert "secret@v1#0" not in now
    assert "fees@v1#0" not in now
    past = [h.chunk.id for h in index.search(query, allowed, 20.0, 8)]
    assert "fees@v1#0" in past
    assert index.search(query, frozenset(), 60.0, 8) == []


def test_in_memory_retire_and_validation() -> None:
    index = InMemoryIndex()
    index.upsert(CHUNKS)
    assert index.retire(DocId("exam"), Version(1), 30.0) == 1
    query = EMB.embed_one("attendance")
    hits = index.search(query, frozenset({DocId("exam")}), 40.0, 8)
    assert hits == []
    assert len(index.all_chunks()) == 4
    with pytest.raises(ValueError):
        index.upsert([CHUNKS[0].model_copy(update={"embedding": None})])


def test_filter_expression_and_ids() -> None:
    expr = milvus_filter([DocId("b"), DocId("a")], 12.5)
    assert (
        expr
        == 'doc_id in ["a", "b"] and valid_from <= 12.5 and valid_to > 12.5'
    )
    with pytest.raises(ValueError):
        milvus_filter([], 1.0)
    with pytest.raises(ValueError):
        check_id('x" or 1 == 1 or "')


def test_milvus_lite_matches_exact_index(tmp_path: Path) -> None:
    from pymilvus import MilvusClient

    client = MilvusClient(str(tmp_path / "milvus.db"))
    index = MilvusIndex(client, dim=16)
    index.upsert(CHUNKS)
    exact = InMemoryIndex()
    exact.upsert(CHUNKS)
    query = EMB.embed_one("hostel fees")
    allowed = frozenset({DocId("fees"), DocId("exam")})
    for at in (20.0, 60.0):
        got = [h.chunk.id for h in index.search(query, allowed, at, 8)]
        want = [h.chunk.id for h in exact.search(query, allowed, at, 8)]
        assert got == want
    assert index.retire(DocId("fees"), Version(2), 70.0) == 1
    after = index.search(query, allowed, 80.0, 8)
    assert {h.chunk.doc_id for h in after} == {"exam"}
    hit = index.search(query, allowed, 60.0, 1)[0]
    assert hit.chunk.valid_to == 70.0
    assert index.search(query, frozenset(), 60.0, 8) == []
    reopened = MilvusIndex(client, dim=16)
    assert reopened.search(query, allowed, 60.0, 1)[0].chunk.id == "fees@v2#0"
    with pytest.raises(ValueError):
        index.upsert([CHUNKS[0].model_copy(update={"embedding": None})])
    client.close()


def _embed_app(dim: int, wrong: bool = False) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        inputs = json.loads(request.content)["inputs"]
        size = dim + 1 if wrong else dim
        return httpx.Response(200, json=[[0.5] * size for _ in inputs])

    return httpx.MockTransport(handler)


async def test_http_embedder_batches_and_checks_shape() -> None:
    async with httpx.AsyncClient(transport=_embed_app(4)) as client:
        embedder = HttpEmbedder(client, "http://embed/", dim=4, batch_size=2)
        vectors = await embedder.embed(["a", "b", "c"])
    assert len(vectors) == 3
    assert embedder.dim == 4
    async with httpx.AsyncClient(transport=_embed_app(4, wrong=True)) as client:
        with pytest.raises(EmbeddingError):
            await HttpEmbedder(client, "http://embed", dim=4).embed(["a"])


async def test_http_embedder_wraps_http_errors() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(500))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(EmbeddingError):
            await HttpEmbedder(client, "http://embed", dim=4).embed(["a"])


def test_milvus_errors_become_index_store_errors() -> None:
    from pymilvus.exceptions import MilvusException

    from askiisc.retrieval.store import IndexStoreError

    class Broken:
        def has_collection(self, name: str) -> bool:
            return True

        def load_collection(self, name: str) -> None:
            return None

        def upsert(self, *a: object, **k: object) -> None:
            raise MilvusException(6, "value length exceeds max_length")

        query = search = upsert

    index = MilvusIndex(Broken(), dim=16)
    with pytest.raises(IndexStoreError, match="upsert"):
        index.upsert([CHUNKS[0]])
    with pytest.raises(IndexStoreError, match="query"):
        index.retire(DocId("fees"), Version(1), 1.0)
    with pytest.raises(IndexStoreError, match="search"):
        index.search(EMB.embed_one("x"), frozenset({DocId("fees")}), 1.0, 8)


async def test_http_embedder_retries_one_transport_error() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) % 2:
            raise httpx.ConnectError("blip")
        return httpx.Response(200, json=[[0.0] * 4])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    embedder = HttpEmbedder(client, "http://embed", dim=4)
    assert await embedder.embed(["x"]) == [(0.0, 0.0, 0.0, 0.0)]
    assert len(calls) == 2

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    broken = HttpEmbedder(
        httpx.AsyncClient(transport=httpx.MockTransport(down)),
        "http://e",
        dim=4,
    )
    with pytest.raises(EmbeddingError):
        await broken.embed(["x"])
