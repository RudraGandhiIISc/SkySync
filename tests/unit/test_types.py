import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from askiisc.common.errors import RateLimited
from askiisc.common.types import (
    Answer,
    Chunk,
    ChunkId,
    Citation,
    DocId,
    Version,
)

times = st.floats(min_value=0, max_value=1e10, allow_nan=False)


def _chunk(valid_from: float, valid_to: float | None) -> Chunk:
    return Chunk(
        id=ChunkId("c1"),
        doc_id=DocId("d1"),
        version=Version(1),
        text="text",
        page=1,
        valid_from=valid_from,
        valid_to=valid_to,
    )


@given(start=times, length=st.floats(min_value=1e-3, max_value=1e6))
def test_valid_interval_is_accepted(start: float, length: float) -> None:
    chunk = _chunk(start, start + length)
    assert chunk.is_current(start)
    assert not chunk.is_current(start + length)
    assert not chunk.is_current(None)


@given(start=times, back=st.floats(min_value=0, max_value=1e6))
def test_end_not_after_start_is_rejected(start: float, back: float) -> None:
    with pytest.raises(ValidationError):
        _chunk(start, start - back)


def test_current_chunk_has_no_end() -> None:
    chunk = _chunk(10.0, None)
    assert chunk.is_current(None)
    assert chunk.is_current(10.0)
    assert not chunk.is_current(9.0)


def test_chunk_is_frozen() -> None:
    chunk = _chunk(0.0, None)
    with pytest.raises(ValidationError):
        chunk.text = "changed"  # type: ignore[misc]


@given(
    doc=st.text(min_size=1, max_size=20),
    chunk=st.text(min_size=1, max_size=20),
    version=st.integers(min_value=1, max_value=1000),
    page=st.integers(min_value=1, max_value=5000),
    stale=st.booleans(),
)
def test_citation_json_round_trip(
    doc: str, chunk: str, version: int, page: int, stale: bool
) -> None:
    citation = Citation(
        doc_id=DocId(doc),
        version=Version(version),
        chunk_id=ChunkId(chunk),
        page=page,
        is_stale=stale,
    )
    assert Citation.model_validate_json(citation.model_dump_json()) == citation


def test_answer_defaults_and_rate_limited_retry() -> None:
    assert Answer(text="x").citations == ()
    err = RateLimited("busy", retry_after_s=2.0)
    assert err.retry_after_s == 2.0


def test_file_object_store(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from askiisc.common.errors import NotFound
    from askiisc.ingest.objects import FileObjectStore, InMemoryObjectStore

    store = FileObjectStore(tmp_path / "objects")
    store.put("doc/v1", b"bytes", "text/plain")
    assert store.get("doc/v1") == b"bytes"
    assert (
        tmp_path / "objects" / "doc" / "v1.type"
    ).read_text() == "text/plain"
    with pytest.raises(NotFound):
        store.get("doc/v2")
    for bad in ("../escape", "/etc/passwd", ""):
        with pytest.raises(ValueError):
            store.put(bad, b"x", "text/plain")
    memory = InMemoryObjectStore()
    memory.put("k", b"v", "text/plain")
    assert memory.keys() == ["k"]
    with pytest.raises(NotFound):
        memory.get("missing")
