"""FR-5: a stale citation names the version that replaced it."""

from pathlib import Path

from askiisc.common.types import (
    Chunk,
    ChunkDraft,
    ChunkId,
    DocId,
    DocRef,
    Version,
)
from askiisc.ingest.lineage import Lineage, next_editions
from askiisc.ingest.versions import InMemoryVersionStore, SQLiteVersionStore
from askiisc.llm.rag import parse_answer

DRAFT = (ChunkDraft(text="Fees are due in July.", page=1, start=0, end=21),)
HANDBOOK = DocId("handbook")
H2025, H2026, H2027 = DocId("h-2025"), DocId("h-2026"), DocId("h-2027")


def store() -> InMemoryVersionStore:
    versions = InMemoryVersionStore()
    versions.publish(HANDBOOK, DRAFT, 10.0, "a")
    versions.publish(HANDBOOK, DRAFT, 20.0, "b")
    versions.publish(HANDBOOK, DRAFT, 30.0, "c")
    for doc, at in ((H2025, 1.0), (H2026, 2.0)):
        versions.publish(doc, DRAFT, at, doc)
    return versions


ROWS = [
    {"doc_id": "h-2026", "family": "hb", "valid_from": 2.0},
    {"doc_id": "h-2025", "family": "hb", "valid_from": 1.0},
    {"doc_id": "h-2027", "family": "hb", "valid_from": 3.0},
    {"doc_id": "alone", "family": "solo", "valid_from": 1.0},
    {"doc_id": "undated", "family": "hb"},
    {"doc_id": "nofamily", "valid_from": 1.0},
]


def test_next_editions_follow_each_family_by_start() -> None:
    assert next_editions(ROWS) == {H2025: H2026, H2026: H2027}


def test_a_revision_is_replaced_by_the_next_version() -> None:
    lineage = Lineage(store(), next_editions(ROWS))
    assert lineage(HANDBOOK, Version(1)) == DocRef(doc_id=HANDBOOK, version=2)
    assert lineage(HANDBOOK, Version(2)) == DocRef(doc_id=HANDBOOK, version=3)
    assert lineage(HANDBOOK, Version(3)) is None


def test_an_edition_is_replaced_by_the_next_editions_current_version() -> None:
    lineage = Lineage(store(), next_editions(ROWS))
    assert lineage(H2025, Version(1)) == DocRef(doc_id=H2026, version=1)
    # h-2027 is in the catalogue but not ingested: unknown, so None.
    assert lineage(H2026, Version(1)) is None
    assert lineage(DocId("never-seen"), Version(1)) is None


def test_the_sqlite_store_gives_the_same_lineage(tmp_path: Path) -> None:
    versions = SQLiteVersionStore(tmp_path / "v.db")
    versions.publish(HANDBOOK, DRAFT, 10.0, "a")
    versions.publish(HANDBOOK, DRAFT, 20.0, "b")
    lineage = Lineage(versions, {})
    assert lineage(HANDBOOK, Version(1)) == DocRef(doc_id=HANDBOOK, version=2)


def chunk(doc: str, version: int, valid_to: float | None) -> Chunk:
    return Chunk(
        id=ChunkId(f"{doc}@v{version}#0"),
        doc_id=DocId(doc),
        version=Version(version),
        text="Fees are due in July.",
        page=1,
        valid_from=0.0,
        valid_to=valid_to,
    )


def test_parse_answer_fills_superseded_by_for_stale_citations_only() -> None:
    old, new = chunk("handbook", 1, 15.0), chunk("handbook", 3, None)
    lineage = Lineage(store(), {})
    text = "Fees are due in July [chunk:handbook@v1#0] [chunk:handbook@v3#0]."
    answer = parse_answer(text, (old, new), now=100.0, successor=lineage)
    first, second = answer.citations
    assert first.is_stale and first.superseded_by == DocRef(
        doc_id=HANDBOOK, version=2
    )
    assert not second.is_stale and second.superseded_by is None
    plain = parse_answer(text, (old, new), now=100.0)
    assert plain.citations[0].is_stale
    assert plain.citations[0].superseded_by is None


async def test_the_answer_event_and_sources_carry_the_flags(
    tmp_path: Path,
) -> None:
    from askiisc.web.chat import enrich
    from tests.web_fixtures import HDR, build_web, sse

    web = build_web(tmp_path)
    web.account("u_cds_stu")
    await web.login("u_cds_stu")
    r = await web.client.post(
        "/api/temporary",
        json={"question": "What are the hostel fees rules?"},
        headers=HDR,
    )
    answer = dict(sse(r))["answer"]
    assert answer["staleness_flag"] is answer["stale"]
    payload = {
        "citations": [
            {"doc_id": "a", "page": 1, "chunk_id": "a@v1#0",
             "superseded_by": {"doc_id": "b", "version": 2}},
            {"doc_id": "a", "page": 1, "chunk_id": "a@v1#0",
             "superseded_by": None},
        ]
    }  # fmt: skip
    first, second = enrich(web.deps, payload)["citations"]
    assert first["superseded_by"]["version"] == 2  # noqa: PLR2004
    assert "title" in first["superseded_by"]
    assert second["superseded_by"] is None
