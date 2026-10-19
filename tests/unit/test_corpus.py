import json
from pathlib import Path

import pytest

from askiisc.common.errors import ConfigError, NotFound
from askiisc.common.types import DocId
from askiisc.ingest.corpus import (
    CorpusSource,
    load_corpus,
    write_policies,
)


def _corpus(root: Path) -> Path:
    pub = root / "public" / "cds" / "department"
    pub.mkdir(parents=True)
    (pub / "cds-index-1.html").write_text("<p>public page</p>")
    entry = {
        "doc_id": "cds-index-1",
        "path": "public/cds/department/cds-index-1.html",
        "etag": '"e1"',
    }
    (root / "manifest.jsonl").write_text(json.dumps(entry) + "\n")
    res = root / "restricted" / "cds"
    res.mkdir(parents=True)
    (res / "cds-minutes.txt").write_text("minutes")
    (res / "cds-minutes.policy.json").write_text('{"groups": ["cds-faculty"]}')
    return root


async def test_corpus_policies_and_source(tmp_path: Path) -> None:
    docs = load_corpus(_corpus(tmp_path))
    by_id = {d.policy.doc_id: d for d in docs}
    assert by_id[DocId("cds-index-1")].policy.visibility == "public"
    minutes = by_id[DocId("cds-minutes")].policy
    assert minutes.visibility == "restricted"
    assert minutes.groups == {"cds-faculty"}
    rows = json.loads(write_policies(tmp_path, docs).read_text())
    assert {r["doc_id"]: r["visibility"] for r in rows} == {
        "cds-index-1": "public",
        "cds-minutes": "restricted",
    }
    source = CorpusSource(docs)
    listed = {d.doc_id: d for d in await source.list_documents()}
    assert listed[DocId("cds-index-1")].etag == '"e1"'
    assert listed[DocId("cds-minutes")].content_type == "text/plain"
    assert await source.fetch(DocId("cds-minutes")) == b"minutes"
    with pytest.raises(NotFound):
        await source.fetch(DocId("missing"))


def test_empty_corpus(tmp_path: Path) -> None:
    assert load_corpus(tmp_path) == []


@pytest.mark.parametrize(
    ("policy", "message"),
    [
        ("{}", "needs a rule"),
        ('{"groups": [], "roles": []}', "needs a rule"),
        ('{"groups": ["g"], "visibility": "public"}', "come from the path"),
        ('{"groups": ["g"], "colour": "red"}', "malformed"),
        ("not json", "malformed"),
    ],
)
def test_bad_restricted_policies(
    tmp_path: Path, policy: str, message: str
) -> None:
    _corpus(tmp_path)
    (tmp_path / "restricted/cds/cds-minutes.policy.json").write_text(policy)
    with pytest.raises(ConfigError, match=message):
        load_corpus(tmp_path)


def test_restricted_needs_exactly_one_file_and_unique_ids(
    tmp_path: Path,
) -> None:
    _corpus(tmp_path)
    (tmp_path / "restricted/cds/cds-minutes.pdf").write_bytes(b"%PDF")
    with pytest.raises(ConfigError, match="one document file"):
        load_corpus(tmp_path)
    (tmp_path / "restricted/cds/cds-minutes.pdf").unlink()
    other = tmp_path / "restricted" / "csa"
    other.mkdir()
    (other / "cds-index-1.txt").write_text("clash")
    (other / "cds-index-1.policy.json").write_text('{"roles": ["staff"]}')
    with pytest.raises(ConfigError, match="duplicate"):
        load_corpus(tmp_path)
