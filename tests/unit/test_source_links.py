"""Citations link to the exact cited passage; documents carry real titles."""

import json
from pathlib import Path

from askiisc.ingest.corpus import html_title, read_titles, write_titles
from askiisc.web.chat import source_link

URL = "https://www.iisc.ac.in/students/fees/"


def test_pdf_links_open_the_cited_page() -> None:
    assert source_link(URL + "f.pdf", "pdf", 3, "x") == URL + "f.pdf#page=3"
    assert source_link(URL + "f.pdf", "pdf", 0, None).endswith("#page=1")


def test_web_links_highlight_the_opening_words() -> None:
    text = (
        "Home\nFees\n"
        "Hostel fees are paid each term through the e-portal, by the 5th."
    )
    link = source_link(URL, "html", 1, text)
    assert link == (
        URL
        + "#:~:text=Hostel%20fees%20are%20paid%20each%20term%20through%20the"
    )
    dashed = source_link(
        URL, "html", 1, "Check-in and check-out times are fixed"
    )
    assert "Check%2Din%20and" in dashed  # "-" is reserved in text fragments


def test_links_without_text_or_address() -> None:
    assert source_link("", "pdf", 2, "x") == ""
    assert source_link(URL, "html", 1, None) == URL
    assert source_link(URL, "html", 1, "Home\nMenu") == URL
    assert source_link(URL + "a.docx", "docx", 1, "x") == URL + "a.docx"


def test_titles(tmp_path: Path) -> None:
    assert html_title(b"<html><TITLE> Fees &amp;\n Dues | IISc </TITLE>") == (
        "Fees & Dues | IISc"
    )
    assert html_title(b"<p>none</p>") == ""
    (tmp_path / "a.html").write_bytes(b"<title>Page A</title>")
    (tmp_path / "b.pdf").write_bytes(b"%PDF")
    rows = [
        {"doc_id": "a", "path": "a.html", "content_type": "text/html"},
        {"doc_id": "b", "path": "b.pdf", "content_type": "application/pdf"},
        {"doc_id": "c", "path": "c.html", "content_type": "text/html",
         "title": "Kept"},
    ]  # fmt: skip
    (tmp_path / "c.html").write_bytes(b"<title>Ignored</title>")
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    assert write_titles(tmp_path) == 2  # noqa: PLR2004
    assert read_titles(tmp_path) == {"a": "Page A", "c": "Kept"}
    (tmp_path / "titles.json").write_text("not json")
    assert read_titles(tmp_path) == {}


def test_enrich_adds_title_and_link() -> None:
    from types import SimpleNamespace

    from askiisc.web.chat import enrich

    web = SimpleNamespace(
        docs=lambda d: {"title": "Fee rules", "url": URL, "kind": "html",
                        "site": "www.iisc.ac.in"},
        chunk_text=lambda c: "Hostel fees are paid each term online.",
    )  # fmt: skip
    payload = {
        "citations": [{"doc_id": "d1", "chunk_id": "d1@v1#0", "page": 1}]
    }
    out = enrich(web, payload)  # type: ignore[arg-type]
    cite = out["citations"][0]
    assert cite["title"] == "Fee rules" and cite["site"] == "www.iisc.ac.in"
    assert cite["link"].startswith(URL + "#:~:text=Hostel%20fees")
