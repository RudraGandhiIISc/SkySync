import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from askiisc.common.errors import ExtractionError
from askiisc.common.types import Page
from askiisc.ingest.chunk import WhitespaceCounter, chunk_pages
from askiisc.ingest.extract import extract_text
from tests.helpers import make_pdf


def test_pdf_pages_keep_numbers() -> None:
    data = make_pdf(["Attendance must be 80%.", "Fees are due in July."])
    pages = extract_text(data, "application/pdf")
    assert [p.number for p in pages] == [1, 2]
    assert "Attendance" in pages[0].text
    assert "Fees" in pages[1].text


def test_scanned_pdf_is_rejected() -> None:
    with pytest.raises(ExtractionError, match="no text layer"):
        extract_text(make_pdf(["", ""]), "application/pdf")


def test_broken_pdf_and_unknown_type_are_rejected() -> None:
    with pytest.raises(ExtractionError, match="unreadable"):
        extract_text(b"%PDF-1.4 garbage", "application/pdf")
    with pytest.raises(ExtractionError, match="unsupported"):
        extract_text(b"x", "image/png")


def test_html_drops_navigation_and_scripts() -> None:
    html = (
        b"<html><head><script>evil()</script><style>p{}</style></head>"
        b"<body><nav>Menu</nav><h1>Hostel rules</h1><p>Lights out at 11.</p>"
        b"<footer>Copyright</footer></body></html>"
    )
    pages = extract_text(html, "text/html; charset=utf-8")
    assert pages == [Page(number=1, text="Hostel rules\nLights out at 11.")]


def test_plain_text() -> None:
    assert extract_text(b"hello\n", "text/plain")[0].text == "hello"


def test_chunk_size_overlap_and_pages() -> None:
    sentence = "This is one short sentence of eight words."
    page = Page(number=3, text=" ".join([sentence] * 20))
    drafts = chunk_pages([page], target_tokens=20, overlap=8)
    counter = WhitespaceCounter()
    assert all(counter.count(d.text) <= 20 for d in drafts)
    assert all(d.page == 3 for d in drafts)
    assert drafts[1].start < drafts[0].end  # overlap


def test_table_rows_are_never_split() -> None:
    row = "| " + " | ".join(f"cell{i}" for i in range(30)) + " |"
    page = Page(number=1, text=f"Intro sentence.\n{row}\nOutro sentence.")
    drafts = chunk_pages([page], target_tokens=10, overlap=2)
    assert any(row in d.text for d in drafts)


def test_long_sentence_splits_on_words() -> None:
    page = Page(number=1, text=" ".join(f"w{i}" for i in range(50)))
    drafts = chunk_pages([page], target_tokens=10, overlap=0)
    assert len(drafts) == 5


def test_bad_sizes_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_pages([], target_tokens=10, overlap=10)


words = st.text(alphabet="abcdefgh", min_size=1, max_size=8)
sentences = st.lists(words, min_size=1, max_size=30).map(
    lambda ws: " ".join(ws) + "."
)
page_texts = st.lists(sentences, min_size=1, max_size=30).map(" ".join)


@settings(max_examples=60, deadline=None)
@given(
    texts=st.lists(page_texts, min_size=1, max_size=3),
    target=st.integers(min_value=5, max_value=60),
    overlap_fraction=st.floats(min_value=0, max_value=0.5),
)
def test_chunks_cover_text_and_respect_limits(
    texts: list[str], target: int, overlap_fraction: float
) -> None:
    overlap = int(target * overlap_fraction)
    pages = [Page(number=i + 1, text=t) for i, t in enumerate(texts)]
    drafts = chunk_pages(pages, target_tokens=target, overlap=overlap)
    counter = WhitespaceCounter()
    for page in pages:
        mine = [d for d in drafts if d.page == page.number]
        covered = [False] * len(page.text)
        for d in mine:
            assert d.text == page.text[d.start : d.end]
            assert counter.count(d.text) <= target
            for i in range(d.start, d.end):
                covered[i] = True
        for m in re.finditer(r"\S", page.text):
            assert covered[m.start()]
