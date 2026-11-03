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


def test_html_drops_site_chrome_and_keeps_main_content() -> None:
    body = " ".join(["Applicants need a valid GATE score."] * 10)
    html = (
        "<html><body><div id='responsive-menu'><a>Home</a><a>People</a></div>"
        "<div class='sub-menu'><a>Faculty</a></div>"
        "<div id='header-widget-area' class='hw-widget'>Search</div>"
        f"<div id='mainpanel'><h1>MTech admissions</h1><p>{body}</p></div>"
        "<aside>Recent posts</aside><div id='footer-bottom'>Copyright</div>"
        "<div class='footer'>Contact</div></body></html>"
    ).encode()
    text = extract_text(html, "text/html")[0].text
    assert text.startswith("MTech admissions")
    for chrome in (
        "Home",
        "Faculty",
        "Search",
        "Recent posts",
        "Copyright",
        "Contact",
    ):
        assert chrome not in text


def test_menu_state_class_on_body_keeps_the_page() -> None:
    # Regression: www.csa.iisc.ac.in puts "responsive-menu-slide-right" on
    # <body>, and the whole body was dropped as a menu (20 characters left).
    rules = " ".join(["The ERP admits working professionals."] * 10)
    html = (
        "<html><body class='page responsive-menu-slide-right'>"
        "<div class='wrap-menu-open'><div id='mainpanel'>"
        f"<h1>External Registration Programme</h1><p>{rules}</p>"
        "</div></div></body></html>"
    ).encode()
    text = extract_text(html, "text/html")[0].text
    assert text.startswith("External Registration Programme")
    assert "working professionals" in text


def test_elementor_widget_containers_are_content() -> None:
    # Regression: every Elementor block is "elementor-widget-container";
    # matching "widget" anywhere in a class removed whole pages.
    para = " ".join(["The BSL-3 facility is a containment laboratory."] * 8)
    html = (
        "<html><body><div class='widget'>Recent posts</div>"
        "<div class='elementor-nav-menu'>Home About</div>"
        f"<div class='elementor-widget-container'><p>{para}</p></div>"
        "</body></html>"
    ).encode()
    text = extract_text(html, "text/html")[0].text
    assert "containment laboratory" in text
    assert "Recent posts" not in text and "Home About" not in text


def test_chrome_classes_that_hold_the_content_fall_back() -> None:
    para = " ".join(["Fee for the M.Tech programme is listed here."] * 6)
    html = (
        f"<html><body><div class='site-sidebar'><p>{para}</p></div>"
        "<p>Short.</p></body></html>"
    ).encode()
    assert "M.Tech programme" in extract_text(html, "text/html")[0].text


def test_html_without_main_region_keeps_body_text() -> None:
    html = (
        b"<html><body><div class='entry'><p>Short rule.</p></div></body></html>"
    )
    assert extract_text(html, "text/html")[0].text == "Short rule."


def test_pypdf_limit_errors_become_extraction_errors(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Regression: a LimitReachedError from pypdf crashed crawl job 1330."""
    from pypdf.errors import LimitReachedError

    from askiisc.ingest import extract

    def boom(*args: object, **kwargs: object) -> None:
        raise LimitReachedError("Reached maximum number of tokens per stream.")

    monkeypatch.setattr(extract, "PdfReader", boom)
    with pytest.raises(ExtractionError, match="unreadable"):
        extract_text(b"%PDF-1.4", "application/pdf")


def test_no_chunk_exceeds_the_character_cap() -> None:
    """Regression: a 105,238-character chunk broke the Milvus upsert in
    ingest job 1348 (text limit 65,535)."""
    from askiisc.ingest.chunk import MAX_CHUNK_CHARS

    blob = "x" * 30_000  # one "word" with no spaces
    row = "| " + "y" * 20_000 + " | z |"
    page = Page(number=1, text=f"Intro.\n{row}\n{blob} end.")
    drafts = chunk_pages([page])
    assert all(len(d.text) <= MAX_CHUNK_CHARS for d in drafts)
    joined = "".join(d.text for d in drafts)
    assert joined.count("x") >= 30_000 and joined.count("y") >= 20_000


def test_lone_surrogates_are_replaced(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Regression: two tender PDFs broke the embedding request."""
    from askiisc.ingest import extract

    monkeypatch.setattr(
        extract, "_pdf_pages", lambda data: ["Tender \ud835 notice"]
    )
    page = extract_text(b"%PDF", "application/pdf")[0]
    page.text.encode("utf-8")  # must not raise
    assert page.text.startswith("Tender")


def _office(files: dict[str, str]) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return buf.getvalue()


def test_office_documents() -> None:
    from askiisc.ingest.extract import DOCX, PPTX, XLSX

    docx = _office({"word/document.xml": (
        "<w:document><w:body><w:p><w:r><w:t>Fee rules</w:t></w:r></w:p>"
        '<w:p><w:r><w:t xml:space="preserve">Pay by </w:t></w:r>'
        "<w:r><w:t>5 May &amp; on time</w:t></w:r></w:p></w:body></w:document>"
    )})  # fmt: skip
    assert (
        extract_text(docx, DOCX)[0].text == "Fee rules\nPay by 5 May & on time"
    )
    slide = "<p:sld><a:p><a:r><a:t>{}</a:t></a:r></a:p></p:sld>"
    pptx = _office({
        "ppt/slides/slide2.xml": slide.format("Second"),
        "ppt/slides/slide1.xml": slide.format("First"),
    })  # fmt: skip
    assert [p.text for p in extract_text(pptx, PPTX)] == ["First", "Second"]
    xlsx = _office({
        "xl/sharedStrings.xml":
            "<sst><si><t>Course</t></si><si><t>Credits</t></si></sst>",
        "xl/worksheets/sheet1.xml": (
            '<worksheet><sheetData><row><c r="A1" t="s"><v>0</v></c>'
            '<c r="B1" t="s"><v>1</v></c></row><row><c r="A2" t="inlineStr">'
            '<is><t>DS252</t></is></c><c r="B2"><v>3</v></c><c r="C2"/></row>'
            "</sheetData></worksheet>"
        ),
    })  # fmt: skip
    assert extract_text(xlsx, XLSX)[0].text == "Course\tCredits\nDS252\t3"


def test_broken_and_oversized_office_files() -> None:
    from askiisc.ingest import extract
    from askiisc.ingest.extract import DOCX

    with pytest.raises(ExtractionError, match="unreadable office"):
        extract_text(b"not a zip", DOCX)
    with pytest.raises(ExtractionError, match="unreadable office"):
        extract_text(_office({"other.xml": "x"}), DOCX)
    big = _office({"word/document.xml": "<w:t>x</w:t>" * 1000})
    old = extract.MAX_UNPACKED_BYTES
    extract.MAX_UNPACKED_BYTES = 100
    try:
        with pytest.raises(ExtractionError, match="too large"):
            extract_text(big, DOCX)
    finally:
        extract.MAX_UNPACKED_BYTES = old
