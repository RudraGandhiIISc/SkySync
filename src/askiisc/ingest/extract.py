"""Turns PDF, HTML, plain-text and Office documents into per-page text.

Office files (.docx, .pptx, .xlsx) are zip archives of XML; their text is
taken with regular expressions over the XML (no XML parser, so no entity
expansion), with a cap on the unpacked size against zip bombs. Word files
are one page, slides and sheets one page each.

Content is not filtered or rewritten here; prompt-injection defence happens
at answer time.
"""

from __future__ import annotations

import html
import io
import re
import zipfile

from bs4 import BeautifulSoup, Tag
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from askiisc.common.errors import ExtractionError
from askiisc.common.types import Page

PDF = "application/pdf"
HTML = "text/html"
TEXT = "text/plain"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
PPTX = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
)
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
OFFICE = frozenset({DOCX, PPTX, XLSX})
MAX_UNPACKED_BYTES = 100_000_000
_RUN = re.compile(r"<(?:w|a):t(?:\s[^>]*)?>([^<]*)</(?:w|a):t>")
_BREAK = re.compile(r"</(?:w|a):p>|<w:br/>|<w:tab/>")
_CELL = re.compile(r"<c\b([^>]*?)(?:/>|>(.*?)</c>)", re.DOTALL)
_VALUE = re.compile(r"<v>([^<]*)</v>|<t[^>]*>([^<]*)</t>")
_SHARED = re.compile(r"<si>(.*?)</si>", re.DOTALL)
_T = re.compile(r"<t[^>]*>([^<]*)</t>")
_ROW_END = re.compile(r"</row>")
_HTML_NOISE = (
    "script",
    "style",
    "nav",
    "header",
    "footer",
    "noscript",
    "aside",
    "form",
)
# Site chrome that WordPress and similar themes put in plain <div>s,
# matched on whole class or id tokens: "widget", "menu-item", "site-footer".
# A word inside a token is not enough: "elementor-widget-container" holds
# the content of every Elementor page.
_CHROME_WORDS = frozenset(
    {
        "menu",
        "nav",
        "navbar",
        "navigation",
        "footer",
        "sidebar",
        "widget",
        "breadcrumb",
        "breadcrumbs",
        "cookie",
        "social",
        "share",
        "skip-link",
        "search-box",
    }
)
_CHROME_PREFIXES = (
    "menu-",
    "nav-",
    "navbar-",
    "footer-",
    "sidebar-",
    "breadcrumb",
    "cookie-",
    "social-",
    "share-",
)
_CHROME_SUFFIXES = (
    "-menu",
    "-nav",
    "-navbar",
    "-navigation",
    "-footer",
    "-sidebar",
    "-widget",
    "-breadcrumb",
    "-breadcrumbs",
    "-social",
)
# If removing chrome by class leaves less than this share of the page's
# text (and under FALLBACK_CHARS), the classes were content: keep the page
# with only scripts, navigation, header and footer elements removed.
FALLBACK_RATIO = 0.4
FALLBACK_CHARS = 2000
_MAIN_ID = re.compile(r"^(main|content|primary|mainpanel|main-content)$")
MIN_MAIN_CHARS = 200
# Never site chrome, whatever their classes say: themes put state classes
# such as "responsive-menu-slide-right" on <body> (all of www.csa.iisc.ac.in).
_STRUCTURAL = frozenset({"html", "body", "main", "article"})


def extract_text(data: bytes, content_type: str) -> list[Page]:
    """Extracts the text of a document, one entry per page.

    Args:
        data: The raw document bytes.
        content_type: A MIME type; parameters such as charset are ignored.

    Returns:
        The pages that contain text, numbered from 1.

    Raises:
        ExtractionError: If the type is unsupported, the file is unreadable,
            or no page has a text layer (scanned PDFs are out of scope).
    """
    kind = content_type.split(";", maxsplit=1)[0].strip().lower()
    if kind == PDF:
        pages = _pdf_pages(data)
    elif kind == HTML:
        pages = [_html_text(data)]
    elif kind == TEXT:
        pages = [data.decode("utf-8", errors="replace")]
    elif kind in OFFICE:
        pages = _office_pages(data, kind)
    else:
        raise ExtractionError(f"unsupported content type: {content_type}")
    # PDFs can yield lone surrogates, which cannot be sent as UTF-8.
    pages = [p.encode("utf-8", "replace").decode("utf-8") for p in pages]
    result = [
        Page(number=i, text=text.strip())
        for i, text in enumerate(pages, start=1)
        if text.strip()
    ]
    if not result:
        raise ExtractionError("no text layer (scanned or empty document)")
    return result


def _pdf_pages(data: bytes) -> list[str]:
    try:
        reader = PdfReader(io.BytesIO(data))
        return [page.extract_text() or "" for page in reader.pages]
    # pypdf raises many types on broken files (PyPdfError subclasses such
    # as LimitReachedError, and plain KeyError, TypeError, RecursionError,
    # AttributeError); any of them means the PDF cannot be used.
    except (
        PyPdfError,
        ValueError,
        OSError,
        KeyError,
        TypeError,
        AttributeError,
        IndexError,
        RecursionError,
    ) as err:
        raise ExtractionError(f"unreadable PDF: {err}") from err


def _runs(xml: str) -> str:
    """The text runs of Word or slide XML, one line per paragraph."""
    marked = _BREAK.sub("\n", xml)
    lines = []
    for line in marked.split("\n"):
        text = "".join(_RUN.findall(line))
        if text.strip():
            lines.append(html.unescape(text))
    return "\n".join(lines)


def _sheet(xml: str, shared: list[str]) -> str:
    """A worksheet as tab-separated rows."""
    rows = []
    for row in _ROW_END.split(xml):
        cells = []
        for attrs, body in _CELL.findall(row):
            found = _VALUE.search(body or "")
            if not found:
                continue
            value = found.group(1) or found.group(2) or ""
            if (
                't="s"' in attrs
                and value.isdigit()
                and int(value) < len(shared)
            ):
                value = shared[int(value)]
            cells.append(html.unescape(value))
        if any(c.strip() for c in cells):
            rows.append("\t".join(cells))
    return "\n".join(rows)


def _office_pages(data: bytes, kind: str) -> list[str]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if sum(i.file_size for i in z.infolist()) > MAX_UNPACKED_BYTES:
                raise ExtractionError("office file unpacks too large")
            names = z.namelist()

            def read(name: str) -> str:
                return z.read(name).decode("utf-8", errors="replace")

            if kind == DOCX:
                return [_runs(read("word/document.xml"))]
            numbered = re.compile(
                r"ppt/slides/slide(\d+)\.xml$"
                if kind == PPTX
                else r"xl/worksheets/sheet(\d+)\.xml$"
            )
            parts = sorted(
                (int(m.group(1)), n) for n in names if (m := numbered.match(n))
            )
            if kind == PPTX:
                return [_runs(read(n)) for _, n in parts]
            shared = []
            if "xl/sharedStrings.xml" in names:
                shared = [
                    "".join(_T.findall(si))
                    for si in _SHARED.findall(read("xl/sharedStrings.xml"))
                ]
            return [_sheet(read(n), shared) for _, n in parts]
    except (zipfile.BadZipFile, KeyError, OSError, ValueError) as err:
        raise ExtractionError(f"unreadable office file: {err}") from err


def _is_boilerplate(tag: Tag) -> bool:
    if tag.name in _STRUCTURAL:
        return False
    attrs = tag.attrs or {}
    tokens = [str(attrs.get("id", "")), *attrs.get("class", [])]
    return any(_chrome_token(t.lower()) for t in tokens if t)


def _chrome_token(token: str) -> bool:
    return (
        token in _CHROME_WORDS
        or token.startswith(_CHROME_PREFIXES)
        or token.endswith(_CHROME_SUFFIXES)
    )


def _protected(soup: BeautifulSoup) -> set[int]:
    """Ids of the main-content candidates and all their ancestors."""
    found = [soup.find("main"), soup.find("article")]
    found += soup.find_all(id=_MAIN_ID)
    keep: set[int] = set()
    for tag in found:
        if isinstance(tag, Tag):
            keep.add(id(tag))
            keep.update(id(parent) for parent in tag.parents)
    return keep


def _main_region(soup: BeautifulSoup) -> Tag | BeautifulSoup:
    for found in (
        soup.find("main"),
        soup.find("article"),
        soup.find(id=_MAIN_ID),
    ):
        if isinstance(found, Tag) and len(found.get_text(strip=True)) >= (
            MIN_MAIN_CHARS
        ):
            return found
    return soup


def _lines(region: Tag | BeautifulSoup) -> str:
    lines = (line.strip() for line in region.get_text("\n").splitlines())
    return "\n".join(line for line in lines if line)


def _html_text(data: bytes) -> str:
    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(list(_HTML_NOISE)):
        tag.decompose()
    plain = _lines(soup)
    keep = _protected(soup)
    for tag in soup.find_all(_is_boilerplate):
        if not tag.decomposed and id(tag) not in keep:
            tag.decompose()
    text = _lines(_main_region(soup))
    if len(text) < FALLBACK_CHARS and len(text) < FALLBACK_RATIO * len(plain):
        return plain
    return text
