"""Turns PDF, HTML and plain-text documents into per-page text.

Content is not filtered or rewritten here; prompt-injection defence happens
at answer time.
"""

from __future__ import annotations

import io

from bs4 import BeautifulSoup
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from askiisc.common.errors import ExtractionError
from askiisc.common.types import Page

PDF = "application/pdf"
HTML = "text/html"
TEXT = "text/plain"
_HTML_NOISE = ("script", "style", "nav", "header", "footer", "noscript")


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
    else:
        raise ExtractionError(f"unsupported content type: {content_type}")
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
    except (PdfReadError, ValueError, OSError) as err:
        raise ExtractionError(f"unreadable PDF: {err}") from err


def _html_text(data: bytes) -> str:
    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(list(_HTML_NOISE)):
        tag.decompose()
    lines = (line.strip() for line in soup.get_text("\n").splitlines())
    return "\n".join(line for line in lines if line)
