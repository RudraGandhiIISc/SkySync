"""Editions: which documents are yearly versions of one another.

IISc publishes a new edition of many documents every academic year (the
Student Information Handbook 2023, 2024, 2025-V2, 2026-August; a course
page /courses/ds252-aug25/ next to /courses/DS252/). Each edition stays a
document of its own; editions of one family get consecutive validity
windows, so a search "as of" any date sees exactly the edition that was in
force then (CLAUDE.md safety rule 6: nothing is deleted, old editions are
retired by their end time).

- `detect_edition` reads an edition date from the address, title, upload
  folder, the opening text or the archive snapshot time, in that order.
- `family_key` is the address with year, month and version words removed.
- `assign_windows` orders a family's editions and sets valid_from/valid_to:
  the oldest edition is valid from the beginning of time (questions about
  earlier years see the oldest edition known), each later one from its own
  start, and the newest one stays current.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import unquote, urlparse

# Academic years at IISc start in August.
ACADEMIC_YEAR_START_MONTH = 8
MIN_YEAR = 2000
MAX_YEAR = 2035
# Characters of opening text searched for a date.
HEAD_CHARS = 1500
# File names shorter than this (after removing dates) are too generic to
# join files from different sites ("brochure", "notice").
MIN_FILE_STEM_CHARS = 12
BEGINNING = 0.0

_MONTHS = {
    name.lower(): i for i, name in enumerate(calendar.month_abbr) if name
}
_MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
_SEP = r"[\s_\-./,]*"
_ACADEMIC = re.compile(
    r"(?<!\d)(20\d\d)\s*[-–/_]\s*(?:20)?(\d\d)(?!\d)", re.IGNORECASE
)
_MONTH_YEAR = re.compile(
    rf"(?<![a-z]){_MONTH}{_SEP}(?:20)?(\d\d)(?!\d)", re.IGNORECASE
)
_YEAR_MONTH = re.compile(
    rf"(?<!\d)(20\d\d){_SEP}{_MONTH}(?![a-z])", re.IGNORECASE
)
_MONTH_RANGE = re.compile(
    rf"(?<![a-z]){_MONTH}\s*(?:-|–|to)\s*{_MONTH}{_SEP}(20\d\d)(?!\d)",
    re.IGNORECASE,
)
_YEAR = re.compile(r"(?<!\d)(20\d\d)(?!\d)")
_UPLOAD = re.compile(r"/uploads/(20\d\d)/(\d\d)/")
_MINOR = re.compile(
    r"(?:[-_ ]v(?:er(?:sion)?)?[-_ ]?(\d{1,2})|[-_ ](ii|iii|iv))(?![a-z0-9])",
    re.IGNORECASE,
)
_ROMAN = {"ii": 2, "iii": 3, "iv": 4}
_COURSE = re.compile(r"^([a-z]{1,3})[-_ ]?(\d{3})(?!\d)")
_ACADEMIC_TEXT = re.compile(
    r"academic\s+(?:year|session)\s*:?\s*(20\d\d)", re.IGNORECASE
)
# Words that name a copy rather than a document.
_NOISE = re.compile(
    r"(?<![a-z])(final|new|updated|revised|latest|web|copy|version|ver|v\d+"
    r"|ii|iii|iv|onwards|batch|ay|edition|current)(?![a-z])"
)


@dataclass(frozen=True)
class Edition:
    """When an edition took effect.

    Attributes:
        start: Epoch seconds, or None when no date was found.
        minor: Revision within the same date (V2 = 2), else 0.
        label: Human-readable edition, for example "2025-26" or "Aug 2025".
        source: Where the date came from ("url", "title", "upload", "text",
            "snapshot") or "" when unknown.
    """

    start: float | None
    minor: int = 0
    label: str = ""
    source: str = ""


def _epoch(year: int, month: int) -> float:
    return datetime(year, month, 1, tzinfo=UTC).timestamp()


def _year(two_or_four: str) -> int:
    value = int(two_or_four)
    return value + 2000 if value < 100 else value  # noqa: PLR2004


def _month(name: str) -> int:
    return _MONTHS[name[:3].lower()]


def _valid(year: int) -> bool:
    return MIN_YEAR <= year <= MAX_YEAR


def _academic(text: str) -> Edition | None:
    for match in _ACADEMIC.finditer(text):
        first, second = int(match.group(1)), int(match.group(2))
        if _valid(first) and second == (first + 1) % 100:
            start = _epoch(first, ACADEMIC_YEAR_START_MONTH)
            return Edition(start, label=f"{first}-{second:02d}")
    return None


def _month_year(text: str) -> Edition | None:
    found: list[tuple[int, int]] = []
    ranges = list(_MONTH_RANGE.finditer(text))
    for match in ranges:  # "Jan-Apr 2026" starts in January
        found.append((int(match.group(3)), _month(match.group(1))))
    if not ranges:
        for match in _MONTH_YEAR.finditer(text):
            found.append((_year(match.group(2)), _month(match.group(1))))
    for match in _YEAR_MONTH.finditer(text):
        found.append((int(match.group(1)), _month(match.group(2))))
    found = [f for f in found if _valid(f[0])]
    if not found:
        return None
    year, month = max(found)
    label = f"{calendar.month_abbr[month]} {year}"
    return Edition(_epoch(year, month), label=label)


def _bare_year(text: str, upload: tuple[int, int] | None) -> Edition | None:
    years = [int(y) for y in _YEAR.findall(text) if _valid(int(y))]
    if not years:
        return None
    year = max(years)
    month = upload[1] if upload and upload[0] == year else 1
    return Edition(_epoch(year, month), label=str(year))


def _dated(text: str, upload: tuple[int, int] | None) -> Edition | None:
    """The most specific date in a short text (address or title)."""
    return _academic(text) or _month_year(text) or _bare_year(text, upload)


def _minor(text: str) -> int:
    match = _MINOR.search(text)
    if not match:
        return 0
    if match.group(1):
        return int(match.group(1))
    return _ROMAN[match.group(2).lower()]


def _file_name(url: str) -> str:
    path = unquote(urlparse(url).path)
    return path.rstrip("/").rsplit("/", 1)[-1] or path


def _text_date(head: str) -> Edition | None:
    head = head[:HEAD_CHARS]
    found = _academic(head) or _month_year(head)
    if found:
        return found
    match = _ACADEMIC_TEXT.search(head)
    if match and _valid(int(match.group(1))):
        year = int(match.group(1))
        return Edition(_epoch(year, ACADEMIC_YEAR_START_MONTH), label=str(year))
    return None


def detect_edition(
    url: str, title: str = "", head: str = "", snapshot: float | None = None
) -> Edition:
    """Finds when a document's edition took effect.

    Args:
        url: The document's original address (not an archive address).
        title: Its title.
        head: Its opening text.
        snapshot: For an archived copy, when the archive saw it.

    Returns:
        The edition; start is None when nothing gave a date.
    """
    path = unquote(urlparse(url).path)
    upload_match = _UPLOAD.search(path)
    upload = (
        (int(upload_match.group(1)), int(upload_match.group(2)))
        if upload_match
        else None
    )
    name = _UPLOAD.sub("/", path).rstrip("/")
    minor = _minor(name)
    for source, text in (("url", name), ("title", title)):
        found = _dated(text, upload) if text else None
        if found:
            found = _refine_year(found, head)
            return Edition(found.start, minor, found.label, source)
    if upload:
        label = f"{calendar.month_abbr[upload[1]]} {upload[0]}"
        return Edition(_epoch(*upload), minor, label, "upload")
    found = _text_date(head)
    if found:
        return Edition(found.start, minor, found.label, "text")
    if snapshot is not None:
        when = datetime.fromtimestamp(snapshot, UTC)
        label = f"{calendar.month_abbr[when.month]} {when.year}"
        return Edition(snapshot, minor, label, "snapshot")
    return Edition(None, minor)


def _refine_year(found: Edition, head: str) -> Edition:
    """A bare year ("ds221-...-2025") takes its month from the text.

    The address says only the year; the opening text ("Aug-Dec 2025")
    says which term, and so which academic year, the edition is for.
    """
    if not found.label.isdigit() or found.start is None:
        return found
    year = datetime.fromtimestamp(found.start, UTC).year
    text = _text_date(head)
    if text is None or text.start is None:
        return found
    if datetime.fromtimestamp(text.start, UTC).year != year:
        return found
    return Edition(text.start, found.minor, text.label, found.source)


def academic_year_label(start: float) -> str:
    """'2025-26' for any date from August 2025 to July 2026."""
    when = datetime.fromtimestamp(start, UTC)
    year = (
        when.year if when.month >= ACADEMIC_YEAR_START_MONTH else when.year - 1
    )
    return f"{year}-{(year + 1) % 100:02d}"


def _strip_dates(text: str) -> str:
    text = _UPLOAD.sub("/", text.lower())
    for pattern in (_ACADEMIC, _MONTH_RANGE, _YEAR_MONTH, _MONTH_YEAR, _YEAR):
        text = pattern.sub(" ", text)
    text = _NOISE.sub(" ", text)
    text = re.sub(r"[-_ .]\d{1,2}(?=$|/)", "", text)  # WordPress "-2" copies
    text = re.sub(r"[^a-z0-9/]+", "-", text)
    text = re.sub(r"-*/-*", "/", text)
    return re.sub(r"-+", "-", text).strip("-")


def family_key(url: str, is_file: bool) -> str:
    """The address with dates and version words removed.

    Files (PDF, Office) with a specific enough name are matched across
    sites (the same handbook is posted by several departments); web pages
    only within their own site. Pages named after a course code (ds221,
    DS256-jan20, ds221-introduction-to-scalable-systems-2025) are one family
    per course within their folder.

    Args:
        url: The document's original address.
        is_file: Whether it is a PDF or Office file rather than a page.

    Returns:
        The family key.
    """
    parts = urlparse(url)
    host = parts.netloc.lower().removeprefix("www.")
    path = unquote(parts.path).lower().rstrip("/")
    parent, _, last = path.rpartition("/")
    course = _COURSE.match(last)
    if course:  # /courses/ds221-introduction-...-2025/ and /courses/ds221/
        return f"{host}{_strip_dates(parent)}/course-{''.join(course.groups())}"
    if is_file:
        stem = _strip_dates(_file_name(url).rsplit(".", 1)[0])
        stem = stem.replace("/", "-").strip("-")
        if len(stem) >= MIN_FILE_STEM_CHARS and "-" in stem:
            return f"file:{stem}"
    path = re.sub(r"/index\.(?:html?|php)$", "/", unquote(parts.path))
    path = re.sub(r"/+", "/", _strip_dates(path)).rstrip("/")
    return f"{host}{path or '/'}"


@dataclass(frozen=True)
class Window:
    """When an edition is the one in force."""

    valid_from: float
    valid_to: float | None


def assign_windows(
    starts: Sequence[tuple[float, int]],
) -> list[Window]:
    """Validity windows for the editions of one family.

    Editions with the same (start, minor) are in force together; each
    later group retires the group before it.

    Args:
        starts: Per edition, its start time and minor revision.

    Returns:
        One window per edition, in the order given.
    """
    keys = sorted(set(starts))
    begins = [BEGINNING if i == 0 else k[0] for i, k in enumerate(keys)]
    # Same start, higher minor: the revision takes over a moment later.
    for i in range(1, len(keys)):
        if begins[i] <= begins[i - 1]:
            begins[i] = begins[i - 1] + 1.0
    ends: list[float | None] = [*begins[1:], None]
    position = {k: i for i, k in enumerate(keys)}
    return [Window(begins[position[s]], ends[position[s]]) for s in starts]
