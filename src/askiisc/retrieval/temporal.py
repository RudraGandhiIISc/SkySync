"""Which time a question is about, so the right edition is searched.

Documents that change every academic year (curricula, handbooks, course
pages) are kept in every edition, each valid for its own period. The
search runs "as of" an instant and sees only the edition in force then.

- No time words, or "this year", "current", "latest": now (newest edition).
- "2025", "2025-26", "the 2025 batch": the academic year that starts in
  2025, searched at its start (1 October 2025, the August term) and its
  end (1 July 2026, after the January term), so a course offered in
  January 2026 is found as the 2025-26 offering.
- "January 2025", "Aug 2025": that month.
- "last year", "previous year": one year ago.
- "what changed", "compared to", "difference" with one time or none: also
  the year before, so both editions are retrieved.
- Several years ("2024 and 2025"): each of them (at most MAX_TIME_SCOPES).

The search text drops the time phrase ("For the 2025-26 academic year,"):
the as-of filter already applies it, and its words only pull in chunks
that mention a year (3% more year questions found their fact: job 1734).
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from askiisc.common.targets import MAX_TIME_SCOPES

ACADEMIC_YEAR_START_MONTH = 8
# The instant used for an academic year: well inside August-July.
ACADEMIC_YEAR_PROBE_MONTH = 10
ACADEMIC_YEAR_END_PROBE_MONTH = 7
EARLIEST_YEAR = 2010
YEAR = timedelta(days=365)

_MONTHS = {n.lower(): i for i, n in enumerate(calendar.month_abbr) if n}
_MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_ACADEMIC = re.compile(r"(?<!\d)(20\d\d)\s*[-–/]\s*(?:20)?(\d\d)(?!\d)")
_MONTH_YEAR = re.compile(rf"(?<![a-z]){_MONTH}\s*,?\s*(20\d\d)(?!\d)")
# A year, unless it is an amount ("Rs 2000", "2025 rupees") or a code.
_YEAR = re.compile(
    r"(?<![\d₹$.])(?<!rs )(?<!rs\. )(?<!inr )(20\d\d)"
    r"(?!\d|,\d|\s*(?:rupees|rs\b|inr\b|/-))"
)
_PAST = re.compile(
    r"\b(last|previous|past|prior)\s+(academic\s+)?(year|session)\b"
)
_COMPARE = re.compile(
    r"\b(chang\w*|differ\w*|compar\w*|vs\.?|versus|revis\w*)\b"
)


# "for the 2025-26 academic year", "in 2025", "during academic year
# 2024-25", "in August 2025", "this year", "last academic year".
_TIME_PHRASE = re.compile(
    rf"\b(?:for|in|during)\s+(?:the\s+)?(?:academic\s+year\s+)?"
    rf"(?:{_MONTH}\s+)?20\d\d(?:\s*[-–/]\s*(?:20)?\d\d)?"
    rf"(?:\s+academic\s+year)?\s*,?"
    rf"|\b(?:this|last|previous|current)\s+(?:academic\s+)?year\b",
    re.IGNORECASE,
)

_SPACE_BEFORE_MARK = re.compile(r"\s+([?.,!])")


def without_time(question: str) -> str:
    """The question without its time phrase, for the search text.

    Returns:
        The question with "for the 2025-26 academic year", "in 2025",
        "this year" and the like removed (unchanged if nothing is left).
    """
    text = " ".join(_TIME_PHRASE.sub(" ", question).split())
    text = _SPACE_BEFORE_MARK.sub(r"\1", text)
    return question if not text.strip("?.,!") else text


@dataclass(frozen=True)
class TimeScope:
    """The as-of instants to search and a label for the prompt.

    Attributes:
        instants: Times to search, oldest first; one for most questions.
        label: What the question asks about ("" when it names no time).
    """

    instants: tuple[float, ...]
    label: str = ""


def _instant(year: int, month: int, day: int = 1) -> float:
    return datetime(year, month, day, tzinfo=UTC).timestamp()


def _academic_label(year: int) -> str:
    return f"academic year {year}-{(year + 1) % 100:02d}"


def _named_times(text: str, now: datetime) -> list[tuple[float, str]]:
    """Explicit years and months in the question."""
    found: list[tuple[float, str]] = []
    rest = text
    for match in _ACADEMIC.finditer(text):
        first, second = int(match.group(1)), int(match.group(2))
        if second == (first + 1) % 100:
            found += _academic_year(first)
            rest = rest.replace(match.group(0), " ")
    for match in _MONTH_YEAR.finditer(rest):
        year, month = int(match.group(2)), _MONTHS[match.group(1)[:3]]
        found.append(
            (_instant(year, month, 15), f"{calendar.month_name[month]} {year}")
        )
        rest = rest.replace(match.group(0), " ")
    for match in _YEAR.finditer(rest):
        year = int(match.group(1))
        if EARLIEST_YEAR <= year <= now.year + 1:
            found += _academic_year(year)
    return found


def _academic_year(year: int) -> list[tuple[float, str]]:
    """Both probes of an academic year, with its label."""
    label = _academic_label(year)
    return [
        (_instant(year, ACADEMIC_YEAR_PROBE_MONTH), label),
        (_instant(year + 1, ACADEMIC_YEAR_END_PROBE_MONTH), label),
    ]


def time_scope(question: str, now: float) -> TimeScope:
    """Finds the time or times a question is about.

    Args:
        question: The question (any case).
        now: The current time.

    Returns:
        The instants to search; just `now` when no time is named.
    """
    text = question.lower()
    today = datetime.fromtimestamp(now, UTC)
    found = _named_times(text, today)
    past = bool(_PAST.search(text))
    if past:
        found.append((now - YEAR.total_seconds(), "the previous year"))
    named = {label for _, label in found}
    if _COMPARE.search(text) and len(named) < 2:  # noqa: PLR2004
        if past and len(named) == 1:
            # "changed compared to last year": last year against now.
            found.append((now, "the current year"))
        else:
            if not found:
                found.append((now, "the current year"))
            earlier = min(f[0] for f in found) - YEAR.total_seconds()
            found.append((earlier, "the year before"))
    if not found:
        return TimeScope((now,))
    unique = sorted({min(t, now): label for t, label in found}.items())
    unique = unique[-MAX_TIME_SCOPES:]
    labels = list(dict.fromkeys(label for _, label in unique))
    return TimeScope(tuple(t for t, _ in unique), " and ".join(labels))
