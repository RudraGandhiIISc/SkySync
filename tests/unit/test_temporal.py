from datetime import UTC, datetime

import pytest

from askiisc.retrieval.temporal import time_scope, without_time

NOW = datetime(2026, 10, 26, tzinfo=UTC).timestamp()


def day(year: int, month: int, d: int = 1) -> float:
    return datetime(year, month, d, tzinfo=UTC).timestamp()


@pytest.mark.parametrize(
    "question",
    [
        "What is the course structure of the cloud computing course?",
        "Cloud computing course structure this year",
        "What is the latest M.Tech CDS curriculum?",
        "Is the fee Rs 2000 or 20000?",
        "How many credits does DS252 carry?",
    ],
)
def test_no_time_means_now(question: str) -> None:
    scope = time_scope(question, NOW)
    assert scope.instants == (NOW,) and scope.label == ""


def ay(year: int) -> tuple[float, float]:
    """The two probes of an academic year: October and the next July."""
    return (day(year, 10), day(year + 1, 7))


def test_years_mean_their_academic_year() -> None:
    scope = time_scope("Cloud computing course structure for 2025", NOW)
    assert scope.instants == ay(2025)
    assert scope.label == "academic year 2025-26"
    assert time_scope("CDS curriculum 2024-25", NOW).instants == ay(2024)
    assert time_scope("rules for the 2023 batch", NOW).instants == ay(2023)
    assert time_scope("DS252 in 2025-2026", NOW).instants == ay(2025)
    # The current academic year's end is in the future: searched as now.
    assert time_scope("DS252 in 2026-27", NOW).instants == (day(2026, 10), NOW)


def test_months_and_future_years() -> None:
    scope = time_scope("UG calendar for January 2025", NOW)
    assert (
        scope.instants == (day(2025, 1, 15),) and "January 2025" in scope.label
    )
    assert time_scope("Aug 2026 timetable", NOW).instants == (day(2026, 8, 15),)
    # Next year's edition does not exist yet: the newest one is searched.
    assert time_scope("fees for 2027", NOW).instants == (NOW,)
    assert time_scope("history since 1998", NOW).instants == (NOW,)


def test_last_year_and_comparisons() -> None:
    year = 365 * 86400.0
    assert time_scope("last year's DS252 syllabus", NOW).instants == (
        NOW - year,
    )
    changed = time_scope("What changed in the CDS curriculum?", NOW)
    assert changed.instants == (NOW - year, NOW)
    compare = time_scope("Compare the 2024 and 2025 handbooks", NOW)
    assert compare.instants == (*ay(2024), *ay(2025))
    assert compare.label == "academic year 2024-25 and academic year 2025-26"
    since = time_scope("How is the 2025 handbook different?", NOW)
    assert since.instants == (day(2024, 10), *ay(2025))
    many = time_scope("handbooks of 2019, 2020, 2021 and 2022", NOW)
    assert many.instants == (*ay(2021), *ay(2022))  # the latest four


def test_change_since_last_year_compares_with_now() -> None:
    # Regression: "compared to last year" searched last year and the year
    # before it, so the current DS252 edition was never retrieved.
    year = 365 * 86400.0
    scope = time_scope(
        "What changed in the cloud computing course compared to last year?",
        NOW,
    )
    assert scope.instants == (NOW - year, NOW)
    assert scope.label == "the previous year and the current year"


@pytest.mark.parametrize(
    ("question", "text"),
    [
        (
            "For the 2025-26 academic year, who teaches DS252?",
            "who teaches DS252?",
        ),
        (
            "What was the DS252 syllabus in 2025?",
            "What was the DS252 syllabus?",
        ),
        ("Hostel fee during academic year 2024-25", "Hostel fee"),
        ("Who taught DS252 in August 2025?", "Who taught DS252?"),
        ("DS252 course structure this year", "DS252 course structure"),
        ("What changed since last academic year?", "What changed since?"),
        ("Is the fee Rs 2000?", "Is the fee Rs 2000?"),
        ("in 2025", "in 2025"),  # nothing would be left
    ],
)
def test_search_text_drops_the_time_phrase(question: str, text: str) -> None:
    assert without_time(question) == text
