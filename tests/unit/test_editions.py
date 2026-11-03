from datetime import UTC, datetime

from askiisc.ingest.editions import (
    BEGINNING,
    academic_year_label,
    assign_windows,
    detect_edition,
    family_key,
)


def ts(year: int, month: int) -> float:
    return datetime(year, month, 1, tzinfo=UTC).timestamp()


HANDBOOKS = (
    "https://iisc.ac.in/wp-content/uploads/2023/05/"
    "Student-Information-Handbook-2023.pdf",
    "https://iisc.ac.in/wp-content/uploads/2024/01/"
    "Student-Information-Handbook-2024.pdf",
    "http://www.civil.iisc.ac.in/wp/wp-content/uploads/2024/07/"
    "Student-Information-Handbook-2024-V2.pdf",
    "https://iisc.ac.in/wp-content/uploads/2025/10/"
    "Student-Information-Handbook-2025-V2.pdf",
    "https://caos.iisc.ac.in/wp-content/uploads/2026/07/"
    "Student-Information-Handbook-2026-August-Version.pdf",
)


def test_dates_from_addresses() -> None:
    first, second, v2, _, august = (detect_edition(u) for u in HANDBOOKS)
    # A bare year in the name; the upload month refines it.
    assert first.start == ts(2023, 5) and first.label == "2023"
    assert second.start == ts(2024, 1) and second.source == "url"
    assert v2.start == ts(2024, 7) and v2.minor == 2  # noqa: PLR2004
    assert august.start == ts(2026, 8) and august.label == "Aug 2026"
    aug25 = detect_edition("https://cds.iisc.ac.in/courses/ds252-aug25/")
    assert aug25.start == ts(2025, 8) and aug25.label == "Aug 2025"
    scheme = detect_edition(
        "https://cds.iisc.ac.in/scheme-of-instructions-to-students-2014-15/"
    )
    assert scheme.start == ts(2014, 8) and scheme.label == "2014-15"
    calendar = detect_edition(
        "https://bs-ug.iisc.ac.in/assets/UG%20Academic%20Calendar%20"
        "Jan-Apr%202026%20(2).pdf"
    )
    assert calendar.start == ts(2026, 1)


def test_dates_from_title_upload_text_and_snapshot() -> None:
    page = "https://kiac.iisc.ac.in/academics/curriculum/"
    titled = detect_edition(page, title="M.Tech AI curriculum 2023-25 batch")
    assert titled.source == "title" and titled.start == ts(2023, 1)
    upload = detect_edition(
        "https://x.iisc.ac.in/wp-content/uploads/2025/03/a.pdf"
    )
    assert upload.source == "upload" and upload.start == ts(2025, 3)
    text = detect_edition(page, head="Course plan for August-December 2025")
    assert text.source == "text" and text.start == ts(2025, 8)
    academic = detect_edition(
        page, head="Curriculum for the academic year 2024"
    )
    assert academic.start == ts(2024, 8)
    copyright_only = detect_edition(page, head="(c) 2026 IISc. All rights.")
    assert copyright_only.start is None  # bare years in text are ignored
    snap = ts(2025, 11)
    archived = detect_edition(page, snapshot=snap)
    assert archived.source == "snapshot" and archived.label == "Nov 2025"
    assert detect_edition(page).start is None


def test_years_out_of_range_and_non_consecutive_pairs() -> None:
    assert detect_edition("https://x.iisc.ac.in/report-1998.pdf").start is None
    # 2026/06 is a blog path, not the academic year 2026-27.
    blog = detect_edition("https://connect.iisc.ac.in/2026/06/cloudy-trail/")
    assert blog.label == "2026"


def test_family_keys_join_editions_only() -> None:
    keys = {family_key(u, is_file=True) for u in HANDBOOKS}
    assert keys == {"file:student-information-handbook"}
    assert family_key("https://cds.iisc.ac.in/courses/DS252/", False) == (
        family_key("https://cds.iisc.ac.in/courses/ds252-aug25/", False)
    )
    assert family_key("https://cds.iisc.ac.in/courses/ds256/", False) != (
        family_key("https://cds.iisc.ac.in/courses/ds252/", False)
    )
    # Generic names are matched within their own site only.
    a = family_key("https://a.iisc.ac.in/brochure-2024.pdf", True)
    b = family_key("https://b.iisc.ac.in/brochure-2025.pdf", True)
    assert a != b and a.startswith("a.iisc.ac.in")
    assert family_key("https://www.iisc.ac.in/academics/index.html", False) == (
        family_key("https://iisc.ac.in/academics/", False)
    )


def test_windows_follow_edition_order() -> None:
    starts = [(ts(2025, 8), 0), (ts(2023, 1), 0), (ts(2024, 1), 0)]
    windows = assign_windows(starts)
    assert windows[1].valid_from == BEGINNING  # oldest covers all before
    assert windows[1].valid_to == ts(2024, 1)
    assert windows[2].valid_from == ts(2024, 1)
    assert windows[2].valid_to == ts(2025, 8)
    assert windows[0].valid_from == ts(2025, 8) and windows[0].valid_to is None


def test_windows_for_revisions_and_copies() -> None:
    same = assign_windows([(ts(2024, 1), 0), (ts(2024, 1), 0)])
    assert same[0] == same[1] and same[0].valid_to is None
    revised = assign_windows([(ts(2024, 7), 2), (ts(2024, 7), 0)])
    assert revised[1].valid_to == revised[0].valid_from
    assert revised[0].valid_to is None
    assert revised[0].valid_from > revised[1].valid_from
    first_v2 = assign_windows([(BEGINNING, 2), (BEGINNING, 0)])
    assert first_v2[0].valid_from > first_v2[1].valid_from
    assert assign_windows([(ts(2024, 1), 0)])[0].valid_from == BEGINNING


def test_course_pages_form_one_family_per_course() -> None:
    urls = (
        "https://cds.iisc.ac.in/courses/ds221/",
        "https://cds.iisc.ac.in/courses/ds221-2022/",
        "https://cds.iisc.ac.in/courses/ds221-introduction-to-scalable-systems-2025/",
    )
    assert len({family_key(u, False) for u in urls}) == 1
    assert family_key(urls[0], False) != family_key(
        "https://cds.iisc.ac.in/courses/ds256-jan20/", False
    )
    assert family_key("https://cds.iisc.ac.in/courses/DS256-jan20/", False) == (
        family_key("https://cds.iisc.ac.in/courses/ds256-jan21/", False)
    )
    # Year-coded file names are not course codes.
    assert "course-" not in family_key(
        "https://cds.iisc.ac.in/wp-content/uploads/SOI_2016-17.pdf", True
    )


def test_bare_year_takes_its_term_from_the_text() -> None:
    url = "https://cds.iisc.ac.in/courses/ds221-introduction-to-scalable-systems-2025/"
    aug = detect_edition(url, head="DS221 Aug-Dec 2025 offering. Lectures ...")
    assert aug.start == ts(2025, 8) and aug.label == "Aug 2025"
    other = detect_edition(url, head="Archived from Jan 2019 ...")
    assert other.start == ts(2025, 1) and other.label == "2025"
    assert academic_year_label(ts(2026, 1)) == "2025-26"
    assert academic_year_label(ts(2025, 8)) == "2025-26"
