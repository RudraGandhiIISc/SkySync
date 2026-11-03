"""Site template learning and removal (ingest/boilerplate)."""

from __future__ import annotations

from pathlib import Path

import pytest

from askiisc.common.errors import ExtractionError
from askiisc.ingest import boilerplate
from askiisc.ingest.pipeline import extract_and_chunk

MENU = "Home\nAbout\nPeople\nFaculty\nAcademics\nContact"
FOOTER = "Contact Info\nHow to Reach\n(c) 2026 Department"


def page(body: str) -> str:
    return f"{MENU}\n{body}\n{FOOTER}"


def bodies(n: int) -> list[str]:
    return [
        f"Page {i} title\nProfessor\nSome fact number {i} here."
        for i in range(n)
    ]


def test_learns_menu_and_footer_but_not_repeated_single_lines() -> None:
    pages = [boilerplate.shingles(page(b)) for b in bodies(10)]
    template = boilerplate.learn({"h": pages})["h"]
    cleaned = boilerplate.strip(page(bodies(1)[0]), template)
    assert cleaned == "Page 0 title\nProfessor\nSome fact number 0 here."


def test_too_few_pages_learn_nothing() -> None:
    pages = [boilerplate.shingles(page(b)) for b in bodies(3)]
    assert boilerplate.learn({"h": pages}) == {}


def test_share_threshold_scales_with_group_size() -> None:
    menu_pages = [boilerplate.shingles(page(b)) for b in bodies(5)]
    plain = [
        boilerplate.shingles(f"Unique {i}\nA {i}\nB {i}") for i in range(500)
    ]
    # 5 of 505 pages is under 2%: a section menu, not the site template.
    assert boilerplate.learn({"h": menu_pages + plain}) == {}


def test_strip_without_template_keeps_text() -> None:
    assert boilerplate.strip("a\n\nb", frozenset()) == "a\n\nb"


def test_whitespace_and_case_do_not_hide_template() -> None:
    pages = [boilerplate.shingles(page(b)) for b in bodies(6)]
    template = boilerplate.learn({"h": pages})["h"]
    noisy = page("Body").replace("Home", "  HOME ").replace("About", "about")
    assert boilerplate.strip(noisy, template) == "Body"


def test_group_of_live_and_archived() -> None:
    live = {"url": "https://www.physics.iisc.ac.in/academics/"}
    old = {
        "url": "https://web.archive.org/web/2024/x",
        "original_url": "https://www.physics.iisc.ac.in/academics/",
        "snapshot": 1718000000.0,
    }
    assert boilerplate.group_of(live) == "www.physics.iisc.ac.in"
    assert boilerplate.group_of(old) == "www.physics.iisc.ac.in@2024"


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "boilerplate.json"
    assert boilerplate.load(path) == {}
    boilerplate.save(path, {"h": frozenset({3, 1})})
    assert boilerplate.load(path) == {"h": frozenset({1, 3})}


def test_by_group_collects_pages() -> None:
    rows = [({"url": "https://a/x"}, [1]), ({"url": "https://a/y"}, [2])]
    assert boilerplate.by_group(rows) == {"a": [[1], [2]]}


def test_extract_and_chunk_removes_template_from_html_only() -> None:
    pages = [boilerplate.shingles(page(b)) for b in bodies(6)]
    template = boilerplate.learn({"h": pages})["h"]
    html = (
        "<html><body>"
        + "".join(
            f"<p>{line}</p>" for line in page("Real content.").splitlines()
        )
        + "</body></html>"
    )
    drafts = extract_and_chunk(html.encode(), "text/html", template)
    assert [d.text for d in drafts] == ["Real content."]
    text = page("Real content.").encode()
    plain = extract_and_chunk(text, "text/plain", template)
    assert "Home" in plain[0].text


def test_template_only_page_has_no_text() -> None:
    pages = [boilerplate.shingles(page(b)) for b in bodies(6)]
    template = boilerplate.learn({"h": pages})["h"]
    html = (
        "<html><body>"
        + "".join(
            f"<p>{line}</p>" for line in (MENU + "\n" + FOOTER).splitlines()
        )
        + "</body></html>"
    )
    with pytest.raises(ExtractionError, match="template"):
        extract_and_chunk(html.encode(), "text/html", template)
