import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from askiisc.ingest import minhash
from askiisc.ingest.catalog import build
from askiisc.ingest.editions import BEGINNING
from tests.helpers import make_pdf

RULES = " ".join(
    f"Rule {i}: students must register for {i + 3} credits each term."
    for i in range(60)
)
NEWS = " ".join(
    f"Seminar {i} on fluid dynamics was held today." for i in range(60)
)


def ts(year: int, month: int) -> float:
    return datetime(year, month, 1, tzinfo=UTC).timestamp()


def test_minhash_estimates_similarity() -> None:
    a = minhash.signature(RULES)
    assert minhash.similarity(a, minhash.signature(RULES)) == 1.0
    edited = RULES.replace("Rule 5:", "Rule five:")
    assert minhash.similarity(a, minhash.signature(edited)) > 0.9  # noqa: PLR2004
    assert minhash.similarity(a, minhash.signature(NEWS)) < 0.1  # noqa: PLR2004
    sigs = [a, minhash.signature(edited), minhash.signature(NEWS)]
    assert (0, 1) in minhash.candidate_pairs(sigs)
    assert (0, 2) not in minhash.candidate_pairs(sigs)
    assert minhash.signature("").dtype == np.uint64
    assert minhash.normalise_text("A-b  C") == "a b c"


def page(body: str, title: str = "Page") -> bytes:
    return (
        f"<html><head><title>{title}</title></head>"
        f"<body><p>{body}</p></body></html>"
    ).encode()


def write(
    root: Path, rows: list[tuple[str, str, bytes, dict]], name: str
) -> None:
    lines = []
    for doc_id, url, data, extra in rows:
        kind = "application/pdf" if url.endswith(".pdf") else "text/html"
        rel = f"public/x/{doc_id}.{'pdf' if kind.endswith('pdf') else 'html'}"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
        lines.append(
            json.dumps(
                {
                    "doc_id": doc_id,
                    "path": rel,
                    "url": url,
                    "category": "handbooks",
                    "department": "x",
                    "content_type": kind,
                    "fetched_at": ts(2026, 10),
                    "title": "" if kind.endswith("pdf") else "Page",
                    **extra,
                }
            )
        )
    (root / name).write_text("\n".join(lines) + "\n")


COURSE = " ".join(
    f"Week {i}: cloud computing lecture {i} and assignment {i}."
    for i in range(60)
)
TENDER = " ".join(
    f"Clause {i}: supply of a {{metal}} sputter target, delivery in {i} weeks."
    for i in range(30)
)
REPORT = "Monthly safety report for {month}. " + " ".join(
    f"Inspection {i} of laboratory {i} found no issues." for i in range(30)
)
DS25 = page("Aug-Dec 2025 offering. " + COURSE.replace("Week 7:", "Week 7a:"))


def handbook(year: int, extra: str = "") -> bytes:
    rules = RULES.replace("Rule 1:", f"Rule 1 ({year}):") + extra
    return make_pdf([rules[:1800], rules[1800:]])


def test_catalogue_removes_copies_and_orders_editions(tmp_path: Path) -> None:
    up = "https://iisc.ac.in/wp-content/uploads"
    live = [
        (
            "h23",
            f"{up}/2023/05/Student-Information-Handbook-2023.pdf",
            handbook(2023, " Hostel fee 10000."),
            {},
        ),
        (
            "h25",
            f"{up}/2025/10/Student-Information-Handbook-2025-V2.pdf",
            handbook(2025, " Hostel fee 12000."),
            {},
        ),
        (
            "h26",
            f"{up}/2026/07/Student-Information-Handbook-2026-August.pdf",
            handbook(2026, " Hostel fee 15000."),
            {},
        ),
        # The same 2026 handbook posted by a department: a copy.
        (
            "h26c",
            "https://caos.iisc.ac.in/wp-content/uploads/2026/07/"
            "Student-Information-Handbook-2026-August.pdf",
            handbook(2026, " Hostel fee 15000."),
            {},
        ),
        (
            "ds",
            "https://cds.iisc.ac.in/courses/DS252/",
            page("Aug-Dec 2026 offering. " + COURSE),
            {},
        ),
        # Almost the same text, but another edition: never a copy.
        ("ds25", "https://cds.iisc.ac.in/courses/ds252-aug25/", DS25, {}),
        ("news", "http://www.iisc.ac.in/news/seminar/", page(NEWS), {}),
        ("newsw", "https://iisc.ac.in/news/seminar/", page(NEWS + " "), {}),
        ("empty", "https://x.iisc.ac.in/empty/", b"<html></html>", {}),
        # Two tenders differing in one word: two documents, not copies.
        (
            "gold",
            "https://iisc.ac.in/wp-content/uploads/2021/07/Gold-sputter.pdf",
            make_pdf([TENDER.format(metal="Gold")]),
            {},
        ),
        (
            "platinum",
            "https://iisc.ac.in/wp-content/uploads/2021/07/Platinum-sputter.pdf",
            make_pdf([TENDER.format(metal="Platinum")]),
            {},
        ),
        # A monthly series is not a set of editions.
        (
            "feb",
            "https://olseh.iisc.ac.in/uploads/2024/06/February-2024-report.pdf",
            make_pdf([REPORT.format(month="February")]),
            {},
        ),
        (
            "mar",
            "https://olseh.iisc.ac.in/uploads/2024/06/March-2024-report.pdf",
            make_pdf([REPORT.format(month="March")]),
            {},
        ),
    ]
    write(tmp_path, live, "manifest.jsonl")
    archived = [
        (
            "snap",
            "https://web.archive.org/web/20251101000000/"
            "https://cds.iisc.ac.in/courses/DS252/",
            DS25,
            {
                "original_url": "https://cds.iisc.ac.in/courses/DS252/",
                "snapshot": ts(2025, 11),
            },
        ),
    ]
    write(tmp_path, archived, "archive.jsonl")

    summary = build(tmp_path, workers=1)
    rows = {
        r["doc_id"]: r
        for r in map(json.loads, (tmp_path / "catalog.jsonl").open())
    }
    dropped = {
        d["doc_id"]: d["kept"]
        for d in map(json.loads, (tmp_path / "duplicates.jsonl").open())
    }
    # Copies: the department's handbook, the www/http page, the snapshot
    # equal to the live 2025 course page.
    assert dropped == {"h26c": "h26", "news": "newsw", "snap": "ds25"}
    assert any("caos" in a for a in rows["h26"]["aliases"])
    assert set(rows) == {
        "h23",
        "h25",
        "h26",
        "ds",
        "ds25",
        "newsw",
        "gold",
        "platinum",
        "feb",
        "mar",
    }
    assert rows["feb"]["editions"] == rows["mar"]["editions"] == 1
    assert rows["feb"]["valid_to"] is None
    assert summary["duplicates_dropped"] == 3  # noqa: PLR2004

    family = {rows[d]["family"] for d in ("h23", "h25", "h26")}
    assert family == {"h26"}
    assert rows["h23"]["valid_from"] == BEGINNING
    assert rows["h23"]["valid_to"] == rows["h25"]["valid_from"]
    assert rows["h25"]["valid_to"] == rows["h26"]["valid_from"] == ts(2026, 8)
    assert rows["h26"]["valid_to"] is None and rows["h26"]["editions"] == 3  # noqa: PLR2004

    assert rows["ds"]["family"] == rows["ds25"]["family"] == "ds"
    assert rows["ds25"]["valid_to"] == rows["ds"]["valid_from"] == ts(2026, 8)
    assert rows["ds"]["edition"] == "Aug 2026"
    assert rows["newsw"]["editions"] == 1 and rows["newsw"]["valid_to"] is None
    assert summary["families_with_several_editions"] == 2  # noqa: PLR2004
    assert rows["ds"]["title"] == "Page"


def menu_page(body: str) -> bytes:
    words = ["Home", "About", "People", "Contact"]
    menu = "".join(f"<p>{w}</p>" for w in words)
    return f"<html><body>{menu}<p>{body}</p>{menu}</body></html>".encode()


def test_catalogue_removes_site_template(tmp_path: Path) -> None:
    host = "https://www.physics.iisc.ac.in"
    live = [
        (f"p{i}", f"{host}/p{i}/", menu_page(f"Distinct fact {i}. " * 9), {})
        for i in range(6)
    ]
    live.append(("hub", f"{host}/academics/", menu_page(""), {}))
    write(tmp_path, live, "manifest.jsonl")
    build(tmp_path, workers=1)
    rows = {
        r["doc_id"]: r
        for r in map(json.loads, (tmp_path / "catalog.jsonl").open())
    }
    assert set(rows) == {f"p{i}" for i in range(6)}  # the menu-only hub
    templates = json.loads((tmp_path / "boilerplate.json").read_text())
    assert list(templates) == ["www.physics.iisc.ac.in"]
