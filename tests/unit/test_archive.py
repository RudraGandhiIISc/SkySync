import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from askiisc.ingest.archive import (
    ArchiveConfig,
    ArchiveFetcher,
    Snapshot,
    academic_year,
    choose_snapshots,
    parse_cdx,
    read_manifests,
    targets,
    wayback_digest,
)

LIVE = (
    b"<html><body>" + b"Curriculum 2026 core courses. " * 30 + b"</body></html>"
)
OLD = (
    b"<html><body>" + b"Curriculum 2024 core courses. " * 30 + b"</body></html>"
)
OLDER = (
    b"<html><body>" + b"Curriculum 2023 core courses. " * 30 + b"</body></html>"
)
URL = "https://cds.iisc.ac.in/academics/curriculum/"


class Sleeper:
    def __init__(self) -> None:
        self.waits: list[float] = []
        self.now = 0.0

    async def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.now += seconds

    def clock(self) -> float:
        return self.now


def cdx(rows: list[tuple[str, bytes]]) -> list[list[str]]:
    header = ["timestamp", "digest", "mimetype", "statuscode"]
    body = [[t, wayback_digest(d), "text/html", "200"] for t, d in rows]
    return [header, *body, ["20250101000000", "X", "text/html", "404"]]


def test_years_digests_and_choice() -> None:
    assert academic_year(datetime(2025, 7, 31, tzinfo=UTC)) == 2024  # noqa: PLR2004
    assert academic_year(datetime(2025, 8, 1, tzinfo=UTC)) == 2025  # noqa: PLR2004
    snaps = parse_cdx(
        cdx(
            [
                ("20230901000000", OLDER),
                ("20240301000000", OLDER),  # same year, later: kept
                ("20241001000000", OLD),
                ("20250501000000", LIVE),  # same as the live copy
                ("20260901000000", OLD),  # the current year: live covers it
            ]
        )
    )
    assert len(snaps) == 5  # noqa: PLR2004 - the 404 row is dropped
    config = ArchiveConfig(first_year=2022, current_year=2026)
    chosen = choose_snapshots(snaps, config, {wayback_digest(LIVE)})
    assert [s.timestamp for s in chosen] == ["20240301000000"]
    # Without the live copy, 2024-25 keeps its last snapshot too.
    chosen = choose_snapshots(snaps, config, set())
    assert [s.timestamp[:4] for s in chosen] == ["2024", "2025"]
    assert parse_cdx([]) == []
    assert Snapshot("20251101000000", "", "").when.month == 11  # noqa: PLR2004


def test_targets_keep_academic_pages_first() -> None:
    rows = [
        {"url": "https://x.iisc.ac.in/news/a/", "title": ""},
        {"url": "https://x.iisc.ac.in/admissions/", "title": ""},
        {"url": "https://x.iisc.ac.in/courses/", "title": ""},
        {"url": "http://x.iisc.ac.in/courses", "title": ""},  # same page
        {"url": "https://x.iisc.ac.in/handbook/", "title": "Student Handbook"},
        {"url": "https://x.iisc.ac.in/~me/courses/lec1.pdf", "title": ""},
        {"url": "https://x.iisc.ac.in/phd/thesis.pdf", "title": ""},
        {"url": "https://x.iisc.ac.in/a/", "title": "Student Handbook"},
    ]
    picked = [str(r["url"]) for r in targets(rows)]
    assert picked == [
        "https://x.iisc.ac.in/courses/",
        "https://x.iisc.ac.in/handbook/",
        "https://x.iisc.ac.in/admissions/",
    ]


async def test_fetcher_saves_past_editions_and_resumes(tmp_path: Path) -> None:
    calls: list[str] = []
    busy = {"left": 1}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path == "/cdx/search/cdx":
            if busy["left"]:
                busy["left"] -= 1
                return httpx.Response(429)
            rows = [("20231001000000", OLDER), ("20241001000000", OLD)]
            return httpx.Response(200, json=cdx(rows))
        body = OLDER if "2023" in request.url.path else OLD
        if "short" in str(request.url):
            body = b"<html>tiny</html>"
        return httpx.Response(
            200, headers={"content-type": "text/html"}, content=body
        )

    sleeper = Sleeper()
    live = tmp_path / "live.html"
    live.write_bytes(LIVE)
    rows = [
        {
            "url": URL,
            "content_type": "text/html",
            "category": "course-catalogue",
            "live_path": str(live),
        },
        {"url": URL + "short/", "content_type": "text/html"},
    ]
    config = ArchiveConfig(first_year=2022, current_year=2026)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    fetcher = ArchiveFetcher(
        config, client, tmp_path, sleeper.sleep, sleeper.clock, lambda: 1.0
    )
    report = await fetcher.run(rows)
    assert report.saved == 2 and report.too_little_text == 2  # noqa: PLR2004
    assert report.by_year == {"2023": 1, "2024": 1}
    assert max(sleeper.waits) >= config.delay_s  # backed off after the 429
    saved = [json.loads(ln) for ln in (tmp_path / "archive.jsonl").open()]
    assert {s["original_url"] for s in saved} == {URL}
    first = saved[0]
    assert first["url"].startswith("https://web.archive.org/web/2023")
    assert "id_" in calls[2]  # raw bytes, not the archive's page frame
    assert (tmp_path / first["path"]).read_bytes() == OLDER
    assert first["snapshot"] == datetime(2023, 10, 1, tzinfo=UTC).timestamp()
    fetcher.checkpoint()
    again = ArchiveFetcher(config, client, tmp_path, sleeper.sleep)
    again.load()
    before = len(calls)
    assert (await again.run(rows)).saved == 0
    assert len(calls) == before  # both addresses were done


async def test_deadline_and_errors(tmp_path: Path) -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    sleeper = Sleeper()
    client = httpx.AsyncClient(transport=httpx.MockTransport(broken))
    rows = [{"url": URL, "content_type": "text/html"}]
    fetcher = ArchiveFetcher(
        ArchiveConfig(), client, tmp_path, sleeper.sleep, sleeper.clock
    )
    assert (await fetcher.run(rows)).errors == 1
    late = ArchiveFetcher(
        ArchiveConfig(deadline_s=0.0),
        client,
        tmp_path,
        sleeper.sleep,
        sleeper.clock,
    )
    assert (await late.run(rows)).stopped_by == "deadline"


def test_read_manifests(tmp_path: Path) -> None:
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps({"url": URL, "path": "public/a.html"}) + "\n"
    )
    rows = read_manifests([tmp_path / "manifest.jsonl", tmp_path / "none"])
    assert rows[0]["live_path"] == str(tmp_path / "public/a.html")
