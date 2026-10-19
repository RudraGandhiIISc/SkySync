import json
from pathlib import Path

import httpx

from askiisc.ingest.crawl import (
    CrawlConfig,
    Crawler,
    categorise,
    doc_id_for,
    normalise_url,
)
from askiisc.retrieval.store import check_id
from tests.helpers import make_pdf

LONG = " ".join(["Admission eligibility rules for the programme."] * 20)


def html(body: str, title: str = "Page") -> bytes:
    head = f"<head><title>{title}</title></head>"
    return f"<html>{head}<body>{body}</body></html>".encode()


ADMISSIONS = html(
    f"<p>{LONG}</p><a href='/admissions/phd.pdf'>PhD</a>"
    "<a href='/sports/'>Sports</a><a href='/admissions/copy/'>copy</a>"
    "<a href='/admissions/short/'>short</a><a href='/img.png'>i</a>"
    "<a href='https://elsewhere.org/'>x</a>"
    "<a href='/admissions/scan.pdf'>scan</a>"
    "<a href='/admissions/gone/'>gone</a>"
    "<a href='/admissions/timeout/'>slow</a>",
    "Admissions",
)

SITE: dict[str, tuple[int, str, bytes]] = {
    "https://www.iisc.ac.in/robots.txt": (200, "text/plain", b""),
    "https://www.iisc.ac.in/admissions/": (200, "text/html", ADMISSIONS),
    "https://www.iisc.ac.in/admissions/phd.pdf": (
        200,
        "application/pdf",
        make_pdf([LONG[:300], LONG[:300]]),
    ),
    "https://www.iisc.ac.in/sports/": (
        200,
        "text/html",
        html(f"<p>{LONG}</p>", "Sports"),
    ),
    "https://www.iisc.ac.in/admissions/copy/": (200, "text/html", ADMISSIONS),
    "https://www.iisc.ac.in/admissions/short/": (
        200,
        "text/html",
        html("<p>tiny</p>"),
    ),
    "https://www.iisc.ac.in/admissions/scan.pdf": (
        200,
        "application/pdf",
        make_pdf([""]),
    ),
    "https://www.iisc.ac.in/admissions/gone/": (404, "text/html", b""),
    "https://cds.iisc.ac.in/robots.txt": (
        200,
        "text/plain",
        b"User-agent: *\nDisallow: /private/\n",
    ),
    "https://cds.iisc.ac.in/": (
        200,
        "text/html",
        html(
            "<p>"
            + "Department of Computational and Data Sciences news. " * 20
            + "</p><a href='/private/x/'>p</a><a href='/people/'>people</a>"
        ),
    ),
    "https://cds.iisc.ac.in/people/": (
        200,
        "text/html",
        html(
            "<p>" + "Faculty members and research groups of CDS. " * 20 + "</p>"
        ),
    ),
    "https://www.csa.iisc.ac.in/robots.txt": (403, "text/html", b"no"),
    "https://www.csa.iisc.ac.in/": (
        200,
        "text/html",
        html("<p>" + "Computer Science and Automation courses. " * 20 + "</p>"),
    ),
}


def _transport(calls: list[str]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        assert request.headers["user-agent"].startswith("AskIIScBot")
        if "down.iisc" in url:
            return httpx.Response(503)
        if "broken.iisc" in url:
            raise httpx.ConnectError("no route")
        if url.endswith("/admissions/timeout/"):
            raise httpx.ReadTimeout("slow")
        status, kind, body = SITE.get(url, (404, "text/html", b""))
        return httpx.Response(
            status, content=body, headers={"content-type": kind, "etag": '"v1"'}
        )

    return httpx.MockTransport(handler)


CONFIG = CrawlConfig(
    seeds=(
        "https://www.iisc.ac.in/admissions/",
        "https://cds.iisc.ac.in/",
        "https://www.csa.iisc.ac.in/",
        "https://down.iisc.ac.in/",
        "https://broken.iisc.ac.in/",
    ),
    departments={
        "cds.iisc.ac.in": "cds",
        "www.csa.iisc.ac.in": "csa",
        "down.iisc.ac.in": "x",
        "broken.iisc.ac.in": "y",
    },
    topical_hosts=frozenset({"www.iisc.ac.in"}),
)


def _crawl(
    tmp_path: Path, config: CrawlConfig = CONFIG
) -> tuple[Crawler, list[str], list[float]]:
    calls: list[str] = []
    sleeps: list[float] = []
    now = [0.0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    crawler = Crawler(
        config,
        httpx.Client(transport=_transport(calls)),
        tmp_path,
        sleep=sleep,
        clock=lambda: now[0],
    )
    crawler.run()
    return crawler, calls, sleeps


def test_crawl_saves_on_topic_documents_with_text(tmp_path: Path) -> None:
    crawler, calls, _ = _crawl(tmp_path)
    report = crawler.report
    saved = {e.url: e for e in crawler.manifest}
    assert set(saved) == {
        "https://www.iisc.ac.in/admissions/",
        "https://www.iisc.ac.in/admissions/phd.pdf",
        "https://cds.iisc.ac.in/",
        "https://cds.iisc.ac.in/people/",
        "https://www.csa.iisc.ac.in/",
    }
    assert report.off_topic == 1  # /sports/ on the topical host
    assert report.duplicates == 1  # /admissions/copy/ has the same bytes
    assert report.too_little_text == 1
    assert report.no_text_layer == 1
    # cds /private/, the host whose robots.txt is 503, the unreachable host
    assert report.blocked_by_robots == 3
    assert report.errors == 1
    assert "https://cds.iisc.ac.in/private/x/" not in calls
    assert not any("elsewhere.org" in c or c.endswith(".png") for c in calls)
    pdf = saved["https://www.iisc.ac.in/admissions/phd.pdf"]
    assert pdf.category == "admissions" and pdf.department == "institute"
    assert pdf.visibility == "public" and pdf.etag == '"v1"'
    assert (tmp_path / pdf.path).read_bytes()[:4] == b"%PDF"
    assert pdf.path.startswith("public/institute/admissions/")
    assert saved["https://cds.iisc.ac.in/people/"].path.startswith(
        "public/cds/department/"
    )
    for entry in crawler.manifest:
        check_id(entry.doc_id)


def test_politeness_delay_per_host(tmp_path: Path) -> None:
    _, calls, sleeps = _crawl(tmp_path)
    assert sleeps and all(0 < s <= CONFIG.delay_s for s in sleeps)
    assert len(calls) > len(sleeps)  # first request to each host waits none


def test_manifest_round_trip_and_limits(tmp_path: Path) -> None:
    config = CrawlConfig(
        seeds=CONFIG.seeds,
        departments=CONFIG.departments,
        topical_hosts=CONFIG.topical_hosts,
        max_docs=2,
    )
    crawler, _, _ = _crawl(tmp_path, config)
    assert crawler.report.saved == 2
    lines = crawler.write_manifest().read_text().splitlines()
    assert [json.loads(ln)["doc_id"] for ln in lines] == [
        e.doc_id for e in crawler.manifest
    ]
    capped = CrawlConfig(
        seeds=CONFIG.seeds,
        departments=CONFIG.departments,
        topical_hosts=CONFIG.topical_hosts,
        max_requests=1,
    )
    assert _crawl(tmp_path, capped)[0].report.requests <= 2


def test_helpers() -> None:
    assert (
        normalise_url("https://WWW.IISC.ac.in/a?b=1#c")
        == "https://www.iisc.ac.in/a"
    )
    assert normalise_url("https://cds.iisc.ac.in") == "https://cds.iisc.ac.in/"
    assert categorise("https://x/admissions/phd") == "admissions"
    assert categorise("https://x/a", "Student Handbook 2026") == "handbooks"
    assert categorise("https://x/courses/") == "course-catalogue"
    assert categorise("https://x/sports/") is None
    first = doc_id_for("https://www.iisc.ac.in/admissions/")
    assert first.startswith("iisc-admissions-") and first != doc_id_for(
        "https://www.iisc.ac.in/admissions"
    )
    assert doc_id_for("https://cds.iisc.ac.in/").startswith("cds-index-")


def test_manifest_is_written_as_documents_are_saved(tmp_path: Path) -> None:
    crawler, _, _ = _crawl(tmp_path)
    lines = (tmp_path / "manifest.jsonl").read_text().splitlines()
    assert len(lines) == crawler.report.saved
