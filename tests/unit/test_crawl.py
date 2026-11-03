import json
from pathlib import Path

import httpx

from askiisc.ingest.crawl import (
    CrawlConfig,
    Crawler,
    canonical_key,
    categorise,
    department_of,
    doc_id_for,
    excluded,
    normalise_url,
    sitemap_urls,
)
from askiisc.retrieval.store import check_id
from tests.helpers import make_pdf

LONG = " ".join(["Admission eligibility rules for the programme."] * 20)


def html(body: str, title: str = "Page") -> bytes:
    head = f"<head><title>{title}</title></head>"
    return f"<html>{head}<body>{body}</body></html>".encode()


ADMISSIONS = html(
    f"<p>{LONG}</p><a href='/admissions/phd.pdf'>PhD</a>"
    "<a href='/admissions/copy/'>copy</a><a href='/admissions/short/'>s</a>"
    "<a href='/img.png'>i</a><a href='https://elsewhere.org/'>x</a>"
    "<a href='/admissions/scan.pdf'>scan</a><a href='/wp-admin/x'>a</a>"
    "<a href='https://webmail.iisc.ac.in/'>mail</a>"
    "<a href='https://ece.iisc.ac.in/'>ece</a>"
    "<a href='/admissions/timeout/'>slow</a><a href='/old/'>old</a>",
    "Admissions",
)
TEXT = "Department news and research groups. " * 20
SITE: dict[str, tuple[int, str, bytes]] = {
    "https://www.iisc.ac.in/robots.txt": (
        200,
        "text/plain",
        b"Sitemap: https://www.iisc.ac.in/s.xml\n",
    ),
    "https://www.iisc.ac.in/s.xml": (
        200,
        "application/xml",
        (
            b"<sitemapindex><sitemap><loc>https://www.iisc.ac.in/s2.xml</loc>"
            b"</sitemap></sitemapindex>"
        ),
    ),
    "https://www.iisc.ac.in/s2.xml": (
        200,
        "application/xml",
        (
            b"<urlset><url><loc>https://www.iisc.ac.in/hidden/</loc></url>"
            b"</urlset>"
        ),
    ),
    "https://www.iisc.ac.in/hidden/": (
        200,
        "text/html",
        html(f"<p>{TEXT}</p>", "Hidden"),
    ),
    "https://www.iisc.ac.in/admissions/": (200, "text/html", ADMISSIONS),
    "https://www.iisc.ac.in/admissions/phd.pdf": (
        200,
        "application/pdf",
        make_pdf([LONG[:300], LONG[:300]]),
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
    "https://cds.iisc.ac.in/robots.txt": (
        200,
        "text/plain",
        b"User-agent: *\nDisallow: /private/\n",
    ),
    "https://cds.iisc.ac.in/": (
        200,
        "text/html",
        html(
            f"<p>{TEXT}</p><a href='/private/x/'>p</a><a href='/people/'>pe</a>"
        ),
    ),
    "https://cds.iisc.ac.in/people/": (
        200,
        "text/html",
        html(
            "<p>" + "Faculty members and research groups of CDS. " * 20 + "</p>"
        ),
    ),
    "https://ece.iisc.ac.in/robots.txt": (403, "text/html", b"no"),
    "https://ece.iisc.ac.in/": (200, "text/html", html(f"<p>{TEXT} ECE</p>")),
}


class Fake:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.now = 0.0
        self.sleeps: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.calls.append(url)
        assert request.headers["user-agent"].startswith("AskIIScBot")
        if "down.iisc" in url:
            return httpx.Response(503)
        if url.endswith("/admissions/timeout/"):
            raise httpx.ReadTimeout("slow")
        if url.endswith("/old/"):
            return httpx.Response(301, headers={"location": "/admissions/"})
        status, kind, body = SITE.get(url, (404, "text/html", b""))
        return httpx.Response(
            status, content=body, headers={"content-type": kind, "etag": '"v1"'}
        )

    def crawler(self, out: Path, **overrides: object) -> Crawler:
        seeds = (
            "https://www.iisc.ac.in/admissions/",
            "https://cds.iisc.ac.in/",
            "https://down.iisc.ac.in/",
        )
        config = CrawlConfig(seeds=seeds, **overrides)  # type: ignore[arg-type]
        client = httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
        return Crawler(
            config,
            client,
            out,
            sleep=self.sleep,
            clock=lambda: self.now,
            wall=lambda: 1.0,
        )


def _manifest(out: Path) -> dict[str, dict[str, object]]:
    lines = (out / "manifest.jsonl").read_text().splitlines()
    return {json.loads(ln)["url"]: json.loads(ln) for ln in lines}


async def test_crawl_saves_documents_across_hosts(tmp_path: Path) -> None:
    fake = Fake()
    report = await fake.crawler(tmp_path).run()
    saved = _manifest(tmp_path)
    assert set(saved) == {
        "https://www.iisc.ac.in/admissions/",
        "https://www.iisc.ac.in/admissions/phd.pdf",
        "https://www.iisc.ac.in/hidden/",  # found only through the sitemap
        "https://cds.iisc.ac.in/",
        "https://cds.iisc.ac.in/people/",
        "https://ece.iisc.ac.in/",  # robots.txt 403: no restrictions
    }
    assert report.saved == 6
    assert report.duplicates == 2  # /admissions/copy/ and the /old/ redirect
    assert report.too_little_text == 1 and report.no_text_layer == 1
    assert report.blocked_by_robots == 1  # cds /private/x/
    assert report.hosts_unavailable == 1  # robots.txt 503: kept for later
    assert report.errors == 1  # the timeout
    assert report.hosts == 4
    fetched_pages = [c for c in fake.calls if not c.endswith("robots.txt")]
    assert not any("down.iisc.ac.in" in c for c in fetched_pages)
    assert not any(
        x in c
        for c in fake.calls
        for x in (
            "elsewhere.org",
            "webmail",
            ".png",
            "wp-admin",
            "cds.iisc.ac.in/private",
        )
    )
    pdf = saved["https://www.iisc.ac.in/admissions/phd.pdf"]
    assert str(pdf["path"]).startswith("public/institute/admissions/")
    assert pdf["etag"] == '"v1"' and pdf["visibility"] == "public"
    assert (tmp_path / str(pdf["path"])).read_bytes()[:4] == b"%PDF"
    assert str(saved["https://ece.iisc.ac.in/"]["path"]).startswith(
        "public/ece/general/"
    )
    assert report.by_department == {"institute": 3, "cds": 2, "ece": 1}
    for entry in saved.values():
        check_id(str(entry["doc_id"]))


async def test_politeness_per_host(tmp_path: Path) -> None:
    fake = Fake()
    await fake.crawler(tmp_path).run()
    assert fake.sleeps and all(0 < s <= 1.0 for s in fake.sleeps)


async def test_limits_and_resume(tmp_path: Path) -> None:
    fake = Fake()
    first = fake.crawler(tmp_path, max_docs=2)
    assert (await first.run()).stopped_by == "max_docs"
    first.checkpoint()
    # max_docs is a soft cap: requests already in flight still finish.
    after_first = len(_manifest(tmp_path))
    assert 2 <= after_first < 6
    state = json.loads((tmp_path / "crawl_state.json").read_text())
    assert any(h["queue"] for h in state["hosts"].values())
    second = fake.crawler(tmp_path)
    second.load()
    report = await second.run()
    assert len(_manifest(tmp_path)) == 6  # nothing saved twice
    assert report.saved == 6 - after_first


async def test_deadline_signal_and_host_cap(tmp_path: Path) -> None:
    fake = Fake()
    report = await fake.crawler(tmp_path, deadline_s=0.0).run()
    assert report.stopped_by == "deadline"
    capped_dir = tmp_path / "capped"
    capped_dir.mkdir()
    capped = await fake.crawler(capped_dir, max_requests_per_host=2).run()
    assert capped.saved < 6
    stopped_dir = tmp_path / "stopped"
    stopped_dir.mkdir()
    stopping = fake.crawler(stopped_dir)
    stopping.stop()
    assert (await stopping.run()).stopped_by == "signal"


async def test_unreachable_robots_skips_host(tmp_path: Path) -> None:
    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    config = CrawlConfig(seeds=("https://x.iisc.ac.in/",), use_sitemaps=False)
    client = httpx.AsyncClient(transport=httpx.MockTransport(broken))
    crawler = Crawler(config, client, tmp_path, sleep=Fake().sleep)
    report = await crawler.run()
    # A host that is down is kept for the next run, not marked as blocked.
    assert report.blocked_by_robots == 0 and report.saved == 0
    assert report.hosts_unavailable == 1
    assert report.stopped_by == "hosts unavailable"
    crawler.checkpoint()
    state = json.loads((tmp_path / "crawl_state.json").read_text())
    assert state["hosts"]["x.iisc.ac.in"]["queue"] == [
        ["https://x.iisc.ac.in/", 0]
    ]


async def test_host_down_in_one_run_is_crawled_in_the_next(
    tmp_path: Path,
) -> None:
    # Regression: www.csa.iisc.ac.in answered 502 for an hour; its whole
    # queue was counted as blocked and never crawled again.
    page = html(f"<p>{LONG}</p><a href='/academics/'>a</a>", "CSA")
    up = {"value": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if not up["value"]:
            return httpx.Response(502)
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(
            200, headers={"content-type": "text/html"}, content=page
        )

    seeds = ("https://www.csa.iisc.ac.in/",)
    config = CrawlConfig(seeds=seeds, use_sitemaps=False)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    first = Crawler(config, client, tmp_path, sleep=Fake().sleep)
    assert (await first.run()).saved == 0
    first.checkpoint()
    up["value"] = True
    second = Crawler(config, client, tmp_path, sleep=Fake().sleep)
    second.load()
    report = await second.run()
    # Home page saved; /academics/ has the same text, so it is a copy.
    assert report.saved == 1 and report.duplicates == 1
    assert report.stopped_by == "frontier empty"


async def test_retry_hosts_requeues_only_empty_hosts(tmp_path: Path) -> None:
    fake = Fake()
    crawler = fake.crawler(tmp_path)
    await crawler.run()
    crawler.checkpoint()
    again = fake.crawler(tmp_path)
    again.load()
    assert again.retry_hosts(["cds.iisc.ac.in", "down.iisc.ac.in"]) == 1
    assert again._hosts["down.iisc.ac.in"].queue  # noqa: SLF001


def test_helpers() -> None:
    assert normalise_url("https://WWW.IISC.ac.in/a?b=1#c") == (
        "https://www.iisc.ac.in/a"
    )
    assert normalise_url("https://cds.iisc.ac.in") == "https://cds.iisc.ac.in/"
    assert categorise("https://x/admissions/phd") == "admissions"
    assert categorise("https://x/a", "Student Handbook 2026") == "handbooks"
    assert categorise("https://x/courses/") == "course-catalogue"
    assert categorise("https://x/sports/") is None
    assert department_of("www.iisc.ac.in") == "institute"
    assert department_of("www.csa.iisc.ac.in") == "csa"
    assert department_of("www.ee.iisc.ac.in") == "ee"
    assert doc_id_for("https://www.iisc.ac.in/admissions/").startswith(
        "iisc-admissions-"
    )
    assert doc_id_for("https://cds.iisc.ac.in/").startswith("cds-index-")
    assert sitemap_urls("<loc> https://a.iisc.ac.in/x </loc>") == [
        "https://a.iisc.ac.in/x"
    ]


def test_canonical_key_joins_spellings_of_one_address() -> None:
    same = (
        "https://www.iisc.ac.in/academics/",
        "http://iisc.ac.in/academics",
        "https://iisc.ac.in/academics/index.html",
        "https://WWW.iisc.ac.in/academics/index.php",
    )
    assert {canonical_key(u) for u in same} == {"iisc.ac.in/academics"}
    # Path case is kept: servers may treat /DS252/ and /ds252/ apart.
    assert canonical_key("https://cds.iisc.ac.in/courses/DS252/") != (
        canonical_key("https://cds.iisc.ac.in/courses/ds252-aug25/")
    )
    key = canonical_key("https://cds.iisc.ac.in/")
    assert canonical_key(key) == key  # checkpoints store keys


async def test_http_and_www_spellings_are_fetched_once(tmp_path: Path) -> None:
    page = html(f"<p>{LONG}</p>", "Admissions")
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(
            200, headers={"content-type": "text/html"}, content=page
        )

    seeds = (
        "https://www.x.iisc.ac.in/admissions/",
        "http://x.iisc.ac.in/admissions",
    )
    config = CrawlConfig(seeds=seeds, use_sitemaps=False)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    report = await Crawler(config, client, tmp_path, sleep=Fake().sleep).run()
    assert report.saved == 1
    assert [c for c in calls if "robots" not in c] == [seeds[0]]


def test_academic_pages_are_queued_first(tmp_path: Path) -> None:
    config = CrawlConfig(seeds=())
    crawler = Crawler(config, httpx.AsyncClient(), tmp_path)
    for path in ("/news/a/", "/events/b/", "/academics/curriculum/"):
        crawler._enqueue(f"https://x.iisc.ac.in{path}", 1)  # noqa: SLF001
    queue = crawler._hosts["x.iisc.ac.in"].queue  # noqa: SLF001
    assert queue[0][0] == "https://x.iisc.ac.in/academics/curriculum/"


async def test_one_failing_page_never_stops_the_crawl(tmp_path: Path) -> None:
    """Regression: an exception while saving one page aborted job 1330."""
    fake = Fake()
    crawler = fake.crawler(tmp_path)
    original = crawler._maybe_save
    calls = [0]

    async def flaky(url, kind, response, title):  # type: ignore[no-untyped-def]
        calls[0] += 1
        if calls[0] == 1:
            raise ValueError("unexpected failure while saving")
        await original(url, kind, response, title)

    crawler._maybe_save = flaky  # type: ignore[method-assign]
    report = await crawler.run()
    # The failing page is counted and its links are lost; the crawl goes on
    # (it ends waiting only for the fake site's host that is down).
    assert report.stopped_by == "hosts unavailable"
    assert report.errors >= 1 and report.saved >= 3


async def test_repositories_are_out_of_scope_even_on_resume(
    tmp_path: Path,
) -> None:
    fake = Fake()
    crawler = fake.crawler(tmp_path)
    assert not crawler.in_scope("https://eprints.iisc.ac.in/123/")
    assert not crawler.in_scope("https://etd.iisc.ac.in/handle/1")
    state = {
        "seen": [],
        "hosts": {
            "eprints.iisc.ac.in": {
                "queue": [["https://eprints.iisc.ac.in/1/", 1]],
                "requests": 0,
            },
            "cds.iisc.ac.in": {
                "queue": [["https://cds.iisc.ac.in/people/", 1]],
                "requests": 3,
            },
        },
    }
    (tmp_path / "crawl_state.json").write_text(json.dumps(state))
    crawler.load()
    crawler.checkpoint()
    hosts = json.loads((tmp_path / "crawl_state.json").read_text())["hosts"]
    assert "eprints.iisc.ac.in" not in hosts
    assert hosts["cds.iisc.ac.in"]["queue"] == [
        ["https://cds.iisc.ac.in/people/", 1]
    ]


def _docx(text: str) -> bytes:
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", f"<w:p><w:t>{text}</w:t></w:p>")
    return buf.getvalue()


async def test_office_files_titles_and_rediscovery(tmp_path: Path) -> None:
    from askiisc.ingest.crawl import ct_hosts
    from askiisc.ingest.extract import DOCX

    fake = Fake()
    SITE["https://www.iisc.ac.in/admissions/form.docx"] = (
        200, "application/octet-stream", _docx(LONG),
    )  # fmt: skip
    page = ADMISSIONS.replace(
        b"</body>", b"<a href='/admissions/form.docx'>f</a></body>"
    )
    SITE["https://www.iisc.ac.in/forms/"] = (200, "text/html", page)
    try:
        crawler = fake.crawler(tmp_path)
        crawler.seed_hosts(["www.iisc.ac.in"])
        crawler._enqueue("https://www.iisc.ac.in/forms/", 0)  # noqa: SLF001
        await crawler.run()
        saved = _manifest(tmp_path)
        office = saved["https://www.iisc.ac.in/admissions/form.docx"]
        assert office["content_type"] == DOCX
        assert str(office["path"]).endswith(".docx")
        assert (
            saved["https://www.iisc.ac.in/admissions/"]["title"] == "Admissions"
        )
        # A new run that has only the manifest finds the links again.
        again = fake.crawler(tmp_path)
        again.load()
        assert again.rediscover() > 0
        assert again.seed_hosts(["cds.iisc.ac.in"]) == 0  # already saved
        assert again.seed_hosts(["new.iisc.ac.in"]) == 1
        internal = ["www.saps4qa.iisc.ac.in", "staging.x.iisc.ac.in"]
        assert again.seed_hosts(internal) == 0
    finally:
        del SITE["https://www.iisc.ac.in/admissions/form.docx"]
        del SITE["https://www.iisc.ac.in/forms/"]
    rows = [
        {"name_value": "a.iisc.ac.in\n*.b.iisc.ac.in\nA.IISC.AC.IN"},
        {"name_value": "x.example.org"},
        {},
    ]
    assert ct_hosts(rows) == ["a.iisc.ac.in"]


def test_generated_api_docs_are_out_of_scope() -> None:
    base = "https://math.iisc.ac.in/~gadgil"
    assert excluded(f"{base}/PfsProgs25doc/Aesop/Builder/Apply.html")
    assert excluded(f"{base}/proofs-and-programs-2023/doc/Mathlib/Data.html")
    assert excluded(f"{base}/proofs-and-programs-2023/doc/Init/")
    assert not excluded(f"{base}/proofs-and-programs-2023/notes/overview/")
    assert not excluded("https://iisc.ac.in/wp-content/uploads/doc/Fees.pdf")
    assert not excluded("https://www.iisc.ac.in/docs/handbook.html")
    assert not excluded("https://www.iisc.ac.in/doc/Admissions.html")
    assert excluded("https://eprints.iisc.ac.in/123/")
