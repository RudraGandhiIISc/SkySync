"""A polite, parallel, resumable crawler for public IISc sites.

Scope: every public host under iisc.ac.in (minus mail, login and similar
portals and the research repositories): HTML pages, PDFs and Office files
(.docx, .pptx, .xlsx). Politeness: robots.txt per RFC 9309 (a
4xx robots file means no restrictions; a 5xx or network error means skip
the host), at most one request per second to each host, and at most
`max_in_flight` requests in flight overall. Hosts are crawled in parallel,
which is what makes a whole-institute crawl take hours rather than days.

Speed-ups: each host's sitemaps (robots.txt `Sitemap:` lines,
/wp-sitemap.xml, /sitemap.xml) seed its queue, so most pages are found
without walking menus.

Resumable: every saved document is appended to manifest.jsonl at once, and
the frontier and visited set are checkpointed to crawl_state.json every
minute and on exit, so a run cut off by a Slurm time limit continues where
it stopped.

Hosts that are down: a robots.txt that fails (server error or no
connection) sets the host aside for this run with its queue kept; the next
run tries it again. `retry_hosts` re-queues hosts that saved nothing (for
example after an outage during an earlier run).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import signal
import time
import urllib.robotparser
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from askiisc.common.errors import AskIIScError, ExtractionError
from askiisc.ingest.extract import (
    DOCX,
    HTML,
    OFFICE,
    PDF,
    PPTX,
    XLSX,
    extract_text,
)

_LOG = logging.getLogger(__name__)
USER_AGENT = "AskIIScBot/0.2 (DS252 course project, IISc; polite crawler)"
CATEGORIES = {
    "admissions": ("admission", "apply", "eligib", "selection", "intake"),
    "course-catalogue": (
        "course",
        "curricul",
        "syllab",
        "scheme",
        "programme",
        "program",
        "catalog",
    ),
    "handbooks": ("handbook", "guide", "manual", "brochure", "prospectus"),
    "academic-policy": (
        "regulation",
        "rule",
        "polic",
        "ordinance",
        "academic",
        "exam",
        "thesis",
        "phd",
        "fee",
        "scholar",
        "calendar",
        "student",
    ),
}
SKIP_SUFFIXES = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".css",
    ".js",
    ".zip",
    ".rar",
    ".gz",
    ".tar",
    ".mp4",
    ".mp3",
    ".avi",
    ".mov",
    ".doc",
    ".ppt",
    ".xls",
    ".ics",
    ".exe",
    ".iso",
    ".tex",
    ".bib",
)
SKIP_PATHS = (
    "/wp-admin",
    "/wp-json",
    "/feed",
    "/wp-login",
    "/cart",
    "/login",
    "/logout",
    "/calendar/",
    "/tag/",
    "/author/",
    "/page/",
    "/xmlrpc",
)
EXCLUDED_HOST_WORDS = (
    "mail",
    "vpn",
    "login",
    "sso",
    "erp",
    "moodle",
    "webmail",
    "outlook",
    "intranet",
    "proxy",
    "git",
    "jupyter",
    "ldap",
    # Internal systems found in certificate-transparency logs.
    "sap",
    "s4h",
    "s4p",
    "staging",
    "test",
    "cpanel",
    "webdisk",
    "radius",
    "cyberoam",
    "sophos",
    "grafana",
    "phabricator",
    "chatwoot",
    "mx",
    "dev",
    "tmp",
    "admin",
    "nodues",
    "boking",
    "record",
)
# Research-output repositories (papers, theses, journal articles): out of
# scope for an academic-policy assistant, and ~95% of all crawlable URLs.
EXCLUDED_HOSTS = frozenset(
    {"eprints.iisc.ac.in", "etd.iisc.ac.in", "journal.iisc.ac.in"}
)
# Generated API reference trees (doc-gen4 output for Lean libraries on a
# course site: 10,900 pages of math.iisc.ac.in in October 2026): a folder
# named *doc holding CamelCase namespace folders. Code reference, not
# institute information, and a third of the corpus if kept.
GENERATED_DOCS = re.compile(r"/[^/]*doc/[A-Z][A-Za-z0-9_]*/")
DEPARTMENT_HOSTS = {
    "www.iisc.ac.in": "institute",
    "iisc.ac.in": "institute",
    "www.csa.iisc.ac.in": "csa",
    "csa.iisc.ac.in": "csa",
}
MIN_TEXT_CHARS = 400
SUFFIX_OF = {
    HTML: ".html",
    PDF: ".pdf",
    DOCX: ".docx",
    PPTX: ".pptx",
    XLSX: ".xlsx",
}
KIND_OF_SUFFIX = {v: k for k, v in SUFFIX_OF.items() if k in OFFICE}
# Topics crawled before everything else on a host (news and events last),
# so a host cap or deadline never cuts off the curricula and rules.
PRIORITY_CATEGORIES = frozenset(
    {"course-catalogue", "handbooks", "academic-policy", "admissions"}
)


def excluded(url: str) -> bool:
    """Whether a URL is out of scope whatever the crawl settings."""
    parts = urlparse(url)
    return parts.netloc in EXCLUDED_HOSTS or bool(
        GENERATED_DOCS.search(parts.path)
    )


_INDEX_FILE = re.compile(r"/index\.(?:html?|php)$", re.IGNORECASE)
_SITEMAP_LOC = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")


@dataclass(frozen=True)
class CrawlConfig:
    """What to crawl and how hard.

    Attributes:
        seeds: Start URLs.
        allowed_suffix: Hosts must end with this (or equal it without the
            leading dot).
        max_docs: Stop after saving this many documents in total.
        max_requests_per_host: Cap per host, so one big site cannot take
            the whole run.
        max_depth: Link depth from a seed or sitemap entry.
        delay_s: Minimum gap between requests to one host.
        max_in_flight: Requests in flight across all hosts.
        max_bytes: Larger responses are skipped.
        deadline_s: Stop (and checkpoint) after this many seconds.
        use_sitemaps: Seed each host from its sitemaps.
    """

    seeds: tuple[str, ...]
    allowed_suffix: str = ".iisc.ac.in"
    max_docs: int = 100_000
    max_requests_per_host: int = 50_000
    max_depth: int = 10
    delay_s: float = 1.0
    max_in_flight: int = 32
    max_bytes: int = 25_000_000
    deadline_s: float = 7.0 * 24 * 3600
    use_sitemaps: bool = True


@dataclass
class ManifestEntry:
    """One saved document."""

    doc_id: str
    path: str
    url: str
    category: str
    department: str
    visibility: str
    content_type: str
    etag: str | None
    last_modified: str | None
    sha256: str
    bytes: int
    fetched_at: float
    title: str = ""


@dataclass
class CrawlReport:
    """Counts for the run log."""

    requests: int = 0
    saved: int = 0
    duplicates: int = 0
    too_little_text: int = 0
    no_text_layer: int = 0
    blocked_by_robots: int = 0
    errors: int = 0
    hosts: int = 0
    hosts_unavailable: int = 0
    stopped_by: str = "frontier empty"
    by_department: dict[str, int] = field(default_factory=dict)


def content_kind(response: httpx.Response) -> str:
    """The MIME type; Office files sent as octet-stream go by suffix."""
    header = str(response.headers.get("content-type", ""))
    kind = header.split(";", maxsplit=1)[0].strip()
    if kind in SUFFIX_OF:
        return kind
    for suffix, office in KIND_OF_SUFFIX.items():
        if urlparse(str(response.url)).path.lower().endswith(suffix):
            return str(office)
    return kind


def normalise_url(url: str) -> str:
    """Drops fragments and query strings and lower-cases the host."""
    clean, _ = urldefrag(url)
    parts = urlparse(clean)
    return f"{parts.scheme}://{parts.netloc.lower()}{parts.path or '/'}"


def canonical_key(url: str) -> str:
    """One key for every spelling of the same address.

    http and https, a leading "www.", a trailing slash and an index file
    name ("/a/index.html") all name the same page on these sites.

    Args:
        url: A normalised URL.

    Returns:
        Host (without "www.") and path, without scheme or trailing slash.
    """
    parts = urlparse(url)
    host = parts.netloc.lower().removeprefix("www.")
    path = _INDEX_FILE.sub("/", parts.path or "/").rstrip("/")
    return f"{host}{path}"


def categorise(url: str, title: str = "") -> str | None:
    """Maps a URL (and title) to a topic folder, or None."""
    text = f"{urlparse(url).path} {title}".lower()
    for category, words in CATEGORIES.items():
        if any(w in text for w in words):
            return category
    return None


def department_of(host: str) -> str:
    """Folder name for a host: known mappings, else its first label."""
    if host in DEPARTMENT_HOSTS:
        return DEPARTMENT_HOSTS[host]
    labels = [lb for lb in host.split(".") if lb != "www"]
    return re.sub(r"[^a-z0-9-]", "-", labels[0]) if labels else "institute"


def doc_id_for(url: str) -> str:
    """A stable, filter-safe id: host prefix, path slug and a short hash."""
    parts = urlparse(url)
    host = parts.netloc.split(".")[0].replace("www", "iisc") or "iisc"
    slug = re.sub(r"[^A-Za-z0-9]+", "-", parts.path).strip("-").lower()
    digest = hashlib.sha256(url.encode()).hexdigest()[:8]
    return f"{host}-{slug[:80] or 'index'}-{digest}"


def links(html: bytes, base: str) -> list[str]:
    """Absolute http(s) links in a page."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for tag in soup.find_all("a", href=True):
        href = urljoin(base, str(tag["href"]))
        if urlparse(href).scheme in {"http", "https"}:
            out.append(normalise_url(href))
    return out


def page_title(html: bytes) -> str:
    """The <title> text, or empty."""
    soup = BeautifulSoup(html, "html.parser")
    return soup.title.get_text(" ", strip=True) if soup.title else ""


def sitemap_urls(xml: str) -> list[str]:
    """Every <loc> URL in a sitemap or sitemap index."""
    return [normalise_url(u) for u in _SITEMAP_LOC.findall(xml)]


class Host:
    """Per-host queue, politeness clock and robots rules."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.queue: deque[tuple[str, int]] = deque()
        self.last_request = -1e18
        self.requests = 0
        self.rules: urllib.robotparser.RobotFileParser | bool | None = None
        self.sitemaps: list[str] = []
        self.lock = asyncio.Lock()
        # robots.txt failed in this run: queue kept for the next run.
        self.unavailable = False


Sleep = Callable[[float], Awaitable[None]]


class Crawler:
    """Crawls hosts in parallel; saves documents with enough text."""

    def __init__(
        self,
        config: CrawlConfig,
        client: httpx.AsyncClient,
        out_dir: Path,
        sleep: Sleep = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._client = client
        self._out = out_dir
        self._sleep = sleep
        self._clock = clock
        self._wall = wall
        self._hosts: dict[str, Host] = {}
        self._seen: set[str] = set()  # canonical keys
        self._hashes: set[str] = set()
        self._saved_urls: set[str] = set()
        self._in_flight = asyncio.Semaphore(config.max_in_flight)
        self._started = clock()
        self._stop = False
        self.report = CrawlReport()

    # Scope.

    def in_scope(self, url: str) -> bool:
        """Whether a URL may be crawled at all."""
        parts = urlparse(url)
        host, path = parts.netloc, parts.path.lower()
        suffix = self._config.allowed_suffix
        if not (host.endswith(suffix) or host == suffix.lstrip(".")):
            return False
        if excluded(url):
            return False
        label = next((p for p in host.split(".") if p != "www"), "")
        if any(word in label for word in EXCLUDED_HOST_WORDS):
            return False
        if path.endswith(SKIP_SUFFIXES):
            return False
        return not any(p in path for p in SKIP_PATHS)

    def _enqueue(self, url: str, depth: int) -> None:
        key = canonical_key(url)
        if key in self._seen or not self.in_scope(url):
            return
        self._seen.add(key)
        host = urlparse(url).netloc
        if host not in self._hosts:
            self._hosts[host] = Host(host)
        if categorise(url) in PRIORITY_CATEGORIES:
            self._hosts[host].queue.appendleft((url, depth))
        else:
            self._hosts[host].queue.append((url, depth))

    # Requests.

    async def _get(self, host: Host, url: str) -> httpx.Response:
        async with host.lock:
            wait = host.last_request + self._config.delay_s - self._clock()
            if wait > 0:
                await self._sleep(wait)
            host.last_request = self._clock()
        host.requests += 1
        self.report.requests += 1
        async with self._in_flight:
            return await self._client.get(
                url,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
                timeout=30.0,
            )

    async def _robots(self, host: Host, scheme: str) -> None:
        try:
            response = await self._get(
                host, f"{scheme}://{host.name}/robots.txt"
            )
        except httpx.HTTPError:
            host.unavailable = True  # RFC 9309: unreachable means skip
            return
        if 400 <= response.status_code < 500:  # noqa: PLR2004
            host.rules = True  # RFC 9309: unavailable means no restrictions
            return
        if response.status_code >= 500:  # noqa: PLR2004
            host.unavailable = True
            return
        parser = urllib.robotparser.RobotFileParser()
        lines = response.text.splitlines()
        parser.parse(lines)
        host.rules = parser
        host.sitemaps = [
            ln.split(":", 1)[1].strip()
            for ln in lines
            if ln.lower().startswith("sitemap:")
        ]

    def _allowed(self, host: Host, url: str) -> bool:
        if isinstance(host.rules, bool):
            return host.rules
        return host.rules is not None and host.rules.can_fetch(USER_AGENT, url)

    async def _seed_sitemaps(self, host: Host, scheme: str) -> None:
        base = f"{scheme}://{host.name}"
        pending = deque(
            host.sitemaps or [base + "/wp-sitemap.xml", base + "/sitemap.xml"]
        )
        fetched = 0
        while pending and fetched < 50:  # noqa: PLR2004
            url = pending.popleft()
            fetched += 1
            try:
                response = await self._get(host, url)
            except httpx.HTTPError:
                continue
            if response.status_code != httpx.codes.OK:
                continue
            for loc in sitemap_urls(response.text):
                if loc.endswith(".xml") or "sitemap" in urlparse(loc).path:
                    pending.append(loc)
                else:
                    self._enqueue(loc, 1)

    # Per-host worker.

    def _budget_left(self) -> bool:
        if self.report.saved >= self._config.max_docs:
            self.report.stopped_by = "max_docs"
            return False
        if self._clock() - self._started >= self._config.deadline_s:
            self.report.stopped_by = "deadline"
            return False
        return not self._stop

    async def _work(self, host: Host) -> None:
        scheme = urlparse(host.queue[0][0]).scheme if host.queue else "https"
        if host.rules is None:
            await self._robots(host, scheme)
            if host.unavailable:
                self.report.hosts_unavailable += 1
                _LOG.warning("%s is down: kept for the next run", host.name)
                return
            if self._config.use_sitemaps and host.rules is not False:
                await self._seed_sitemaps(host, scheme)
        while host.queue and self._budget_left():
            if host.requests >= self._config.max_requests_per_host:
                break
            url, depth = host.queue.popleft()
            if not self._allowed(host, url):
                self.report.blocked_by_robots += 1
                continue
            for nxt in await self._visit(host, url):
                if depth < self._config.max_depth:
                    self._enqueue(nxt, depth + 1)

    async def run(self) -> CrawlReport:
        """Crawls until the budget ends or every queue is empty."""
        for seed in self._config.seeds:
            self._enqueue(normalise_url(seed), 0)
        running: dict[str, asyncio.Task[None]] = {}
        while self._budget_left():
            for name, host in self._hosts.items():
                if (
                    host.queue
                    and name not in running
                    and not host.unavailable
                    and (host.requests < self._config.max_requests_per_host)
                ):
                    running[name] = asyncio.create_task(self._work(host))
            if not running:
                break
            done, _ = await asyncio.wait(
                running.values(),
                timeout=1.0,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for name in [n for n, t in running.items() if t in done]:
                running.pop(name).result()
        for task in running.values():
            task.cancel()
        await asyncio.gather(*running.values(), return_exceptions=True)
        self.report.hosts = len(self._hosts)
        waiting = any(h.unavailable and h.queue for h in self._hosts.values())
        if waiting and self.report.stopped_by == "frontier empty":
            self.report.stopped_by = "hosts unavailable"
        return self.report

    def stop(self) -> None:
        """Asks the crawl to stop soon (for SIGTERM)."""
        self._stop = True
        self.report.stopped_by = "signal"

    # Fetch and save.

    async def _visit(self, host: Host, url: str) -> list[str]:
        """Fetches one URL; any failure is counted, never fatal."""
        try:
            return await self._fetch_and_save(host, url)
        except (httpx.HTTPError, AskIIScError, ValueError, OSError) as err:
            self.report.errors += 1
            _LOG.warning("skipped %s: %s", url, type(err).__name__)
            return []

    async def _fetch_and_save(self, host: Host, url: str) -> list[str]:
        response = await self._get(host, url)
        kind = content_kind(response)
        if response.status_code != httpx.codes.OK or kind not in SUFFIX_OF:
            return []
        if len(response.content) > self._config.max_bytes:
            return []
        final = normalise_url(str(response.url))
        title = page_title(response.content) if kind == HTML else ""
        await self._maybe_save(final, kind, response, title)
        return links(response.content, final) if kind == HTML else []

    async def _maybe_save(
        self, url: str, kind: str, response: httpx.Response, title: str
    ) -> None:
        try:
            pages = await asyncio.to_thread(
                extract_text, response.content, kind
            )
        except ExtractionError:
            self.report.no_text_layer += 1
            return
        if sum(len(p.text) for p in pages) < MIN_TEXT_CHARS:
            self.report.too_little_text += 1
            return
        digest = hashlib.sha256(response.content).hexdigest()
        # Two links can redirect to one page whose bytes differ per fetch
        # (timestamps), so the URL is checked as well as the content hash.
        key = canonical_key(url)
        if digest in self._hashes or key in self._saved_urls:
            self.report.duplicates += 1
            return
        self._hashes.add(digest)
        self._saved_urls.add(key)
        self._save(
            url,
            kind,
            response,
            categorise(url, title) or "general",
            digest,
            title,
        )

    def _save(
        self,
        url: str,
        kind: str,
        response: httpx.Response,
        category: str,
        digest: str,
        title: str = "",
    ) -> None:
        department = department_of(urlparse(url).netloc)
        doc_id = doc_id_for(url)
        folder = self._out / "public" / department / category
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{doc_id}{SUFFIX_OF[kind]}"
        path.write_bytes(response.content)
        entry = ManifestEntry(
            doc_id=doc_id,
            path=str(path.relative_to(self._out)),
            url=url,
            category=category,
            department=department,
            visibility="public",
            content_type=kind,
            etag=response.headers.get("etag"),
            last_modified=response.headers.get("last-modified"),
            sha256=digest,
            bytes=len(response.content),
            fetched_at=self._wall(),
            title=" ".join(title.split())[:160],
        )
        with (self._out / "manifest.jsonl").open("a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")
        self.report.saved += 1
        counts = self.report.by_department
        counts[department] = counts.get(department, 0) + 1

    # Resume.

    def rediscover(self) -> int:
        """Queues unseen links of every saved HTML page (depth 0).

        Links a scope or depth rule once dropped (Office files, deep pages)
        are found again without re-fetching the pages.

        Returns:
            The number of URLs queued.
        """
        before = len(self._seen)
        manifest = self._out / "manifest.jsonl"
        if not manifest.exists():
            return 0
        for line in manifest.read_text().splitlines():
            entry = json.loads(line)
            if entry["content_type"] != HTML:
                continue
            path = self._out / entry["path"]
            if path.exists():
                for url in links(path.read_bytes(), entry["url"]):
                    self._enqueue(url, 0)
        return len(self._seen) - before

    def retry_hosts(self, hosts: list[str]) -> int:
        """Crawls hosts again that saved nothing in earlier runs.

        Their unsaved addresses are forgotten and their home pages queued,
        so an outage during an earlier run does not lose a whole site.

        Args:
            hosts: Host names to retry.

        Returns:
            How many of them had saved nothing and were re-queued.
        """
        saved = {k.split("/", 1)[0] for k in self._saved_urls}
        retried = 0
        for name in hosts:
            bare = name.lower().removeprefix("www.")
            if bare in saved:
                continue
            self._seen = {k for k in self._seen if k.split("/", 1)[0] != bare}
            self._hosts.pop(name, None)
            self._enqueue(f"https://{name}/", 0)
            retried += 1
        return retried

    def seed_hosts(self, hosts: list[str]) -> int:
        """Queues the home page of each host (those not yet seen)."""
        before = len(self._seen)
        for host in hosts:
            self._enqueue(f"https://{host}/", 0)
        return len(self._seen) - before

    def load(self) -> None:
        """Restores the manifest and the checkpoint, if any."""
        manifest = self._out / "manifest.jsonl"
        if manifest.exists():
            for line in manifest.read_text().splitlines():
                entry = json.loads(line)
                self._hashes.add(entry["sha256"])
                self._saved_urls.add(canonical_key(entry["url"]))
                self._seen.add(canonical_key(entry["url"]))
        state = self._out / "crawl_state.json"
        if not state.exists():
            return
        data = json.loads(state.read_text())
        self._seen |= {canonical_key(u) for u in data["seen"]}
        for name, item in data["hosts"].items():
            queue = [(u, d) for u, d in item["queue"] if self.in_scope(u)]
            if not queue and not item["requests"]:
                continue
            host = self._hosts.setdefault(name, Host(name))
            host.queue.extend(queue)  # scope rules may have changed
            host.requests = item["requests"]

    def checkpoint(self) -> Path:
        """Writes the frontier and visited set (atomically)."""
        data = {
            "seen": sorted(self._seen),
            "hosts": {
                n: {"queue": list(h.queue), "requests": h.requests}
                for n, h in self._hosts.items()
            },
        }
        path = self._out / "crawl_state.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(path)
        return path


IISC_SEEDS = (
    "https://www.iisc.ac.in/",
    "https://www.iisc.ac.in/admissions/",
    "https://www.iisc.ac.in/academics/",
    "https://www.iisc.ac.in/departments/",
    "https://www.iisc.ac.in/students/",
    "https://cds.iisc.ac.in/",
    "https://www.csa.iisc.ac.in/",
)


async def _checkpoint_every(crawler: Crawler, seconds: float) -> None:
    while True:  # pragma: no cover - runs for the life of the job
        await asyncio.sleep(seconds)
        crawler.checkpoint()


async def crawl(
    out: Path, deadline_s: float, retry: bool = False
) -> CrawlReport:  # pragma: no cover
    """Runs (or resumes) the full IISc crawl into `out`.

    `retry` re-queues listed hosts that have saved nothing so far.
    """
    config = CrawlConfig(seeds=IISC_SEEDS, deadline_s=deadline_s)
    limits = httpx.Limits(max_connections=config.max_in_flight)
    async with httpx.AsyncClient(limits=limits) as client:
        crawler = Crawler(config, client, out)
        crawler.load()
        hosts = out / "hosts.txt"
        if hosts.exists():
            n = crawler.seed_hosts(hosts.read_text().split())
            _LOG.info("seeded %d new hosts from %s", n, hosts.name)
            if retry:
                n = crawler.retry_hosts(hosts.read_text().split())
                _LOG.info("re-queued %d hosts that saved nothing", n)
        marker = out / "rediscovered"
        if not marker.exists():
            _LOG.info("rediscovered %d links", crawler.rediscover())
            crawler.checkpoint()
            marker.touch()
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, crawler.stop)
        saver = asyncio.create_task(_checkpoint_every(crawler, 60.0))
        try:
            return await crawler.run()
        finally:
            saver.cancel()
            crawler.checkpoint()


def ct_hosts(
    records: list[dict[str, str]], suffix: str = ".iisc.ac.in"
) -> list[str]:
    """Host names under suffix in certificate-transparency records.

    Args:
        records: crt.sh JSON rows (`name_value` holds names, one per line).
        suffix: The domain to keep.

    Returns:
        Sorted unique host names (wildcards dropped, lower case).
    """
    names = set()
    for record in records:
        for raw in str(record.get("name_value", "")).split():
            name = raw.strip().lower()
            if name.endswith(suffix) and "*" not in name:
                names.add(name)
    return sorted(names)


def _crt_sh() -> list[dict[str, str]]:  # pragma: no cover - network
    """crt.sh rows for *.iisc.ac.in; crt.sh is often busy, so retry."""
    import time as clock  # noqa: PLC0415

    params = {"q": "%.iisc.ac.in", "output": "json", "deduplicate": "Y"}
    for attempt in range(6):
        try:
            response = httpx.get(
                "https://crt.sh/",
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=600.0,
            )
            response.raise_for_status()
            return list(response.json())
        except (httpx.HTTPError, ValueError) as err:
            _LOG.warning("crt.sh attempt %d failed: %s", attempt + 1, err)
            clock.sleep(30.0 * (attempt + 1))
    raise AskIIScError("crt.sh did not answer")


def main() -> None:  # pragma: no cover
    """Entry point: ASKIISC_CORPUS_DIR, optional ASKIISC_CRAWL_SECONDS.

    `python -m askiisc.ingest.crawl hosts` instead writes hosts.txt: every
    iisc.ac.in host name in the public certificate-transparency logs
    (crt.sh), so hosts that nothing links to are crawled too.
    """
    import os  # noqa: PLC0415
    import sys  # noqa: PLC0415

    out = Path(os.environ["ASKIISC_CORPUS_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["hosts"]:
        hosts = ct_hosts(_crt_sh())
        (out / "hosts.txt").write_text("\n".join(hosts) + "\n")
        print(f"{len(hosts)} hosts")
        return
    seconds = float(os.environ.get("ASKIISC_CRAWL_SECONDS", "6600"))
    retry = os.environ.get("ASKIISC_CRAWL_RETRY_EMPTY") == "1"
    report = asyncio.run(crawl(out, seconds, retry))
    text = json.dumps(asdict(report), indent=2)
    (out / "crawl_report.json").write_text(text)
    print(text)


if __name__ == "__main__":  # pragma: no cover
    main()
