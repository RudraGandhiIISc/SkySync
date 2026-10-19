"""A polite crawler for public IISc pages and PDFs.

Rules: only the configured hosts; on www.iisc.ac.in only URLs about
admissions, academic policy, courses and handbooks; everything on the CDS
and CSA sites. robots.txt is obeyed as RFC 9309 says (a 4xx robots file
means no restrictions, a 5xx or network error means do not crawl that
host). One request per second per host. Pages with little text and PDFs
without a text layer are not saved. Every saved file is recorded in a
manifest with its URL, ETag, Last-Modified and SHA-256, so the reconciler
can detect new versions later.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.robotparser
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from askiisc.common.errors import ExtractionError
from askiisc.ingest.extract import HTML, PDF, extract_text

USER_AGENT = "AskIIScBot/0.1 (DS252 course project, IISc; polite crawler)"
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
    ".mp4",
    ".mp3",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".xml",
    ".ics",
)
MIN_TEXT_CHARS = 400


@dataclass(frozen=True)
class CrawlConfig:
    """What to crawl.

    Attributes:
        seeds: Start URLs.
        departments: Host to department folder; hosts here are crawled
            fully.
        topical_hosts: Hosts where only topic URLs (CATEGORIES) are kept.
        max_docs: Stop after saving this many documents.
        max_requests: Hard cap on requests.
        max_depth: Link depth from a seed.
        delay_s: Minimum gap between requests to one host.
        max_bytes: Larger responses are skipped.
    """

    seeds: tuple[str, ...]
    departments: dict[str, str]
    topical_hosts: frozenset[str]
    max_docs: int = 1000
    max_requests: int = 6000
    max_depth: int = 4
    delay_s: float = 1.0
    max_bytes: int = 25_000_000


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


@dataclass
class CrawlReport:
    """Counts for the run log."""

    requests: int = 0
    saved: int = 0
    duplicates: int = 0
    too_little_text: int = 0
    no_text_layer: int = 0
    blocked_by_robots: int = 0
    off_topic: int = 0
    errors: int = 0
    by_category: dict[str, int] = field(default_factory=dict)


def normalise_url(url: str) -> str:
    """Drops fragments and query strings and lower-cases the host."""
    clean, _ = urldefrag(url)
    parts = urlparse(clean)
    path = parts.path or "/"
    return f"{parts.scheme}://{parts.netloc.lower()}{path}"


def categorise(url: str, title: str = "") -> str | None:
    """Maps a URL (and title) to a topic folder, or None if off-topic."""
    text = f"{urlparse(url).path} {title}".lower()
    for category, words in CATEGORIES.items():
        if any(w in text for w in words):
            return category
    return None


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


class Robots:
    """robots.txt per host, cached, with RFC 9309 status handling."""

    def __init__(self, fetch: Callable[[str], httpx.Response]) -> None:
        self._fetch = fetch
        self._rules: dict[str, urllib.robotparser.RobotFileParser | bool] = {}

    def allowed(self, url: str) -> bool:
        """Whether USER_AGENT may fetch the URL."""
        parts = urlparse(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host not in self._rules:
            self._rules[host] = self._load(host)
        rules = self._rules[host]
        if isinstance(rules, bool):
            return rules
        return rules.can_fetch(USER_AGENT, url)

    def _load(self, host: str) -> urllib.robotparser.RobotFileParser | bool:
        try:
            response = self._fetch(host + "/robots.txt")
        except httpx.HTTPError:
            return False  # unreachable: do not crawl the host
        if 400 <= response.status_code < 500:  # noqa: PLR2004
            return True  # RFC 9309: unavailable means no restrictions
        if response.status_code >= 500:  # noqa: PLR2004
            return False
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(response.text.splitlines())
        return parser


class Crawler:
    """Breadth-first crawl that saves on-topic documents with text."""

    def __init__(
        self,
        config: CrawlConfig,
        client: httpx.Client,
        out_dir: Path,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._client = client
        self._out = out_dir
        self._sleep = sleep
        self._clock = clock
        self._last: dict[str, float] = {}
        self._hashes: set[str] = set()
        self.report = CrawlReport()
        self.manifest: list[ManifestEntry] = []
        self._robots = Robots(self._get)

    def _get(self, url: str) -> httpx.Response:
        host = urlparse(url).netloc
        wait = self._last.get(host, -1e9) + self._config.delay_s - self._clock()
        if wait > 0:
            self._sleep(wait)
        self._last[host] = self._clock()
        self.report.requests += 1
        return self._client.get(
            url,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            timeout=30.0,
        )

    def _in_scope(self, url: str) -> bool:
        host = urlparse(url).netloc
        if url.lower().endswith(SKIP_SUFFIXES):
            return False
        if "/wp-admin" in url or "/wp-json" in url or "/feed" in url:
            return False
        return host in self._config.departments or (
            host in self._config.topical_hosts
        )

    def run(self) -> CrawlReport:
        """Crawls until max_docs are saved or the frontier is empty.

        Each saved document is appended to manifest.jsonl at once, so a run
        stopped by its time limit still leaves a usable manifest.
        """
        (self._out / "manifest.jsonl").write_text("")
        queue = deque((normalise_url(s), 0) for s in self._config.seeds)
        seen = {u for u, _ in queue}
        cfg = self._config
        while queue and self.report.saved < cfg.max_docs:
            if self.report.requests >= cfg.max_requests:
                break
            url, depth = queue.popleft()
            for nxt in self._visit(url):
                if (
                    depth < cfg.max_depth
                    and nxt not in seen
                    and self._in_scope(nxt)
                ):
                    seen.add(nxt)
                    queue.append((nxt, depth + 1))
        return self.report

    def _visit(self, url: str) -> Iterable[str]:
        if not self._robots.allowed(url):
            self.report.blocked_by_robots += 1
            return []
        try:
            response = self._get(url)
        except httpx.HTTPError:
            self.report.errors += 1
            return []
        kind = response.headers.get("content-type", "").split(";")[0].strip()
        if response.status_code != httpx.codes.OK or kind not in {HTML, PDF}:
            return []
        if len(response.content) > self._config.max_bytes:
            return []
        final = normalise_url(str(response.url))
        title = page_title(response.content) if kind == HTML else ""
        self._maybe_save(final, kind, response, title)
        return links(response.content, final) if kind == HTML else []

    def _maybe_save(
        self, url: str, kind: str, response: httpx.Response, title: str
    ) -> None:
        host = urlparse(url).netloc
        department = self._config.departments.get(host, "institute")
        category = categorise(url, title)
        if department == "institute" and category is None:
            self.report.off_topic += 1
            return
        try:
            text = "\n".join(
                p.text for p in extract_text(response.content, kind)
            )
        except ExtractionError:
            self.report.no_text_layer += 1
            return
        if len(text) < MIN_TEXT_CHARS:
            self.report.too_little_text += 1
            return
        digest = hashlib.sha256(response.content).hexdigest()
        if digest in self._hashes:
            self.report.duplicates += 1
            return
        self._hashes.add(digest)
        self._save(
            url, kind, response, department, category or "department", digest
        )

    def _save(
        self,
        url: str,
        kind: str,
        response: httpx.Response,
        department: str,
        category: str,
        digest: str,
    ) -> None:
        doc_id = doc_id_for(url)
        folder = self._out / "public" / department / category
        folder.mkdir(parents=True, exist_ok=True)
        suffix = ".pdf" if kind == PDF else ".html"
        path = folder / f"{doc_id}{suffix}"
        path.write_bytes(response.content)
        self.manifest.append(
            ManifestEntry(
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
                fetched_at=self._clock(),
            )
        )
        with (self._out / "manifest.jsonl").open("a") as f:
            f.write(json.dumps(asdict(self.manifest[-1])) + "\n")
        self.report.saved += 1
        key = f"{department}/{category}"
        self.report.by_category[key] = self.report.by_category.get(key, 0) + 1

    def write_manifest(self) -> Path:
        """Writes manifest.jsonl (one JSON object per saved document)."""
        path = self._out / "manifest.jsonl"
        with path.open("w") as f:
            for entry in self.manifest:
                f.write(json.dumps(asdict(entry)) + "\n")
        return path


IISC_CONFIG = CrawlConfig(
    seeds=(
        "https://www.iisc.ac.in/admissions/",
        "https://www.iisc.ac.in/academics/",
        "https://www.iisc.ac.in/students/",
        "https://cds.iisc.ac.in/",
        "https://www.csa.iisc.ac.in/",
    ),
    departments={
        "cds.iisc.ac.in": "cds",
        "www.csa.iisc.ac.in": "csa",
        "csa.iisc.ac.in": "csa",
    },
    topical_hosts=frozenset({"www.iisc.ac.in", "iisc.ac.in"}),
)


def main() -> None:  # pragma: no cover
    """Runs the IISc crawl into ASKIISC_CORPUS_DIR (a Slurm CPU job)."""
    import os  # noqa: PLC0415

    out = Path(os.environ["ASKIISC_CORPUS_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    with httpx.Client() as client:
        crawler = Crawler(IISC_CONFIG, client, out)
        try:
            report = crawler.run()
        finally:
            crawler.write_manifest()
    (out / "crawl_report.json").write_text(json.dumps(asdict(report), indent=2))
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
