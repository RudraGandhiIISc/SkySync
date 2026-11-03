"""Past editions of academic pages from the Internet Archive.

Live sites overwrite last year's curriculum or course page in place, so
the 2025 edition of /courses/DS252/ exists only in archives. For every
academic page the crawl found (curricula, course pages, handbooks, rules,
admissions), this asks the Wayback Machine's CDX index for its snapshots
and keeps the last snapshot of each past academic year whose content
differs from the live copy and from the other kept snapshots.

Saved as corpus/archive/<department>/<doc_id>.<ext>, listed in
archive.jsonl with the snapshot time and the original address. Citations
link to the snapshot on web.archive.org.

Politeness: one request at a time, at least `delay_s` apart; on HTTP 429
or 5xx the fetcher waits (doubling, up to `max_backoff_s`) and retries.
Resumable: finished addresses are recorded in archive_state.json.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from askiisc.common.errors import ExtractionError
from askiisc.ingest.crawl import (
    MIN_TEXT_CHARS,
    PRIORITY_CATEGORIES,
    SUFFIX_OF,
    USER_AGENT,
    canonical_key,
    categorise,
    department_of,
    doc_id_for,
)
from askiisc.ingest.editions import ACADEMIC_YEAR_START_MONTH
from askiisc.ingest.extract import extract_text

LOG = logging.getLogger(__name__)
CDX_URL = "https://web.archive.org/cdx/search/cdx"
RAW_URL = "https://web.archive.org/web/{ts}id_/{url}"
VIEW_URL = "https://web.archive.org/web/{ts}/{url}"
FIRST_ACADEMIC_YEAR = 2022
# Topic order: the pages questions about past years need most come first.
CATEGORY_ORDER = (
    "course-catalogue",
    "handbooks",
    "academic-policy",
    "admissions",
)
# An address must name one of these topics, and none of the noise words
# (course material, research output, news), to be looked up.
ARCHIVE_WORDS = (
    "curricul",
    "syllab",
    "scheme",
    "course",
    "programme",
    "program",
    "handbook",
    "brochure",
    "prospectus",
    "regulation",
    "rules",
    "ordinance",
    "fee",
    "scholarship",
    "calendar",
    "admission",
    "eligib",
    "academic",
    "credit",
    "timetable",
    "time-table",
)
NOISE_WORDS = (
    "/~",
    "thesis",
    "publication",
    "/events/",
    "/news/",
    "/lec",
    "lecture",
    "/hw",
    "homework",
    "solution",
    "exam",
    "paper",
    "colloquium",
    "seminar",
    "conference",
    "talk",
    "slides",
    "presentation",
    "notes",
    "assignment",
    "tutorial",
    "award",
    "feedback",
)
HTTP_TOO_MANY = 429
HTTP_SERVER_ERROR = 500
CHECKPOINT_EVERY = 25

Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class ArchiveConfig:
    """How far back and how fast."""

    first_year: int = FIRST_ACADEMIC_YEAR
    current_year: int = 2026
    delay_s: float = 1.5
    max_backoff_s: float = 600.0
    max_retries: int = 5
    max_bytes: int = 25_000_000
    deadline_s: float = 7.0 * 24 * 3600


@dataclass(frozen=True)
class Snapshot:
    """One CDX row."""

    timestamp: str
    digest: str
    mimetype: str

    @property
    def when(self) -> datetime:
        """The snapshot time (UTC)."""
        return datetime.strptime(self.timestamp, "%Y%m%d%H%M%S").replace(
            tzinfo=UTC
        )


@dataclass
class ArchiveReport:
    """Counts for the run log."""

    addresses: int = 0
    queried: int = 0
    saved: int = 0
    unchanged: int = 0
    too_little_text: int = 0
    errors: int = 0
    stopped_by: str = "done"
    by_year: dict[str, int] = field(default_factory=dict)


def academic_year(when: datetime) -> int:
    """The academic year (by its starting calendar year) of a date."""
    if when.month >= ACADEMIC_YEAR_START_MONTH:
        return when.year
    return when.year - 1


def wayback_digest(data: bytes) -> str:
    """The Wayback Machine's content digest: base32 SHA-1."""
    digest = hashlib.sha1(data, usedforsecurity=False).digest()
    return base64.b32encode(digest).decode()


def parse_cdx(rows: list[list[str]]) -> list[Snapshot]:
    """Snapshots from a CDX JSON answer (first row is the header)."""
    if not rows:
        return []
    header = rows[0]
    out = []
    for row in rows[1:]:
        values = dict(zip(header, row, strict=False))
        if values.get("statuscode", "200") != "200":
            continue
        out.append(
            Snapshot(
                values["timestamp"],
                values.get("digest", ""),
                values.get("mimetype", ""),
            )
        )
    return out


def choose_snapshots(
    snapshots: Iterable[Snapshot],
    config: ArchiveConfig,
    known_digests: set[str],
) -> list[Snapshot]:
    """The last snapshot of each past academic year, new content only.

    Args:
        snapshots: Every snapshot of one address.
        config: Which academic years count as past.
        known_digests: Digests already held (the live copy); updated.

    Returns:
        At most one snapshot per year, oldest first.
    """
    last: dict[int, Snapshot] = {}
    for snap in sorted(snapshots, key=lambda s: s.timestamp):
        year = academic_year(snap.when)
        if config.first_year <= year < config.current_year:
            last[year] = snap
    chosen = []
    for year in sorted(last):
        snap = last[year]
        if snap.digest and snap.digest in known_digests:
            continue
        known_digests.add(snap.digest)
        chosen.append(snap)
    return chosen


def worth_archiving(url: str) -> bool:
    """Whether an address names an academic topic and is not noise."""
    path = urlparse(url).path.lower()
    if any(word in path for word in NOISE_WORDS):
        return False
    return any(word in path for word in ARCHIVE_WORDS)


def targets(
    manifest_rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Academic addresses to look up, most needed topics first."""
    by_key: dict[str, dict[str, object]] = {}
    for row in manifest_rows:
        url = str(row["url"])
        topic = categorise(url, str(row.get("title") or ""))
        if topic not in PRIORITY_CATEGORIES or not worth_archiving(url):
            continue
        by_key.setdefault(canonical_key(url), {**row, "topic": topic})
    rank = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    return sorted(
        by_key.values(),
        key=lambda r: (rank.get(str(r["topic"]), 9), str(r["url"])),
    )


class ArchiveFetcher:
    """Looks up and saves past snapshots, politely and resumably."""

    def __init__(
        self,
        config: ArchiveConfig,
        client: httpx.AsyncClient,
        out_dir: Path,
        sleep: Sleep,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        self._config = config
        self._client = client
        self._out = out_dir
        self._sleep = sleep
        self._clock = clock
        self._wall = wall
        self._last = -1e18
        self._started = clock()
        self.done: set[str] = set()
        self.report = ArchiveReport()

    async def _get(
        self, url: str, params: dict[str, str] | None = None
    ) -> httpx.Response:
        backoff = self._config.delay_s
        for _ in range(self._config.max_retries):
            wait = self._last + self._config.delay_s - self._clock()
            if wait > 0:
                await self._sleep(wait)
            self._last = self._clock()
            response = await self._client.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT},
                follow_redirects=True,
                timeout=60.0,
            )
            status = response.status_code
            if status != HTTP_TOO_MANY and status < HTTP_SERVER_ERROR:
                return response
            backoff = min(backoff * 2, self._config.max_backoff_s)
            await self._sleep(backoff)
        return response

    async def snapshots(self, url: str) -> list[Snapshot]:
        """Every successful snapshot of `url` since the first year."""
        params = {
            "url": url,
            "output": "json",
            "fl": "timestamp,digest,mimetype,statuscode",
            "filter": "statuscode:200",
            "collapse": "digest",
            "from": str(self._config.first_year),
        }
        response = await self._get(CDX_URL, params)
        self.report.queried += 1
        if response.status_code != httpx.codes.OK or not response.text.strip():
            return []
        return parse_cdx(response.json())

    async def fetch_one(self, row: dict[str, object], live: bytes) -> int:
        """Saves the chosen snapshots of one address.

        Returns:
            How many were saved.
        """
        url = str(row["url"])
        known = {wayback_digest(live)} if live else set()
        chosen = choose_snapshots(
            await self.snapshots(url), self._config, known
        )
        saved = 0
        for snap in chosen:
            response = await self._get(
                RAW_URL.format(ts=snap.timestamp, url=url)
            )
            if response.status_code != httpx.codes.OK:
                continue
            if self._save(row, snap, response.content):
                saved += 1
        return saved

    def _save(
        self, row: dict[str, object], snap: Snapshot, data: bytes
    ) -> bool:
        kind = str(row["content_type"])
        if not data or len(data) > self._config.max_bytes:
            return False
        try:
            pages = extract_text(data, kind)
        except ExtractionError:
            self.report.too_little_text += 1
            return False
        if sum(len(p.text) for p in pages) < MIN_TEXT_CHARS:
            self.report.too_little_text += 1
            return False
        url = str(row["url"])
        department = department_of(urlparse(url).netloc)
        doc_id = doc_id_for(f"{url}@{snap.timestamp}")
        folder = self._out / "archive" / department
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{doc_id}{SUFFIX_OF[kind]}"
        path.write_bytes(data)
        entry = {
            "doc_id": doc_id,
            "path": str(path.relative_to(self._out)),
            "url": VIEW_URL.format(ts=snap.timestamp, url=url),
            "original_url": url,
            "snapshot": snap.when.timestamp(),
            "category": row.get("category", row.get("topic", "general")),
            "department": department,
            "visibility": "public",
            "content_type": kind,
            "etag": None,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "fetched_at": self._wall(),
            "title": "",
        }
        with (self._out / "archive.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")
        year = str(academic_year(snap.when))
        self.report.by_year[year] = self.report.by_year.get(year, 0) + 1
        return True

    def _over_time(self) -> bool:
        return self._clock() - self._started >= self._config.deadline_s

    async def run(self, rows: list[dict[str, object]]) -> ArchiveReport:
        """Processes every address not yet done, until the deadline."""
        self.report.addresses = len(rows)
        for row in rows:
            url = str(row["url"])
            if url in self.done:
                continue
            if self._over_time():
                self.report.stopped_by = "deadline"
                break
            try:
                saved = await self.fetch_one(row, _live_bytes(row))
            except (httpx.HTTPError, ValueError, OSError) as err:
                self.report.errors += 1
                LOG.warning("archive lookup failed for %s: %s", url, err)
                continue
            self.report.saved += saved
            self.report.unchanged += saved == 0
            self.done.add(url)
            if len(self.done) % CHECKPOINT_EVERY == 0:
                self.checkpoint()
        return self.report

    def load(self) -> None:
        """Restores the finished addresses."""
        state = self._out / "archive_state.json"
        if state.exists():
            self.done = set(json.loads(state.read_text())["done"])

    def checkpoint(self) -> None:
        """Writes the finished addresses (atomically)."""
        state = self._out / "archive_state.json"
        tmp = state.with_suffix(".tmp")
        tmp.write_text(json.dumps({"done": sorted(self.done)}))
        tmp.replace(state)


def _live_bytes(row: dict[str, object]) -> bytes:
    """The live copy's bytes, if the manifest row points at a file."""
    path = Path(str(row.get("live_path") or ""))
    return path.read_bytes() if row.get("live_path") and path.is_file() else b""


def read_manifests(paths: Iterable[Path]) -> list[dict[str, object]]:
    """Rows of several manifests, each with its file's absolute path."""
    rows: list[dict[str, object]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            row = json.loads(line)
            row["live_path"] = str(path.parent / str(row["path"]))
            rows.append(row)
    return rows


async def run_archive(  # pragma: no cover - network
    out: Path, manifests: list[Path], deadline_s: float
) -> ArchiveReport:
    """Looks up every academic address in the manifests."""
    import asyncio  # noqa: PLC0415

    config = ArchiveConfig(
        current_year=academic_year(datetime.now(UTC)), deadline_s=deadline_s
    )
    rows = targets(read_manifests(manifests))
    async with httpx.AsyncClient() as client:
        fetcher = ArchiveFetcher(config, client, out, asyncio.sleep)
        fetcher.load()
        try:
            return await fetcher.run(rows)
        finally:
            fetcher.checkpoint()


def main() -> None:  # pragma: no cover
    """Entry point: ASKIISC_CORPUS_DIR; manifests to read as arguments."""
    import asyncio  # noqa: PLC0415
    import os  # noqa: PLC0415
    import sys  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(os.environ["ASKIISC_CORPUS_DIR"])
    manifests = [Path(p) for p in sys.argv[1:]] or [out / "manifest.jsonl"]
    seconds = float(os.environ.get("ASKIISC_ARCHIVE_SECONDS", "6600"))
    report = asyncio.run(run_archive(out, manifests, seconds))
    text = json.dumps(asdict(report), indent=2)
    (out / "archive_report.json").write_text(text)
    print(text)


if __name__ == "__main__":  # pragma: no cover
    main()
