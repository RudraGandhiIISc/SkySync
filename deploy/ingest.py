"""Chunks, embeds and stores the corpus in a persistent index (a Slurm job).

State lives in one folder (default $ASKIISC_DATA/state) and is reused by
every later run of the stack:
  versions.db   every document version with ETag or content hash (FR-5)
  milvus.db     chunks and embeddings (Milvus Lite)
  objects/      the original files, by doc_id and version
The reconciler is seeded from versions.db, so a rerun only ingests
documents whose ETag or content changed.

Usage (inside slurm/ingest.sh):
    python -m deploy.ingest --corpus DIR --state-dir DIR --run-dir DIR
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import httpx

from askiisc.common.clock import SystemClock
from askiisc.common.types import ChunkDraft
from askiisc.ingest.corpus import (
    CorpusDoc,
    CorpusSource,
    load_corpus,
    write_policies,
)
from askiisc.ingest.pipeline import (
    Chunker,
    extract_and_chunk,
    retire_edition,
)
from askiisc.ingest.reconciler import Reconciler
from askiisc.ingest.versions import VersionStore
from askiisc.retrieval.embedder import HttpEmbedder
from askiisc.retrieval.store import ChunkIndex
from deploy.stack import embed_server, free_port, open_state, wait_http

LOG = logging.getLogger("askiisc.ingest")


def apply_windows(
    docs: list[CorpusDoc], versions: VersionStore, index: ChunkIndex
) -> int:
    """Retires every edition a newer edition has replaced.

    Covers documents ingested earlier whose newer edition arrived later.

    Returns:
        How many versions were retired now.
    """
    retired = 0
    for doc in docs:
        if doc.valid_to is None:
            continue
        try:
            retired += retire_edition(
                versions, index, doc.policy.doc_id, doc.valid_to
            )
        except ValueError as err:  # the catalogue moved this edition's start
            LOG.warning("window conflict for %s: %s", doc.policy.doc_id, err)
    return retired


def pool_chunker(pool: ProcessPoolExecutor) -> Chunker:
    """Extract-and-chunk in a process pool (pypdf is pure Python)."""

    async def chunk(
        data: bytes, content_type: str, template: frozenset[int]
    ) -> list[ChunkDraft]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            pool, extract_and_chunk, data, content_type, template
        )

    return chunk


async def ingest(
    corpus: Path,
    state: Path,
    run_dir: Path,
    deadline_s: float,
    db_dir: Path | None = None,
) -> dict[str, object]:
    """Ingests every new or changed document, stopping at the deadline."""
    started = time.monotonic()
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    docs = load_corpus(corpus)
    write_policies(run_dir, docs)  # a copy for this run; corpus stays read-only
    async with (
        httpx.AsyncClient() as http,
        embed_server(run_dir, free_port()) as url,
    ):
        pool = ProcessPoolExecutor(max_workers=workers)
        await wait_http(http, url + "/health", 300)
        embedder = HttpEmbedder(http, url, timeout_s=300.0)
        ingestor, versions = open_state(state, embedder, db_dir)
        ingestor.chunker = pool_chunker(pool)
        reconciler = Reconciler(
            CorpusSource(docs),
            ingestor.ingest,
            SystemClock(),
            concurrency=2 * workers,
        )
        reconciler.seed(versions.fingerprints())
        report = await reconciler.run_once(
            should_stop=lambda: time.monotonic() - started > deadline_s
        )
        retired = apply_windows(docs, versions, ingestor.index)
        pool.shutdown()
    chunks = sum(len(versions.current(d).drafts) for d in versions.documents())
    return {
        "documents": len(docs),
        "ingested": len(report.changed),
        "unchanged": len(report.unchanged),
        "failed": sorted(report.failed),
        "stopped_at_deadline": report.stopped,
        "editions_retired": retired,
        "chunks_current": chunks,
        "seconds": round(time.monotonic() - started, 1),
    }


def main() -> None:  # pragma: no cover
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--deadline-s", type=float, default=float("inf"))
    parser.add_argument("--db-dir", type=Path, default=None)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    result = asyncio.run(
        ingest(
            args.corpus,
            args.state_dir,
            args.run_dir,
            args.deadline_s,
            args.db_dir,
        )
    )
    (args.run_dir / "ingest_report.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result, indent=2))
    # 3: stopped at the deadline (the job resubmits itself); 1: failures.
    sys.exit(
        3 if result["stopped_at_deadline"] else 1 if result["failed"] else 0
    )


if __name__ == "__main__":  # pragma: no cover
    main()
