"""Answers the edition-aware question set with the serving retrieval path.

The same components as the serving job, minus admission and caching: the
dense snapshot of the state folder (vector and BM25 search with permission
and validity masks), the access engine, the cross-encoder reranker, the
time scope of each question, edition labels in the prompt, and the model
under test (vLLM started by slurm/llm_job.sh).

Usage (inside slurm/llm_job.sh with MODEL=<model under test>):
    python -m eval.versioned.answer QUESTIONS --corpus DIR --state DIR --out DIR
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx
from deploy.serve import doc_catalogue
from deploy.stack import embed_server, free_port, load_world, wait_http

from askiisc.access.client import LocalAccess
from askiisc.access.policy import PolicyEngine
from askiisc.common.clock import SystemClock
from askiisc.common.config import Settings
from askiisc.common.tokens import TokenIssuer
from askiisc.ingest.versions import SQLiteVersionStore
from askiisc.llm.client import LLMClient
from askiisc.llm.models import select_model
from askiisc.llm.rag import RagAnswerer
from askiisc.retrieval.dense import open_dense
from askiisc.retrieval.embedder import HttpEmbedder, query_prefix
from askiisc.retrieval.rerank import HttpReranker
from askiisc.retrieval.search import Retriever
from askiisc.retrieval.store import MilvusIndex
from askiisc.retrieval.temporal import time_scope

LOG = logging.getLogger("askiisc.eval.answer")
CONCURRENCY = 16


def record(
    item: dict[str, Any], answer: Any, prepared: Any, seconds: float, now: float
) -> dict[str, Any]:
    """One result row: the answer, its citations and what was retrieved."""
    return {
        **item,
        "answer": answer.text,
        # The model's own text before the output guards (eval only).
        "raw": "".join(prepared.state.text),
        "abstained": answer.abstained,
        "citations": [
            {
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "page": c.page,
                "stale": c.is_stale,
            }
            for c in answer.citations
        ],
        "retrieved": [c.id for c in prepared.chunks],
        "retrieved_docs": sorted({c.doc_id for c in prepared.chunks}),
        "time_scope": time_scope(item["question"], now).label,
        "seconds": round(seconds, 2),
    }


async def answer_all(
    rag: RagAnswerer, issuer: TokenIssuer, items: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Answers every item, CONCURRENCY at a time, in order."""
    gate = asyncio.Semaphore(CONCURRENCY)
    clock = SystemClock()

    async def one(item: dict[str, Any]) -> dict[str, Any]:
        async with gate:
            token = issuer.issue(item["user"], clock.now() + 600)
            started = time.monotonic()
            answer, prepared = await rag.answer(token, item["question"])
            row = record(
                item, answer, prepared, time.monotonic() - started, clock.now()
            )
            LOG.info("%s %.1fs", item["id"], row["seconds"])
            return row

    return list(await asyncio.gather(*(one(i) for i in items)))


async def run(questions: Path, corpus: Path, state: Path, out: Path) -> None:
    """Brings up retrieval, answers the questions, writes answers.json."""
    from pymilvus import MilvusClient  # noqa: PLC0415

    settings = Settings()
    clock = SystemClock()
    issuer = TokenIssuer()
    directory, policies, docs = load_world("", corpus)
    items = [json.loads(ln) for ln in questions.open() if ln.strip()]
    versions = SQLiteVersionStore(state / "versions.db")
    client = MilvusClient(str(state / "milvus.db"))
    index = await asyncio.to_thread(
        open_dense, state, MilvusIndex(client), client, versions.stamp()
    )
    async with (
        httpx.AsyncClient(limits=httpx.Limits(max_connections=64)) as http,
        embed_server(out, free_port()) as embed,
    ):
        await wait_http(http, embed.url + "/health", 600)
        access = LocalAccess(
            PolicyEngine(directory, policies), issuer.verifier(clock), clock
        )
        retriever = Retriever(
            access,
            HttpEmbedder(http, embed.url, timeout_s=60.0, api_key=embed.key),
            index,
            clock,
            query_prefix=query_prefix(settings.embed_model),
            reranker=(
                HttpReranker(http, embed.url, timeout_s=60.0, api_key=embed.key)
                if settings.rerank_model
                else None
            ),
            candidates=settings.rerank_candidates,
        )
        llm = LLMClient(
            http,
            lambda: os.environ["ASKIISC_LLM_URL"],
            os.environ["ASKIISC_LLM_MODEL"],
            timeout_s=120.0,
            extra_body=select_model(settings.model).request_extra(),
        )
        rag = RagAnswerer(
            retriever, llm, clock, labels=doc_catalogue(corpus, docs)
        )
        results = await answer_all(rag, issuer, items)
    (out / "answers.json").write_text(json.dumps(results, indent=1))
    LOG.info("%d answers -> %s", len(results), out / "answers.json")


def main() -> None:  # pragma: no cover - needs the GPU job
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run(args.questions, args.corpus, args.state, args.out))


if __name__ == "__main__":  # pragma: no cover
    main()
