"""Retrieval-only benchmark of the edition-aware question set.

For each search setting, runs every question through the serving
retrieval path (time scope, filtered hybrid search, reranker, merge of
the as-of searches, prompt budget) and measures what reaches the prompt:

  doc_recall    an expected document is in the prompt
  fact_recall   every key fact is in the prompt text (lenient match)
  wrong_rate    a wrong edition is in the prompt
  chunks        chunks in the prompt (mean)

No language model is needed, so settings compare in minutes.

Usage (inside slurm/llm_job.sh or any GPU job with the embed server):
    python -m eval.versioned.retrieval QUESTIONS --corpus DIR --state DIR \
        --out DIR
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx
from deploy.stack import embed_server, free_port, load_world, wait_http

from askiisc.access.client import LocalAccess
from askiisc.access.policy import PolicyEngine
from askiisc.common.clock import SystemClock
from askiisc.common.config import Settings
from askiisc.common.tokens import TokenIssuer
from askiisc.ingest.versions import SQLiteVersionStore
from askiisc.llm.prompts import fit_context
from askiisc.llm.rag import merge_hits
from askiisc.retrieval.dense import open_dense
from askiisc.retrieval.embedder import HttpEmbedder, query_prefix
from askiisc.retrieval.rerank import HttpReranker
from askiisc.retrieval.search import Retriever
from askiisc.retrieval.store import MilvusIndex
from askiisc.retrieval.temporal import time_scope, without_time
from eval.grader.rubric import lenient_match

LOG = logging.getLogger("askiisc.eval.retrieval")
CONCURRENCY = 16


@dataclass(frozen=True)
class Setting:
    """One retrieval configuration."""

    name: str
    candidates: int
    top_k: int
    max_chars: int
    rerank: bool = True
    strip_time: bool = False


SETTINGS = (
    Setting("v3-default", 40, 8, 9000),
    Setting("serving", 80, 12, 16000, strip_time=True),
    Setting("k16-20k", 80, 16, 20000, strip_time=True),
)


def measure(item: dict[str, Any], chunks: list[Any]) -> dict[str, Any]:
    """What one question's prompt context holds."""
    docs = {c.doc_id for c in chunks}
    text = "\n".join(c.text for c in chunks)
    facts = item.get("key_facts") or []
    return {
        "id": item["id"],
        "type": item["type"],
        "doc": bool(docs & set(item.get("expected_docs") or [])),
        "fact": bool(facts) and all(lenient_match(f, text) for f in facts),
        "wrong": bool(docs & set(item.get("wrong_docs") or [])),
        "chunks": len(chunks),
    }


async def contexts(
    retriever: Retriever,
    issuer: TokenIssuer,
    items: list[dict[str, Any]],
    setting: Setting,
) -> list[dict[str, Any]]:
    """The prompt context of every answerable item under one setting."""
    gate = asyncio.Semaphore(CONCURRENCY)
    clock = SystemClock()

    async def one(item: dict[str, Any]) -> dict[str, Any]:
        async with gate:
            token = issuer.issue(item["user"], clock.now() + 600)
            scope = time_scope(item["question"], clock.now())
            query = item["question"]
            if setting.strip_time and scope.label:
                query = without_time(query)
            results = [
                await retriever.search(token, query, at)
                for at in scope.instants
            ]
            chunks = fit_context(
                [h.chunk for h in merge_hits(results)], setting.max_chars
            )
            return measure(item, list(chunks))

    return list(await asyncio.gather(*(one(i) for i in items)))


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recall rates overall and per type."""

    def rates(rs: list[dict[str, Any]]) -> dict[str, float]:
        n = max(len(rs), 1)
        return {
            "items": len(rs),
            "doc_recall": round(sum(r["doc"] for r in rs) / n, 4),
            "fact_recall": round(sum(r["fact"] for r in rs) / n, 4),
            "wrong_rate": round(sum(r["wrong"] for r in rs) / n, 4),
            "chunks": round(sum(r["chunks"] for r in rs) / n, 2),
        }

    types = sorted({r["type"] for r in rows})
    return {
        "all": rates(rows),
        **{t: rates([r for r in rows if r["type"] == t]) for t in types},
    }


async def run(questions: Path, corpus: Path, state: Path, out: Path) -> None:
    """Measures every setting; writes retrieval.json."""
    from pymilvus import MilvusClient  # noqa: PLC0415

    settings = Settings()
    clock = SystemClock()
    issuer = TokenIssuer()
    directory, policies, _ = load_world("", corpus)
    items = [
        i
        for i in map(json.loads, questions.open())
        if i["type"] != "unanswerable"
    ]
    versions = SQLiteVersionStore(state / "versions.db")
    client = MilvusClient(str(state / "milvus.db"))
    index = await asyncio.to_thread(
        open_dense, state, MilvusIndex(client), client, versions.stamp()
    )
    report: dict[str, Any] = {}
    async with (
        httpx.AsyncClient(limits=httpx.Limits(max_connections=64)) as http,
        embed_server(out, free_port()) as embed,
    ):
        await wait_http(http, embed.url + "/health", 600)
        access = LocalAccess(
            PolicyEngine(directory, policies), issuer.verifier(clock), clock
        )
        for setting in SETTINGS:
            retriever = Retriever(
                access,
                HttpEmbedder(
                    http, embed.url, timeout_s=60.0, api_key=embed.key
                ),
                index,
                clock,
                top_k=setting.top_k,
                query_prefix=query_prefix(settings.embed_model),
                reranker=(
                    HttpReranker(
                        http, embed.url, timeout_s=120.0, api_key=embed.key
                    )
                    if setting.rerank
                    else None
                ),
                candidates=setting.candidates,
            )
            rows = await contexts(retriever, issuer, items, setting)
            report[setting.name] = asdict(setting) | summarise(rows)
            LOG.info("%s %s", setting.name, report[setting.name]["all"])
            (out / "retrieval.json").write_text(json.dumps(report, indent=1))


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
