"""Runs the whole Ask IISc stack inside one Slurm job (docs/SERVER.md).

Services, all on 127.0.0.1 and free ports, stopped in reverse order:
  1. embedding server (serve venv, BGE-base on CPU), a subprocess
  2. access service (FastAPI, HTTP), its own uvicorn server
  3. Milvus Lite index, filled by the ingestion pipeline from a corpus dir
     (originals stored as files: MinIO has no community binaries any more)
  4. autoscaler controller with a ProcessPool of vLLM workers (0 at rest)
  5. gateway (FastAPI, SSE), its own uvicorn server
Then it runs the smoke questions through the gateway over HTTP, writes
results to the run directory and shuts everything down.

Usage (inside slurm/run_stack.sh):
    python -m deploy.stack --run-dir DIR [--corpus CORPUS_ROOT]
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import secrets
import socket
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import uvicorn

from askiisc.access.client import HttpAccessClient
from askiisc.access.directory import Directory
from askiisc.access.policy import PolicyStore
from askiisc.access.service import create_app as create_access
from askiisc.autoscaler.controller import Controller
from askiisc.autoscaler.policy import ScalingConfig
from askiisc.autoscaler.pool import ProcessPool, ProcessPoolConfig
from askiisc.common.clock import SystemClock
from askiisc.common.config import Settings
from askiisc.common.tokens import TokenIssuer
from askiisc.gateway.app import GatewayDeps
from askiisc.gateway.app import create_app as create_gateway
from askiisc.gateway.audit import FileAuditSink
from askiisc.gateway.auth import Authenticator
from askiisc.gateway.cache import InMemoryAnswerCache
from askiisc.gateway.drr import FairDispatcher
from askiisc.gateway.limits import Admission
from askiisc.ingest.corpus import CorpusDoc, CorpusSource, load_corpus
from askiisc.ingest.objects import FileObjectStore
from askiisc.ingest.pipeline import Ingestor
from askiisc.ingest.reconciler import Reconciler
from askiisc.ingest.versions import InMemoryVersionStore
from askiisc.llm.client import LLMClient
from askiisc.llm.models import select_model
from askiisc.llm.rag import RagAnswerer
from askiisc.observability.gpu import read_gpus
from askiisc.retrieval.embedder import HttpEmbedder
from askiisc.retrieval.search import Retriever
from askiisc.retrieval.store import MilvusIndex

LOG = logging.getLogger("askiisc.stack")
SAMPLE = Path("eval/datasets/sample")
PER_WORKER_SLOTS = 32


def free_port() -> int:
    """Returns a free TCP port on 127.0.0.1."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def wait_http(
    http: httpx.AsyncClient, url: str, timeout_s: float
) -> None:
    """Polls a health URL until it answers 200."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        with contextlib.suppress(httpx.HTTPError):
            if (await http.get(url, timeout=2.0)).status_code == httpx.codes.OK:
                return
        await asyncio.sleep(1.0)
    raise RuntimeError(f"{url} not healthy after {timeout_s} s")


@contextlib.asynccontextmanager
async def serve(app: Any, port: int) -> AsyncIterator[None]:
    """Runs an ASGI app on 127.0.0.1:port for the duration of the block."""
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        if task.done():
            task.result()
        await asyncio.sleep(0.05)
    try:
        yield
    finally:
        server.should_exit = True
        await task


@contextlib.asynccontextmanager
async def embed_server(run_dir: Path, port: int) -> AsyncIterator[str]:
    """Starts the CPU embedding server from the serve venv."""
    python = Path(os.environ["SERVE_VENV"]) / "bin" / "python"
    env = {**os.environ, "PYTHONPATH": "src", "HF_HUB_OFFLINE": "1"}
    with (run_dir / "embed.log").open("ab") as log:
        proc = await asyncio.create_subprocess_exec(
            str(python),
            "-m",
            "askiisc.retrieval.embed_server",
            "--port",
            str(port),
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        await proc.wait()


def load_world(
    directory_path: str = "", corpus: Path = SAMPLE / "corpus"
) -> tuple[Directory, PolicyStore, list[CorpusDoc]]:
    """Loads the users tree, the corpus and its policies.

    Args:
        directory_path: The users tree or export (ASKIISC_DIRECTORY_PATH).
            Real users live only in /data4/.../private/users, never in git;
            empty means the mock users in eval/datasets/sample/users.
        corpus: A corpus root (public/, restricted/, manifest.jsonl).

    Returns:
        The directory, the policy store and the corpus documents.
    """
    directory = Directory()
    directory.load(Path(directory_path) if directory_path else SAMPLE / "users")
    docs = load_corpus(corpus)
    return directory, PolicyStore([d.policy for d in docs]), docs


async def ingest_corpus(docs: list[CorpusDoc], ingestor: Ingestor) -> int:
    """Ingests the corpus through the reconciler (ETag or content hash)."""
    reconciler = Reconciler(CorpusSource(docs), ingestor.ingest, SystemClock())
    report = await reconciler.run_once()
    if report.failed:
        LOG.warning("%d documents failed to ingest", len(report.failed))
    return len(report.changed)


def make_pool(
    settings: Settings, run_dir: Path, http: httpx.AsyncClient
) -> ProcessPool:
    """Builds the vLLM worker pool for the configured model."""
    import pynvml  # noqa: PLC0415

    pynvml.nvmlInit()
    spec = select_model(settings.model)
    config = ProcessPoolConfig(
        vllm_bin=str(Path(os.environ["SERVE_VENV"]) / "bin" / "vllm"),
        model_args=spec.vllm_args(),
        log_dir=str(run_dir),
        max_workers=settings.max_workers,
    )
    return ProcessPool(
        config, SystemClock(), http, lambda: read_gpus(pynvml), free_port
    )


async def ask_all(
    http: httpx.AsyncClient, url: str, issuer: TokenIssuer, out: Path
) -> list[dict[str, Any]]:
    """Sends the smoke questions through the gateway and records events."""
    questions = json.loads((SAMPLE / "questions.json").read_text())
    results = []
    for q in questions:
        token = issuer.issue(q["user"], SystemClock().now() + 600)
        started = asyncio.get_running_loop().time()
        r = await http.post(
            url + "/v1/ask",
            json={"question": q["question"]},
            headers={"Authorization": f"Bearer {token}"},
            timeout=300.0,
        )
        seconds = asyncio.get_running_loop().time() - started
        events = [block for block in r.text.split("\n\n") if block.strip()]
        final = next(
            (
                json.loads(b.split("data: ", 1)[1])
                for b in events
                if b.startswith("event: answer")
            ),
            None,
        )
        kinds = [b.split("\n", 1)[0].removeprefix("event: ") for b in events]
        results.append(
            {
                **q,
                "status": r.status_code,
                "seconds": seconds,
                "events": sorted(set(kinds)),
                "answer": final,
            }
        )
        LOG.info("%s status=%s %.1fs", q["id"], r.status_code, seconds)
    out.write_text(json.dumps(results, indent=2))
    return results


async def run(run_dir: Path, corpus: Path) -> int:
    """Brings the stack up, runs the smoke questions, tears down."""
    settings = Settings()
    clock = SystemClock()
    issuer = TokenIssuer()
    directory, policies, docs = load_world(settings.directory_path, corpus)
    ports = {k: free_port() for k in ("embed", "access", "gateway")}
    (run_dir / "ports.json").write_text(json.dumps(ports))
    async with (
        httpx.AsyncClient() as http,
        embed_server(run_dir, ports["embed"]) as embed_url,
    ):
        await wait_http(http, embed_url + "/health", 300)
        access_app = create_access(
            directory, policies, issuer.verifier(clock), clock
        )
        async with serve(access_app, ports["access"]):
            return await _with_access(
                settings,
                run_dir,
                docs,
                http,
                clock,
                issuer,
                directory,
                embed_url,
                f"http://127.0.0.1:{ports['access']}",
                ports["gateway"],
            )


async def _with_access(
    settings: Settings,
    run_dir: Path,
    docs: list[CorpusDoc],
    http: httpx.AsyncClient,
    clock: SystemClock,
    issuer: TokenIssuer,
    directory: Directory,
    embed_url: str,
    access_url: str,
    gateway_port: int,
) -> int:
    from pymilvus import MilvusClient  # noqa: PLC0415

    embedder = HttpEmbedder(http, embed_url)
    index = MilvusIndex(MilvusClient(str(run_dir / "milvus.db")))
    ingestor = Ingestor(
        InMemoryVersionStore(),
        FileObjectStore(run_dir / "objects"),
        embedder,
        index,
        commit_delay_s=0.0,
    )
    LOG.info("ingested %d documents", await ingest_corpus(docs, ingestor))
    access = HttpAccessClient(http, access_url, clock)
    dispatcher = FairDispatcher(capacity=PER_WORKER_SLOTS, quantum=2048)
    controller = Controller(
        make_pool(settings, run_dir, http),
        ScalingConfig(
            max_workers=settings.max_workers,
            idle_timeout_s=settings.idle_timeout_s,
        ),
        clock,
        on_ready_change=lambda n: dispatcher.resize(
            PER_WORKER_SLOTS * max(1, n)
        ),
    )
    llm = LLMClient(
        http,
        controller.endpoint,
        select_model(settings.model).served_name,
        timeout_s=settings.llm_timeout_s,
    )
    deps = GatewayDeps(
        clock=clock,
        auth=Authenticator(issuer.verifier(clock), directory),
        access=access,
        rag=RagAnswerer(Retriever(access, embedder, index, clock), llm, clock),
        admission=Admission(
            clock,
            settings.tenant_rate_per_s,
            settings.tenant_burst,
            settings.tenant_queue_max,
            settings.max_cold_hold_s + 60,
            concurrency=32,
        ),
        dispatcher=dispatcher,
        cache=InMemoryAnswerCache(clock),
        audit=FileAuditSink(run_dir / "audit.jsonl"),
        audit_salt=secrets.token_hex(16),
        workers=controller,
        max_cold_hold_s=settings.max_cold_hold_s,
        debug_retrieval=True,
    )
    loop_task = asyncio.create_task(controller.run(interval_s=1.0))
    try:
        async with serve(create_gateway(deps), gateway_port):
            results = await ask_all(
                http,
                f"http://127.0.0.1:{gateway_port}",
                issuer,
                run_dir / "results.json",
            )
    finally:
        await controller.close()
        loop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop_task
    timelines = [
        {
            k: getattr(w, k)
            for k in (
                "worker_id",
                "gpu",
                "requested_at",
                "acquired_at",
                "ready_at",
                "released_at",
            )
        }
        for w in controller.history
    ]
    (run_dir / "workers.json").write_text(json.dumps(timelines, indent=2))
    return 0 if all(r["status"] == httpx.codes.OK for r in results) else 1


def main() -> None:  # pragma: no cover
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=SAMPLE / "corpus")
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s"
    )
    sys.exit(asyncio.run(run(args.run_dir, args.corpus)))


if __name__ == "__main__":  # pragma: no cover
    main()
