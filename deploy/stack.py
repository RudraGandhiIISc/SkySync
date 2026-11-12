"""Helpers shared by the jobs that start Ask IISc services.

Free ports and health waits, the embedding server (serve venv), the world
(directory, policies, corpus), the index state, a test ingest, and the
vLLM worker pool. Used by deploy/serve.py, deploy/ingest.py and the
evaluation jobs (eval/versioned); every service binds to 127.0.0.1.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import socket
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

from askiisc.access.directory import Directory
from askiisc.access.policy import PolicyStore
from askiisc.autoscaler.controller import Controller
from askiisc.autoscaler.policy import ScalingConfig
from askiisc.autoscaler.pool import (
    ProcessPool,
    ProcessPoolConfig,
    Worker,
    WorkerPool,
)
from askiisc.common.clock import Clock, SystemClock
from askiisc.common.config import Settings
from askiisc.common.servicekey import KEY_ENV, new_key
from askiisc.common.targets import EMBED_MODEL, RERANK_MODEL
from askiisc.common.types import DocId
from askiisc.gateway.drr import FairDispatcher
from askiisc.ingest.corpus import CorpusDoc, CorpusSource, load_corpus
from askiisc.ingest.objects import FileObjectStore
from askiisc.ingest.pipeline import Ingestor
from askiisc.ingest.reconciler import Reconciler
from askiisc.ingest.versions import SQLiteVersionStore
from askiisc.llm.models import select_model
from askiisc.observability.gpu import (
    descends_from,
    read_gpus,
    read_processes,
)
from askiisc.retrieval.embedder import HttpEmbedder
from askiisc.retrieval.store import MilvusIndex

LOG = logging.getLogger("askiisc.stack")
SAMPLE = Path("eval/datasets/sample")


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


@dataclasses.dataclass(frozen=True)
class Endpoint:
    """A service the job started: its loopback URL and its key."""

    url: str
    key: str = dataclasses.field(repr=False)


def embed_command(port: int, key: str) -> tuple[list[str], dict[str, str]]:
    """The embedding (and reranking) server's command and environment.

    ASKIISC_EMBED_DEVICE picks the device (cpu or cuda), ASKIISC_EMBED_MODEL
    the embedder, ASKIISC_RERANK_MODEL the reranker (empty: none). The key
    goes in the environment, never on the command line (other users can
    read command lines).

    Args:
        port: The loopback port.
        key: The key every request but /health must carry.

    Returns:
        The command and its environment.
    """
    python = Path(os.environ["SERVE_VENV"]) / "bin" / "python"
    env = {
        **os.environ,
        "PYTHONPATH": "src",
        "HF_HUB_OFFLINE": "1",
        KEY_ENV: key,
    }
    gpus = allowed_gpus()
    if os.environ.get("ASKIISC_EMBED_DEVICE", "cpu") == "cuda" and gpus:
        env["CUDA_VISIBLE_DEVICES"] = str(min(gpus))
    cmd = [
        str(python),
        "-m",
        "askiisc.retrieval.embed_server",
        "--port",
        str(port),
        "--device",
        os.environ.get("ASKIISC_EMBED_DEVICE", "cpu"),
        "--model",
        os.environ.get("ASKIISC_EMBED_MODEL", EMBED_MODEL),
        "--reranker",
        os.environ.get("ASKIISC_RERANK_MODEL", RERANK_MODEL),
    ]
    return cmd, env


@contextlib.asynccontextmanager
async def embed_server(run_dir: Path, port: int) -> AsyncIterator[Endpoint]:
    """Starts the embedding server with a fresh key, for a batch job."""
    key = new_key()
    cmd, env = embed_command(port, key)
    with (run_dir / "embed.log").open("ab") as log:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            env=env,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    try:
        yield Endpoint(f"http://127.0.0.1:{port}", key)
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


def open_state(
    state: Path, embedder: HttpEmbedder, db_dir: Path | None = None
) -> tuple[Ingestor, SQLiteVersionStore]:
    """Opens (or creates) the persistent stores in a state folder.

    versions.db holds every document version with its ETag or content hash,
    milvus.db the chunks and embeddings, objects/ the original files.
    `db_dir` keeps the two databases elsewhere (local disk during a bulk
    ingest: each commit on the NFS drive takes ~100 ms).
    """
    from pymilvus import MilvusClient  # noqa: PLC0415

    state.mkdir(parents=True, exist_ok=True)
    db_dir = db_dir or state
    db_dir.mkdir(parents=True, exist_ok=True)
    versions = SQLiteVersionStore(db_dir / "versions.db")
    ingestor = Ingestor(
        versions,
        FileObjectStore(state / "objects"),
        embedder,
        MilvusIndex(MilvusClient(str(db_dir / "milvus.db"))),
        commit_delay_s=0.0,
    )
    return ingestor, versions


async def ingest_corpus(
    docs: list[CorpusDoc],
    ingestor: Ingestor,
    seen: dict[DocId, str] | None = None,
) -> int:
    """Ingests new or changed documents through the reconciler.

    Args:
        docs: The corpus.
        ingestor: Where documents go.
        seen: Fingerprints already ingested (from the version store); those
            documents are skipped unless their ETag or content changed.

    Returns:
        The number of documents ingested.
    """
    reconciler = Reconciler(CorpusSource(docs), ingestor.ingest, SystemClock())
    reconciler.seed(seen or {})
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
        model_args=[
            *spec.vllm_args(),
            *os.environ.get("ASKIISC_VLLM_EXTRA_ARGS", "").split(),
        ],
        log_dir=str(run_dir),
        max_workers=settings.max_workers,
        min_memory_fraction=spec.min_gpu_fraction,
        prefix_caching=settings.prefix_cache,
        api_key=new_key(),
        source_dir=str(Path("src").resolve()),
    )
    allowed = allowed_gpus()
    return ProcessPool(
        config,
        SystemClock(),
        http,
        lambda: read_gpus(pynvml, allowed),
        free_port,
        processes=lambda index: read_processes(pynvml, index),
        is_ours=descends_from(os.getpid()),
    )


def write_workers(path: Path, workers: list[Worker]) -> None:
    """Writes every worker's lifecycle and card state (workers.json).

    The cost model (NVML 6), the cold-start split (NFR-2) and the shared-card
    flag (R-3) are read from this file after a run. Written atomically.

    Args:
        path: The file.
        workers: Every worker the pool started, released ones included.
    """
    rows = [{**dataclasses.asdict(w), "state": str(w.state)} for w in workers]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, indent=1))
    tmp.replace(path)


def make_controller(
    pool: WorkerPool,
    config: ScalingConfig,
    clock: Clock,
    dispatcher: FairDispatcher,
    slots: int,
) -> Controller:
    """Builds the autoscaler, fed by the fair queue's load.

    The controller sees the requests being generated and waiting, so it
    never releases a worker while a request may be streaming from it
    (FR-10: held requests complete), and the queue's slots follow the
    number of ready workers.

    Args:
        pool: The worker pool.
        config: Scaling limits and timeouts.
        clock: The clock.
        dispatcher: The gateway's fair queue.
        slots: LLM slots per ready worker.

    Returns:
        The controller.
    """
    return Controller(
        pool,
        config,
        clock,
        in_flight=lambda: dispatcher.in_use,
        queue_depth=lambda: dispatcher.waiting,
        on_ready_change=lambda n: dispatcher.resize(slots * max(1, n)),
    )


def allowed_gpus() -> frozenset[int] | None:
    """ASKIISC_GPUS (comma-separated card indexes), or None for any card.

    The serving job picks its two cards up front, so the embedding and
    reranking server shares a card with a vLLM worker and the job never
    touches more than two GPUs.
    """
    raw = os.environ.get("ASKIISC_GPUS", "").strip()
    if not raw:
        return None
    return frozenset(int(x) for x in raw.split(","))
