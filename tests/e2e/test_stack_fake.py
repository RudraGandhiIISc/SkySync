"""End to end over real HTTP with the fake LLM (M1 on the fake stack).

Every component is a real server on 127.0.0.1: fake vLLM, access service,
gateway. Retrieval uses Milvus Lite. The autoscaler starts from zero
workers, so the first question is held with a warming event.
"""

import asyncio
import json
from pathlib import Path

import httpx
from deploy.stack import ingest_corpus, load_world

from askiisc.access.client import HttpAccessClient
from askiisc.access.service import create_app as create_access
from askiisc.autoscaler.controller import Controller
from askiisc.autoscaler.policy import ScalingConfig
from askiisc.autoscaler.pool import FakeWorkerPool
from askiisc.common.clock import SystemClock
from askiisc.common.tokens import TokenIssuer
from askiisc.gateway.app import GatewayDeps
from askiisc.gateway.app import create_app as create_gateway
from askiisc.gateway.audit import FileAuditSink
from askiisc.gateway.auth import Authenticator
from askiisc.gateway.cache import InMemoryAnswerCache
from askiisc.gateway.drr import FairDispatcher
from askiisc.gateway.limits import Admission
from askiisc.ingest.objects import FileObjectStore
from askiisc.ingest.pipeline import Ingestor
from askiisc.ingest.versions import InMemoryVersionStore
from askiisc.llm.client import LLMClient
from askiisc.llm.fake_server import FakeLLMConfig
from askiisc.llm.fake_server import create_app as create_llm
from askiisc.llm.rag import RagAnswerer
from askiisc.retrieval.embedder import FakeEmbedder
from askiisc.retrieval.search import Retriever
from askiisc.retrieval.store import MilvusIndex
from tests.integration.conftest import ThreadServer


def _events(text: str) -> list[tuple[str, dict[str, object]]]:
    out = []
    for block in text.split("\n\n"):
        if block.strip():
            lines = dict(ln.split(": ", 1) for ln in block.splitlines())
            out.append((lines["event"], json.loads(lines["data"])))
    return out


async def test_full_stack_over_http(tmp_path: Path) -> None:
    from pymilvus import MilvusClient

    clock, issuer = SystemClock(), TokenIssuer()
    directory, policies, docs = load_world()  # mock users and corpus
    llm_server = ThreadServer(
        create_llm(FakeLLMConfig(model="fake", tokens_per_second=500.0), clock)
    ).start()
    access_server = ThreadServer(
        create_access(directory, policies, issuer.verifier(clock), clock)
    ).start()
    http = httpx.AsyncClient()  # the test's own client
    # The gateway runs on another thread's event loop, so it gets its own
    # client; its connections die with that loop.
    gw_http = httpx.AsyncClient()
    embedder = FakeEmbedder(dim=64)
    index = MilvusIndex(MilvusClient(str(tmp_path / "m.db")), dim=64)
    ingestor = Ingestor(
        InMemoryVersionStore(),
        FileObjectStore(tmp_path / "o"),
        embedder,
        index,
        commit_delay_s=0.0,
    )
    assert await ingest_corpus(docs, ingestor) == len(docs)
    pool = FakeWorkerPool(
        clock, acquire_s=0.2, load_s=0.5, base_url=llm_server.url
    )
    controller = Controller(
        pool,
        ScalingConfig(idle_timeout_s=2.0, hysteresis_s=0.5),
        clock,
        poll_s=0.05,
    )
    access = HttpAccessClient(gw_http, access_server.url, clock, timeout_s=2.0)
    audit = FileAuditSink(tmp_path / "audit.jsonl")
    deps = GatewayDeps(
        clock=clock,
        auth=Authenticator(issuer.verifier(clock), directory),
        access=access,
        rag=RagAnswerer(
            Retriever(access, embedder, index, clock),
            LLMClient(gw_http, controller.endpoint, "fake"),
            clock,
        ),
        admission=Admission(clock, 0.5, 3, 10, 30, concurrency=8),
        dispatcher=FairDispatcher(8, 2048),
        cache=InMemoryAnswerCache(clock),
        audit=audit,
        audit_salt="e2e",
        workers=controller,
        debug_retrieval=True,
    )
    gateway = ThreadServer(create_gateway(deps)).start()
    loop = asyncio.create_task(controller.run(interval_s=0.05))

    async def ask(user: str, question: str) -> httpx.Response:
        token = issuer.issue(user, clock.now() + 60)
        return await http.post(
            gateway.url + "/v1/ask",
            json={"question": question},
            timeout=30,
            headers={"Authorization": f"Bearer {token}"},
        )

    try:
        cold = _events((await ask("stu_cds", "hostel fee late fee")).text)
        assert cold[0][0] == "warming" and cold[-1][0] == "answer"
        answer = cold[-1][1]
        assert answer["citations"][0]["doc_id"] == "inst-hostel-fees"  # type: ignore[index]
        forbidden = _events(
            (await ask("stu_ece", "CDS thesis external examiners")).text
        )
        retrieved = {d for _, d in forbidden[-1][1]["retrieved"]}  # type: ignore[union-attr]
        assert "cds-thesis" not in retrieved
        codes = [
            (await ask("stu_cds", f"exam attendance {i}")).status_code
            for i in range(4)
        ]
        assert 429 in codes  # burst of 3 per tenant
        no_token = await http.post(
            gateway.url + "/v1/ask", json={"question": "x"}
        )
        assert no_token.status_code == 401
        await asyncio.sleep(3.5)  # idle timeout passes: scaled to zero
        assert pool.list_workers() == []
        access_server.stop()
        down = await ask("staff_acc", "travel allowance")
        assert down.status_code == 403
        records = audit.read()
        assert {r.decision for r in records} >= {
            "answered",
            "rejected",
            "denied",
        }
        raw = (tmp_path / "audit.jsonl").read_text()
        # Doc ids may appear; question words and user ids may not.
        assert "late fee" not in raw and "examiners" not in raw
        assert "stu_cds" not in raw
    finally:
        await controller.close()
        loop.cancel()
        gateway.stop()
        llm_server.stop()
        await http.aclose()
