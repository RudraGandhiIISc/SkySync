"""A whole gateway on fakes: fake LLM, in-process access, memory index."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx

from askiisc.access.client import LocalAccess
from askiisc.access.policy import PolicyEngine
from askiisc.common.clock import FakeClock
from askiisc.gateway.app import (
    AlwaysReady,
    GatewayDeps,
    WorkerStatus,
    create_app,
)
from askiisc.gateway.audit import InMemoryAuditSink
from askiisc.gateway.auth import Authenticator
from askiisc.gateway.cache import InMemoryAnswerCache
from askiisc.gateway.drr import FairDispatcher
from askiisc.gateway.limits import Admission
from askiisc.llm.client import LLMClient
from askiisc.llm.fake_server import FakeLLMConfig
from askiisc.llm.fake_server import create_app as create_llm
from askiisc.llm.rag import RagAnswerer
from askiisc.retrieval.search import Retriever
from tests import corpus
from tests.access_fixtures import AccessWorld, world

SALT = "test-salt"


@dataclass
class Stack:
    w: AccessWorld
    deps: GatewayDeps
    client: httpx.AsyncClient
    audit: InMemoryAuditSink
    generation: list[int] = field(default_factory=lambda: [0])

    async def ask(
        self, user: str | None, question: str, **body: object
    ) -> httpx.Response:
        headers = (
            {}
            if user is None
            else {"Authorization": f"Bearer {self.w.token(user)}"}
        )
        return await self.client.post(
            "/v1/ask", json={"question": question, **body}, headers=headers
        )


def events(response: httpx.Response) -> list[tuple[str, dict[str, object]]]:
    out = []
    for block in response.text.split("\n\n"):
        if not block.strip():
            continue
        lines = dict(ln.split(": ", 1) for ln in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def final(response: httpx.Response) -> dict[str, object]:
    return next(p for e, p in events(response) if e == "answer")


def build(
    rate: float = 100.0,
    burst: float = 100.0,
    queue_max: int = 100,
    workers: WorkerStatus | None = None,
    debug: bool = True,
) -> Stack:
    w = world()
    access = LocalAccess(
        PolicyEngine(w.directory, w.policies),
        w.issuer.verifier(w.clock),
        w.clock,
    )
    llm_app = create_llm(FakeLLMConfig(model="fake-model"), FakeClock())
    llm_http = httpx.AsyncClient(transport=httpx.ASGITransport(app=llm_app))
    llm = LLMClient(llm_http, "http://llm", "fake-model")
    retriever = Retriever(access, corpus.EMBEDDER, corpus.index(), w.clock)
    audit = InMemoryAuditSink()
    generation = [0]
    deps = GatewayDeps(
        clock=w.clock,
        auth=Authenticator(w.issuer.verifier(w.clock), w.directory),
        access=access,
        rag=RagAnswerer(retriever, llm, w.clock),
        admission=Admission(
            w.clock, rate, burst, queue_max, max_wait_s=3.0, concurrency=8
        ),
        dispatcher=FairDispatcher(capacity=8, quantum=2000.0),
        cache=InMemoryAnswerCache(w.clock),
        audit=audit,
        audit_salt=SALT,
        workers=workers or AlwaysReady(),
        generation=lambda: generation[0],
        debug_retrieval=debug,
    )
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(deps)),
        base_url="http://gateway",
    )
    return Stack(w, deps, client, audit, generation)
