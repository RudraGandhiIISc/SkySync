"""The gateway API: POST /v1/ask streams a cited answer as Server-Sent Events.

Order: auth -> load limits -> access filter -> cache -> retrieval ->
cold-start hold -> fair queue -> LLM stream -> output filter -> audit.

Events: `warming` (a GPU worker is starting), `token` (answer text),
`answer` (final text and citations; replaces the streamed text),
`error`. Rejections before streaming use HTTP 401, 403, 413, 422 and 429.
Every response carries `X-Request-Id`, the id of its audit record (FR-12).

Access is always decided at the time of the request. `as_of` only chooses
which document versions are searched, and is never later than now: a
client-chosen time must not open a document before its policy window or
reopen it after (FR-2, NFR-10; issues #6 and #12).
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from fastapi import FastAPI, Header, Request, Response
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException

from askiisc.access.client import AccessAPI
from askiisc.access.policy import AccessFilter
from askiisc.common.clock import Clock
from askiisc.common.errors import (
    AccessDenied,
    AccessTimeout,
    AskIIScError,
    AuthError,
    ConfigError,
    GpusFull,
    RateLimited,
)
from askiisc.common.requestid import REQUEST_ID
from askiisc.common.targets import (
    CHAT_HISTORY_CHARS,
    GPUS_FULL_RETRY_AFTER_S,
    LLM_MAX_TOKENS,
    MAX_QUESTION_TOKENS,
)
from askiisc.common.types import Answer, TenantId, UserId
from askiisc.gateway.audit import (
    AuditRecord,
    AuditSink,
    Decision,
    DocVersion,
    LatencyBreakdown,
    RefusalReason,
    salted_hash,
)
from askiisc.gateway.auth import Authenticator, Caller
from askiisc.gateway.cache import (
    AnswerCache,
    CachedAnswer,
    cache_key,
    normalise,
)
from askiisc.gateway.drr import FairDispatcher
from askiisc.gateway.limits import Admission, Ticket
from askiisc.gateway.output_filter import filter_answer
from askiisc.gateway.streams import ManagedStream
from askiisc.llm.rag import Prepared, RagAnswerer
from askiisc.observability.metrics import GatewayMetrics
from askiisc.observability.stages import StageTimer
from askiisc.observability.timing import ask_kind

REQUEST_ID_HEADER = "X-Request-Id"
HTTP_UNAVAILABLE = 503
HTTP_BAD_REQUEST = 400
HTTP_INVALID = 422
HTTP_CLIENT_CLOSED = 499
"""Request-log status of a request whose client left before its answer."""
CALLER_KEY = "askiisc.caller"
"""Scope state: the (Caller, Origin) the session named, set by the web app."""
GPUS_FULL_MESSAGE = (
    "All GPUs on the shared server are in use by other jobs, so no model "
    "worker can start. Please try again in a few minutes."
)


class WorkerStatus(Protocol):
    """What the gateway needs from the autoscaler."""

    def ready(self) -> bool:
        """True if at least one GPU worker can serve now."""
        ...

    async def wait_ready(self, timeout_s: float) -> bool:
        """Waits for a worker; False on timeout or when the GPUs are full."""
        ...

    def gpus_full(self) -> bool:
        """True if no worker can start: other jobs fill every card."""
        ...

    def note_request(self) -> None:
        """Records demand (the autoscaler's arrival signal)."""
        ...


class AlwaysReady:
    """WorkerStatus for a fixed, always-on worker."""

    def ready(self) -> bool:
        """Always true."""
        return True

    async def wait_ready(self, timeout_s: float) -> bool:
        """Returns at once."""
        return True

    def gpus_full(self) -> bool:
        """Never: the worker is always there."""
        return False

    def note_request(self) -> None:
        """Does nothing."""


@dataclass(frozen=True)
class Origin:
    """Where a request came from; set by the server, never the client.

    Attributes:
        source: "api" (POST /v1/ask), "chat" or "temporary" (the web app).
        synthetic: True for the load generator's accounts.
    """

    source: str = "api"
    synthetic: bool = False

    @property
    def label(self) -> str:
        """The metrics label: "locust" or "real"."""
        return "locust" if self.synthetic else "real"


@dataclass(frozen=True)
class RequestEvent:
    """One finished request, for the admin request log.

    It names the user but never holds question or answer text.
    """

    request_id: str
    time: float
    user_id: str
    tenant: str
    source: str
    synthetic: bool
    decision: str
    status: int
    latency_ms: float
    ttft_ms: float | None
    prompt_tokens: int
    completion_tokens: int
    cache_hit: bool
    stale: bool
    citations: int
    documents: tuple[str, ...]
    model: str
    policy_version: int
    cold_hold: bool


@dataclass
class GatewayDeps:
    """Everything the gateway is built from (all injected)."""

    clock: Clock
    auth: Authenticator
    access: AccessAPI
    rag: RagAnswerer
    admission: Admission
    dispatcher: FairDispatcher
    cache: AnswerCache
    audit: AuditSink
    audit_salt: str
    metrics: GatewayMetrics = field(default_factory=GatewayMetrics)
    workers: WorkerStatus = field(default_factory=AlwaysReady)
    generation: Callable[[], int] = lambda: 0
    max_cold_hold_s: float = 150.0
    debug_retrieval: bool = False
    new_id: Callable[[], str] = lambda: uuid.uuid4().hex
    on_request: Callable[[RequestEvent], None] | None = None
    # Seconds from a monotonic clock, for latencies (a fake in tests).
    monotonic: Callable[[], float] = time.perf_counter
    # Ready GPU workers now (the autoscaler's count); None means 1 when
    # `workers.ready()` and 0 otherwise.
    ready_workers: Callable[[], int] | None = None

    def ready_count(self) -> int:
        """Ready GPU workers at this moment."""
        if self.ready_workers is not None:
            return self.ready_workers()
        return 1 if self.workers.ready() else 0


# Request limits on history; the prompt uses less (llm/prompts.py).
TURN_MAX_CHARS = 4 * CHAT_HISTORY_CHARS
HISTORY_MAX_TURNS = 20


class Turn(BaseModel):
    """One earlier message of a chat."""

    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(max_length=TURN_MAX_CHARS)


class AskRequest(BaseModel):
    """POST /v1/ask body."""

    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=8192)
    as_of: float | None = Field(default=None, allow_inf_nan=False)
    history: list[Turn] = Field(
        default_factory=list, max_length=HISTORY_MAX_TURNS
    )


def sse(event: str, payload: object) -> bytes:
    """Formats one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


def _error(status: int, message: str, **headers: str) -> Response:
    request_id = REQUEST_ID.get()
    if request_id:
        headers = {REQUEST_ID_HEADER: request_id, **headers}
    return JSONResponse({"error": message}, status_code=status, headers=headers)


@dataclass
class _Arrival:
    """What is known about a request from its arrival on."""

    started: float
    timer: StageTimer
    active_workers: int


@dataclass
class Claim:
    """An admission taken before the route ran (FR-6 fast shed path).

    `EarlyAdmission` admits the request as soon as the session names the
    tenant and stores the claim on the request; `ask` then uses its ticket
    instead of admitting again. A claim not handed to a stream (`used`) is
    released by the middleware when the route returns.
    """

    ticket: Ticket
    arrival: _Arrival
    used: bool = False


def _arrive(deps: GatewayDeps) -> _Arrival:
    arrival = _Arrival(
        deps.monotonic(), StageTimer(deps.monotonic), deps.ready_count()
    )
    # Every call this request makes (access, embed, vLLM) carries this id.
    REQUEST_ID.set(deps.new_id())
    return arrival


def access_reason(err: AccessDenied) -> RefusalReason:
    """The audit reason for a failed access check."""
    if isinstance(err, AccessTimeout):
        return "access_check_timeout"
    return "access_check_failed"


def record_refusal(
    deps: GatewayDeps,
    header: str | None,
    question: str,
    origin: Origin,
    err: AccessDenied,
    access_s: float,
) -> str:
    """Audits a request the chat app refused before calling `ask` (FR-12).

    The chat app checks access for a chat's history first; when that check
    fails it answers 503 itself (fail closed), and this writes the audit
    record the gateway would have written.

    Args:
        deps: The gateway's dependencies.
        header: The Authorization header the app would have sent.
        question: The question (only its salted hash is stored).
        origin: Where the request came from.
        err: The access failure.
        access_s: Seconds the chat app spent loading the chat and checking
            access before it refused.

    Returns:
        The audit record's request id (for the X-Request-Id header).
    """
    arrival = _arrive(deps)
    arrival.started -= access_s
    arrival.timer.add("access", access_s)
    try:
        caller = deps.auth.authenticate(header)
    except (AuthError, AccessDenied):
        caller = ANONYMOUS
    body = AskRequest.model_construct(question=question, as_of=None, history=[])
    session = _Session(deps, caller, body, None, None, arrival, origin)
    session.audit("denied", 503, access_reason(err))
    return session.request_id


def create_app(deps: GatewayDeps) -> FastAPI:
    """Builds the gateway.

    Raises:
        ConfigError: If no audit salt is configured.
    """
    if not deps.audit_salt:
        raise ConfigError("ASKIISC_AUDIT_SALT must be set")
    app = FastAPI(title="ask-iisc-gateway")

    @app.get("/healthz")
    async def healthz() -> dict[str, object]:
        return {"status": "ok", "workers_ready": deps.workers.ready()}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(deps.metrics.render(), media_type="text/plain")

    @app.post("/v1/ask", response_model=None)
    async def ask(
        body: AskRequest, authorization: str | None = Header(default=None)
    ) -> Response:
        return await _ask(deps, body, authorization, Origin())

    @app.exception_handler(RequestValidationError)
    async def invalid(
        request: Request, exc: RequestValidationError
    ) -> Response:
        response = invalid_response(exc)
        return audit_refused(deps, request, response, _question(exc.body))

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> Response:
        response = await http_exception_handler(request, exc)
        if exc.status_code != HTTP_BAD_REQUEST:  # a body it cannot parse
            return response
        return audit_refused(deps, request, response, "")

    return app


def audit_refused(
    deps: GatewayDeps, request: Request, response: Response, question: str
) -> Response:
    """Audits a question FastAPI refused before its route ran (FR-12).

    FastAPI refuses a body it cannot parse (400) or validate (422: an
    over-long or empty question, an unknown field, bad JSON) before the
    route runs, so such refusals left no audit record (issue #14). Other
    routes' responses pass through unchanged.

    Args:
        deps: The gateway's dependencies.
        request: The refused request.
        response: The refusal.
        question: The question, if the body had one (only its salted
            hash is stored).

    Returns:
        The refusal, with X-Request-Id on question routes.
    """
    kind = ask_kind(request.url.path)
    if kind is None or request.method != "POST":
        return response
    caller, origin = _caller_of(deps, request, kind)
    arrival = _arrive(deps)
    body = AskRequest.model_construct(question=question, as_of=None, history=[])
    session = _Session(deps, caller, body, None, None, arrival, origin)
    session.audit("rejected", response.status_code, "invalid_request")
    response.headers[REQUEST_ID_HEADER] = session.request_id
    return response


def invalid_response(exc: RequestValidationError) -> Response:
    """FastAPI's 422 body without the rejected values, in plain ASCII.

    The default echoes every bad value back: a NaN or infinite number then
    broke the 422 itself (a 500), and a 9,000-character question came
    straight back to the client. ASCII escapes keep a lone surrogate in a
    field name from breaking the encoding.
    """
    detail = [
        {
            "type": e.get("type"),
            "loc": list(e.get("loc", ())),
            "msg": e.get("msg"),
        }
        for e in exc.errors()
    ]
    return Response(
        json.dumps({"detail": detail}, ensure_ascii=True),
        status_code=HTTP_INVALID,
        media_type="application/json",
    )


def _caller_of(
    deps: GatewayDeps, request: Request, kind: str
) -> tuple[Caller, Origin]:
    """Who sent a refused question: the web session, or the API token."""
    named = request.scope.get("state", {}).get(CALLER_KEY)
    if named is not None:
        return named  # type: ignore[no-any-return]
    if kind == "api":
        header = request.headers.get("authorization")
        try:
            return deps.auth.authenticate(header), Origin()
        except (AuthError, AccessDenied):
            return ANONYMOUS, Origin()
    return ANONYMOUS, Origin(kind)


def _question(body: Any) -> str:
    """The question field of a parsed body ("" if there is none)."""
    question = body.get("question") if isinstance(body, dict) else None
    return question if isinstance(question, str) else ""


async def ask(
    deps: GatewayDeps,
    body: AskRequest,
    header: str | None,
    origin: Origin | None = None,
    claim: Claim | None = None,
) -> Response:
    """Answers one question: a JSON rejection or an SSE stream.

    Args:
        deps: The gateway's dependencies.
        body: The request.
        header: The Authorization header value.
        origin: Set by the web app for its own requests; API calls are
            "api" and never synthetic.
        claim: The admission `EarlyAdmission` already took, if any.

    Returns:
        An HTTP response.
    """
    return await _ask(deps, body, header, origin or Origin(), claim)


def shed(
    deps: GatewayDeps,
    caller: Caller,
    body: AskRequest,
    arrival: _Arrival,
    origin: Origin,
    err: RateLimited,
) -> Response:
    """The audited 429 for a request admission refused (FR-6, FR-12)."""
    deps.metrics.shed.labels(caller.tenant).inc()
    deps.metrics.requests.labels(caller.tenant, "rejected").inc()
    _Session(deps, caller, body, None, None, arrival, origin).audit(
        "rejected", 429, "rate_limited"
    )
    retry = str(max(1, math.ceil(err.retry_after_s)))
    return _error(429, "busy, retry later", **{"Retry-After": retry})


def admit_early(
    deps: GatewayDeps, caller: Caller, question: str, origin: Origin
) -> Claim | Response:
    """Admits a request before its route runs (FR-6 fast shed path).

    Args:
        deps: The gateway's dependencies.
        caller: The session's user and tenant (token not needed).
        question: The question (only its salted hash is stored).
        origin: Where the request came from.

    Returns:
        A claim for the route, or the audited 429.
    """
    arrival = _arrive(deps)
    deps.workers.note_request()
    try:
        ticket = deps.admission.admit(caller.tenant)
    except RateLimited as err:
        body = AskRequest.model_construct(
            question=question, as_of=None, history=[]
        )
        return shed(deps, caller, body, arrival, origin, err)
    return Claim(ticket, arrival)


async def _ask(
    deps: GatewayDeps,
    body: AskRequest,
    header: str | None,
    origin: Origin,
    claim: Claim | None = None,
) -> Response:
    arrival = _arrive(deps) if claim is None else claim.arrival
    if body.as_of is not None:  # never a time after now (issue #6)
        now = deps.clock.now()
        body = body.model_copy(update={"as_of": min(body.as_of, now)})

    def reject(
        caller: Caller, decision: Decision, status: int, why: RefusalReason
    ) -> None:
        _Session(deps, caller, body, None, None, arrival, origin).audit(
            decision, status, why
        )

    try:
        with arrival.timer.measure("auth"):
            caller = deps.auth.authenticate(header)
    except AuthError:
        reject(ANONYMOUS, "rejected", 401, "invalid_token")
        return _error(401, "invalid or missing token")
    except AccessDenied:
        reject(ANONYMOUS, "denied", 403, "not_in_directory")
        return _error(403, "not in directory")
    if len(body.question.split()) > MAX_QUESTION_TOKENS:
        reject(caller, "rejected", 413, "question_too_long")
        return _error(413, "question too long")
    if claim is None:
        deps.workers.note_request()
        try:
            ticket = deps.admission.admit(caller.tenant)
        except RateLimited as err:
            return shed(deps, caller, body, arrival, origin, err)
    else:
        ticket = claim.ticket
    try:
        with arrival.timer.measure("access"):
            # Who may read what is decided now, whatever `as_of` says.
            allowed = await deps.access.filter(caller.token)
    except AccessDenied as err:
        if claim is None:  # a claim is released by its middleware
            deps.admission.release(ticket, completed=False)
        reject(caller, "denied", 403, access_reason(err))
        return _error(403, "access check failed")
    session = _Session(deps, caller, body, ticket, allowed, arrival, origin)
    session.handed_off = arrival.timer.now()
    if claim is not None:
        claim.used = True  # the stream releases the ticket
    # The stream frees the ticket and writes the audit record however it
    # ends, even if it never starts (FR-6, FR-12).
    return ManagedStream(
        session.events(),
        media_type="text/event-stream",
        headers={REQUEST_ID_HEADER: session.request_id},
        on_close=(session.abandon,),
    )


ANONYMOUS = Caller(
    user_id=UserId("anonymous"),
    tenant=TenantId("unknown"),
    token="",  # nosec B106 - no token: the caller is unauthenticated
)


class _Session:
    """One /v1/ask request after admission."""

    def __init__(
        self,
        deps: GatewayDeps,
        caller: Caller,
        body: AskRequest,
        ticket: Ticket | None,
        allowed: AccessFilter | None,
        arrival: _Arrival,
        origin: Origin,
    ) -> None:
        self.deps = deps
        self.timer = arrival.timer
        self.active_workers = arrival.active_workers
        self.origin = origin
        self.ttft: float | None = None
        self.cold_hold = False
        self.caller = caller
        self.body = body
        self.ticket = ticket
        self.allowed = allowed
        self.started = arrival.started
        self.request_id = REQUEST_ID.get() or deps.new_id()
        self.handed_off: float | None = None
        self.reason: RefusalReason | None = None
        self.status: int | None = None
        self.prepared: Prepared | None = None
        self.answer: Answer | None = None
        self.cache_hit = False
        self._cached: CachedAnswer | None = None
        # The answer cache's generation when retrieval started (FR-5).
        self._generation: int | None = None
        # When the answer's stale flags were set (FR-5).
        self._flagged_at: float | None = None
        self._closed = False

    @property
    def key(self) -> str | None:
        if (
            self.allowed is None
            or self.body.as_of is not None
            or self.body.history
        ):
            return None
        return cache_key(
            self.allowed.doc_ids,
            self.allowed.policy_version,
            self.deps.generation(),
            self.body.question,
        )

    async def events(self) -> AsyncIterator[bytes]:
        """Runs the request and yields its events."""
        decision: Decision = "error"
        if self.handed_off is not None:
            self.timer.add("start", self.timer.now() - self.handed_off)
        try:
            async for event in self._run():
                yield event
            decision = (
                "abstained"
                if self.answer and self.answer.abstained
                else "answered"
            )
        except AccessDenied as err:
            decision, self.reason = "denied", access_reason(err)
            yield sse("error", {"error": "access check failed"})
        except GpusFull as err:
            self.status, self.reason = HTTP_UNAVAILABLE, "gpus_full"
            yield sse(
                "error",
                {
                    "error": "GpusFull",
                    "message": GPUS_FULL_MESSAGE,
                    "retry_after_s": err.retry_after_s,
                },
            )
        except AskIIScError as err:
            yield sse("error", {"error": type(err).__name__})
        except (asyncio.CancelledError, GeneratorExit):
            self.status, self.reason = HTTP_CLIENT_CLOSED, "client_closed"
            raise
        finally:
            self.close(decision)

    def close(self, decision: Decision = "error") -> None:
        """Frees the admission ticket and writes the audit record, once.

        Args:
            decision: The request's outcome.
        """
        if self._closed:
            return
        self._closed = True
        if self.ticket is not None:
            self.deps.admission.release(
                self.ticket, completed=decision != "error"
            )
        self._finish_metrics(decision)
        self.audit(decision, self.status)

    def abandon(self) -> None:
        """Closes a stream that ends without running to completion.

        Its generator may never have started (the client left first, or
        the route failed after building it), and then its `finally` never
        runs: this frees the ticket and audits the request instead.
        """
        if not self._closed:
            self.status, self.reason = HTTP_CLIENT_CLOSED, "client_closed"
            self.close()

    async def _run(self) -> AsyncIterator[bytes]:
        if await self._from_cache():
            async for event in self._deliver():
                yield event
            return
        rag = self.deps.rag
        self._generation = self.deps.generation()
        self.prepared = await rag.prepare(
            self.caller.token,
            self.body.question,
            self.body.as_of,
            [t.model_dump() for t in self.body.history],
            self.timer,
        )
        if self.prepared.needs_llm:
            async for event in self._hold_for_worker():
                yield event
            async for event in self._generate():
                yield event
        with self.timer.measure("finish"):
            self._flagged_at = self.deps.clock.now()
            answer = rag.finish(self.prepared)
        with self.timer.measure("filter"):
            result = await filter_answer(
                self.deps.access, self.caller.token, answer
            )
        self.answer = result.answer
        if not result.withheld:
            with self.timer.measure("finish"):
                await self._store()
        async for event in self._deliver():
            yield event

    async def _deliver(self) -> AsyncIterator[bytes]:
        # The code after the yield runs when the consumer asks for more:
        # after the chat app has enriched and saved the answer.
        sent = self.timer.now()
        yield self._answer_event()
        self.timer.add("deliver", self.timer.now() - sent)

    async def _from_cache(self) -> bool:
        key = self.key
        cached = None if key is None else await self.deps.cache.get(key)
        if cached is None or self._ended(cached):
            return False
        self.answer, self.cache_hit = cached.answer, True
        self._cached = cached
        self.deps.metrics.cache_hits.inc()
        return True

    async def _store(self) -> None:
        # A revision that became visible while this answer was being made
        # moved the generation: the answer may use the old version, so it
        # must not be cached under the new one (FR-5; run 2234: 171 such
        # answers served unflagged up to 107 s after the old version ended).
        if self._generation != self.deps.generation():
            return
        key = self.key
        if key is None or self.answer is None or self.prepared is None:
            return
        chunks = self.prepared.chunks
        docs = tuple(sorted({(c.doc_id, int(c.version)) for c in chunks}))
        # An edition's end is set in advance and moves no generation, so
        # the entry carries the earliest end of the versions it used and is
        # not served from then on (FR-5, issue #8). Ends from the moment
        # the stale flags were set count: one that passed while the answer
        # was being checked would otherwise slip through unflagged.
        since = self._flagged_at
        if since is None:  # pragma: no cover - finish always comes first
            since = self.deps.clock.now()
        ends = [
            e for c in chunks if (e := c.valid_to) is not None and e > since
        ]
        await self.deps.cache.set(
            key,
            CachedAnswer(
                answer=self.answer,
                documents=docs,
                chunk_ids=tuple(c.id for c in chunks),
                valid_until=min(ends, default=None),
            ),
        )

    def _ended(self, cached: CachedAnswer) -> bool:
        """Whether a version a cached answer used has ended since (FR-5)."""
        end = cached.valid_until
        return end is not None and self.deps.clock.now() >= end

    async def _hold_for_worker(self) -> AsyncIterator[bytes]:
        workers = self.deps.workers
        if workers.ready():
            return
        if workers.gpus_full():
            raise GpusFull("every GPU is full", GPUS_FULL_RETRY_AFTER_S)
        self.deps.metrics.cold_holds.inc()
        self.cold_hold = True
        yield sse(
            "warming",
            {
                "message": "a GPU worker is starting",
                "max_wait_s": self.deps.max_cold_hold_s,
            },
        )
        with self.timer.measure("cold_hold"):
            ready = await workers.wait_ready(self.deps.max_cold_hold_s)
        if not ready and workers.gpus_full():
            raise GpusFull("every GPU is full", GPUS_FULL_RETRY_AFTER_S)
        if not ready:
            raise AskIIScError("cold start took too long")

    async def _generate(self) -> AsyncIterator[bytes]:
        if self.prepared is None:
            raise AskIIScError("generate called before prepare")
        prompt = sum(len(m["content"].split()) for m in self.prepared.messages)
        dispatcher, metrics = self.deps.dispatcher, self.deps.metrics
        metrics.queue_depth.set(dispatcher.waiting + 1)
        with self.timer.measure("queue"):
            await dispatcher.acquire(
                self.caller.tenant, prompt + LLM_MAX_TOKENS
            )
        metrics.queue_depth.set(dispatcher.waiting)
        mark = self.timer.now()
        try:
            async for piece in self.deps.rag.generate(
                self.prepared, self.caller.tenant
            ):
                if self.ttft is None:
                    self._first_token(mark)
                    mark = self.timer.now()
                yield sse("token", {"text": piece})
        finally:
            dispatcher.release()
            stage = "first_token" if self.ttft is None else "generation"
            self.timer.add(stage, self.timer.now() - mark)

    def _first_token(self, dispatched: float) -> None:
        now = self.timer.now()
        self.timer.add("first_token", now - dispatched)
        self.ttft = now - self.started
        metrics, tenant = self.deps.metrics, self.caller.tenant
        metrics.ttft.labels(tenant).observe(self.ttft)
        metrics.origin_ttft.labels(self.origin.label).observe(self.ttft)

    def _answer_event(self) -> bytes:
        answer = self.answer or Answer(text="", abstained=True)
        payload: dict[str, object] = {
            "request_id": self.request_id,
            "text": answer.text,
            "abstained": answer.abstained,
            "citations": [c.model_dump() for c in answer.citations],
            "stale": any(c.is_stale for c in answer.citations),
            # FR-5: true if any citation is superseded ("stale" is kept for
            # older pages).
            "staleness_flag": any(c.is_stale for c in answer.citations),
            "model": self.deps.rag.model,
            "cache_hit": self.cache_hit,
        }
        if self.deps.debug_retrieval:
            payload["retrieved"] = self._retrieved()
        return sse("answer", payload)

    def _retrieved(self) -> list[list[str]]:
        if self.prepared is not None:
            return [[c.id, c.doc_id] for c in self.prepared.chunks]
        if self._cached is not None:
            return [[cid, cid.split("@")[0]] for cid in self._cached.chunk_ids]
        return []

    def _finish_metrics(self, decision: Decision) -> None:
        tenant, metrics = self.caller.tenant, self.deps.metrics
        metrics.requests.labels(tenant, decision).inc()
        seconds = self.deps.monotonic() - self.started
        metrics.latency.labels(tenant).observe(seconds)
        metrics.origin_latency.labels(self.origin.label).observe(seconds)
        if self.prepared is not None:
            usage = self.deps.rag.usage(self.prepared)
            metrics.tokens.labels(tenant).inc(usage.completion_tokens)

    def audit(
        self,
        decision: Decision,
        status: int | None = None,
        reason: RefusalReason | None = None,
    ) -> None:
        """Writes this request's audit record and request event.

        Args:
            decision: The outcome.
            status: The HTTP status, if the request was refused.
            reason: Why it was refused (a fixed code), if it was.
        """
        deps, salt = self.deps, self.deps.audit_salt
        docs: list[DocVersion] = []
        chunk_ids: tuple[str, ...] = ()
        prompt = completion = 0
        if self.prepared is not None:
            chunks = self.prepared.chunks
            docs = sorted(
                {
                    DocVersion(doc_id=c.doc_id, version=int(c.version))
                    for c in chunks
                },
                key=lambda d: (d.doc_id, d.version),
            )
            chunk_ids = tuple(c.id for c in chunks)
            usage = deps.rag.usage(self.prepared)
            prompt, completion = usage.prompt_tokens, usage.completion_tokens
        elif self._cached is not None:
            docs = [
                DocVersion(doc_id=d, version=v)
                for d, v in self._cached.documents
            ]
            chunk_ids = self._cached.chunk_ids
        now = deps.clock.now()
        latency_ms = (deps.monotonic() - self.started) * 1e3
        deps.metrics.by_origin.labels(
            self.origin.label, self.origin.source, decision
        ).inc()
        if deps.on_request is not None:
            deps.on_request(
                RequestEvent(
                    request_id=self.request_id,
                    time=now,
                    user_id=self.caller.user_id,
                    tenant=self.caller.tenant,
                    source=self.origin.source,
                    synthetic=self.origin.synthetic,
                    decision=decision,
                    status=status or (500 if decision == "error" else 200),
                    latency_ms=latency_ms,
                    ttft_ms=None if self.ttft is None else self.ttft * 1e3,
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    cache_hit=self.cache_hit,
                    stale=self._stale(),
                    citations=len(self.answer.citations) if self.answer else 0,
                    documents=tuple(f"{d.doc_id}@v{d.version}" for d in docs),
                    model=deps.rag.model,
                    policy_version=self.allowed.policy_version
                    if self.allowed
                    else 0,
                    cold_hold=self.cold_hold,
                )
            )
        deps.audit.write(
            AuditRecord(
                request_id=self.request_id,
                time=now,
                as_of=now if self.body.as_of is None else self.body.as_of,
                user_hash=salted_hash(salt, self.caller.user_id),
                query_hash=salted_hash(salt, normalise(self.body.question)),
                tenant=self.caller.tenant,
                policy_version=self.allowed.policy_version
                if self.allowed
                else 0,
                documents=tuple(docs),
                chunk_ids=chunk_ids,
                model=deps.rag.model,
                decision=decision,
                prompt_tokens=prompt,
                completion_tokens=completion,
                latency_ms=latency_ms,
                cache_hit=self.cache_hit,
                stale=self._stale(),
                active_workers=self.active_workers,
                cold_start=self.cold_hold,
                latency=LatencyBreakdown.from_stages(
                    self.timer.totals(), latency_ms
                ),
                reason=reason or self.reason,
            )
        )

    def _stale(self) -> bool:
        return bool(
            self.answer and any(c.is_stale for c in self.answer.citations)
        )
