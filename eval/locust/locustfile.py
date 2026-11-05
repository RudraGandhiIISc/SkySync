"""Locust load tests for Ask IISc.

Real logins and real chats, one workload per requirement (see
workloads.py for the table).

Every simulated user logs in through /api/auth/login with a synthetic
account (`python -m askiisc.web.cli seed-locust`), so the admin dashboard
counts its requests as "Locust" and never as real users.

Run it from the Locust web UI the serving job starts (choose a workload
under "Custom parameters"), or headless with slurm/loadtest.sh. Live
numbers go to Prometheus (and so Grafana); a summary.json per run goes to
$ASKIISC_LOCUST_RESULTS/<run>/ and shows on the admin "Load tests" tab.

Environment:
    ASKIISC_LOCUST_ACCOUNTS  synthetic credentials (private, mode 600)
    ASKIISC_CORPUS_DIR       corpus root (policies.json for leak checks)
    ASKIISC_PROM_TARGETS     Prometheus file-discovery folder
    ASKIISC_LOCUST_RESULTS   where run summaries go
"""

from __future__ import annotations

import hmac
import itertools
import json
import os
import random
import re
import secrets
import socket
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import gevent
from flask import redirect, request, session, url_for
from flask_login import UserMixin, login_user
from locust.clients import HttpSession
from prometheus_client import CollectorRegistry, Gauge, start_http_server

from locust import HttpUser, LoadTestShape, between, constant, events, task

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.locust.client_log import (  # noqa: E402
    ClientLog,
    ClientRecord,
    citations_of,
)
from eval.locust.questions import (  # noqa: E402
    FOLLOW_UPS,
    ZipfQuestions,
    question_ids,
)
from eval.locust.workloads import WORKLOADS, plan, stage_at  # noqa: E402
from eval.oracle.oracle import oracle_allows  # noqa: E402
from eval.redteam import load_cases, planted_documents, verdict  # noqa: E402

HDR = {"Content-Type": "application/json", "X-AskIISc": "1"}
REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("ASKIISC_DATA", "/data4/muthangiv/askiisc"))
# ---------------------------------------------------------------- data


def _accounts() -> dict[str, Any]:
    path = Path(
        os.environ.get(
            "ASKIISC_LOCUST_ACCOUNTS",
            DATA / "private" / "app" / "locust_accounts.json",
        )
    )
    if not path.exists():
        return {"password": "", "users": []}
    return json.loads(path.read_text())


def _questions() -> list[dict[str, Any]]:
    path = REPO / "eval" / "datasets" / "heldout" / "questions.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _policies() -> list[dict[str, Any]]:
    """Every policy: public ones listed, restricted ones from their files.

    policies.json lists the public documents; restricted rules come from
    the .policy.json files, the source the access service loads.
    """
    corpus = Path(os.environ.get("ASKIISC_CORPUS_DIR", DATA / "corpus"))
    path = corpus / "policies.json"
    listed = json.loads(path.read_text()) if path.exists() else []
    out = {p["doc_id"]: p for p in listed}
    for rules in sorted((corpus / "restricted").glob("*/*.policy.json")):
        doc_id = rules.name.removesuffix(".policy.json")
        out[doc_id] = {
            "doc_id": doc_id,
            "owner": rules.parent.name,
            "visibility": "restricted",
            **json.loads(rules.read_text()),
        }
    return list(out.values())


ACCOUNTS = _accounts()
USERS: list[dict[str, Any]] = ACCOUNTS["users"]
PASSWORD: str = ACCOUNTS["password"]
QUESTIONS = _questions()
QUESTION_IDS = question_ids(QUESTIONS)
ANSWERABLE = [q for q in QUESTIONS if q["category"] == "answerable"]
PERMISSION = [q for q in QUESTIONS if q["category"] == "permission"]
POLICIES = _policies()
RESTRICTED = {
    p["doc_id"]: p for p in POLICIES if p.get("visibility") != "public"
}
KNOWN = {p["doc_id"] for p in POLICIES}
EXTRA_POLICIES: dict[str, dict[str, Any]] = {}
"""Policies the load test creates itself (the freshness document)."""
ABUSE_ACCOUNT = USERS[-1]["user_id"] if USERS else ""
_next_account = itertools.count()


def allowed(user_id: str, doc_id: str) -> bool:
    """The independent oracle's decision (public docs skip the slow scan)."""
    policy = EXTRA_POLICIES.get(doc_id) or RESTRICTED.get(doc_id)
    if policy is None:
        return doc_id in KNOWN or not POLICIES
    return oracle_allows(USERS, [policy], user_id, doc_id, time.time())


def pick_account(test: Any = None) -> dict[str, Any]:
    """Round robin over accounts that pass `test` (never the abuse one)."""
    pool = [
        u
        for u in USERS
        if u["user_id"] != ABUSE_ACCOUNT and (test is None or test(u))
    ] or [u for u in USERS if u["user_id"] != ABUSE_ACCOUNT]
    if not pool:
        raise RuntimeError(
            "no synthetic accounts: run askiisc.web.cli seed-locust"
        )
    return pool[next(_next_account) % len(pool)]


# --------------------------------------------------------------- results


class Results:
    """Counters shared by all users of one run, written to summary.json."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Clears everything (a new run started)."""
        self.counts: Counter[str] = Counter()
        self.samples: dict[str, list[float]] = defaultdict(list)
        self.answers: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.client = ClientLog()
        self.started = time.time()

    def count(self, key: str, n: int = 1) -> None:
        """Adds to a named counter."""
        with self.lock:
            self.counts[key] += n

    def sample(self, key: str, value: float) -> None:
        """Records one measurement."""
        with self.lock:
            self.samples[key].append(value)

    def event(self, row: dict[str, Any]) -> None:
        """Records one timestamped event (written to events.jsonl)."""
        with self.lock:
            self.events.append({"time": time.time(), **row})


RESULTS = Results()


# ------------------------------------------------------------ the users


def sse_events(response: Any) -> Any:
    """Yields (event, data) pairs from a streamed response."""
    event = None
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            event = None
            continue
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: ") and event:
            yield event, json.loads(line[6:])


class AskUser(HttpUser):
    """A logged-in user; subclasses decide what to ask."""

    abstract = True
    wait_time = between(5, 15)
    account_filter: Any = None

    def on_start(self) -> None:
        """Logs in with a synthetic account."""
        self.account = pick_account(self.account_filter)
        self.user_id = self.account["user_id"]
        self.chat_id: str | None = None
        self.turns = 0
        login(self.client, self.user_id)

    def on_stop(self) -> None:
        """Logs out."""
        self.client.post("/api/auth/logout", headers=HDR, name="logout")

    def new_chat(self) -> None:
        """Starts a chat."""
        r = self.client.post(
            "/api/chats", json={}, headers=HDR, name="new chat"
        )
        self.chat_id = r.json().get("chat_id") if r.ok else None
        self.turns = 0

    def ask(
        self,
        question: str,
        temporary: bool = False,
        name: str = "ask",
        client: HttpSession | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Sends one question and reads the whole answer stream.

        Records the full answer time as the request time and the time to
        first token as a separate entry. Returns the final answer event.
        """
        http = client or self.client
        if temporary or self.chat_id is None:
            path, body = "/api/temporary", {"question": question, "history": []}
        else:
            path, body = (
                f"/api/chats/{self.chat_id}/messages",
                {"question": question},
            )
        started = time.perf_counter()
        ttft = None

        def log(
            status: int, rid: str | None, answer: Any = None, warm: bool = False
        ) -> None:
            RESULTS.client.add(
                ClientRecord(
                    time=time.time(),
                    request_id=rid,
                    status=status,
                    question_id=QUESTION_IDS.get(question),
                    citations=citations_of(answer),
                    warming=warm,
                    seconds=time.perf_counter() - started,
                )
            )

        with http.post(
            path,
            json=body,
            headers=HDR,
            stream=True,
            catch_response=True,
            name=name,
        ) as r:
            rid = r.headers.get("X-Request-Id")
            if r.status_code != 200:  # noqa: PLR2004
                log(r.status_code, rid)
            if r.status_code == 429:  # noqa: PLR2004
                r.request_meta["name"] = f"{name} [429 busy]"
                r.success()
                RESULTS.count("rejected_429")
                # A well-behaved client waits as told before trying again;
                # without this, no-think users turn 3x load into a storm.
                retry = r.headers.get("Retry-After", "1")
                gevent.sleep(float(retry) if retry.isdigit() else 1.0)
                return None
            if r.status_code != 200:  # noqa: PLR2004
                r.failure(f"HTTP {r.status_code}")
                return None
            ttft, answer, error, warm = read_stream(r, started)
            log(200, (answer or {}).get("request_id") or rid, answer, warm)
            r.request_meta["response_time"] = (
                time.perf_counter() - started
            ) * 1e3
            if error or answer is None:
                r.failure(error or "stream ended without an answer")
                return None
            leaks = check_leaks(user_id or self.user_id, answer)
            if leaks:
                r.failure(f"LEAK: cited {leaks}")
            else:
                r.success()
        if ttft is not None:
            self.environment.events.request.fire(
                request_type="SSE",
                name=f"{name} (first token)",
                response_time=ttft * 1e3,
                response_length=0,
                exception=None,
                context={},
            )
        RESULTS.count("answers")
        RESULTS.count("abstained" if answer["abstained"] else "answered")
        RESULTS.count(
            "cache_hits" if answer.get("cache_hit") else "cache_misses"
        )
        return answer


def read_stream(
    response: Any, started: float
) -> tuple[float | None, dict[str, Any] | None, str | None, bool]:
    """Reads an answer stream: first token time, answer, error, warming."""
    ttft = answer = error = None
    warming = False
    for event, data in sse_events(response):
        if event == "token" and ttft is None:
            ttft = time.perf_counter() - started
        elif event == "warming":
            warming = True
            RESULTS.count("cold_holds")
        elif event == "answer":
            answer = data
        elif event == "error":
            error = data.get("error", "error")
    return ttft, answer, error, warming


def login(
    client: HttpSession, user_id: str, password: str | None = None
) -> bool:
    """Logs a session in; returns whether it worked."""
    r = client.post(
        "/api/auth/login",
        json={"user_id": user_id, "password": password or PASSWORD},
        headers=HDR,
        name="login",
    )
    return bool(r.ok)


def check_leaks(user_id: str, answer: dict[str, Any]) -> list[str]:
    """Cited documents the oracle says the user may not read (FR-2)."""
    docs = {c["doc_id"] for c in answer.get("citations", [])}
    leaks = sorted(d for d in docs if not allowed(user_id, d))
    RESULTS.count("citations_checked", len(docs))
    if leaks:
        RESULTS.count("leaks", len(leaks))
    return leaks


class ChatUser(AskUser):
    """Multi-turn chats with think time (NFR-1 steady load)."""

    weight = 7

    @task
    def converse(self) -> None:
        """A new chat every few turns; follow-ups in between."""
        if self.chat_id is None or self.turns >= random.randint(2, 4):
            self.new_chat()
            question = random.choice(ANSWERABLE)["question"]
        else:
            question = random.choice(FOLLOW_UPS)
        if self.ask(question, name="chat message") is not None:
            self.turns += 1


class TemporaryUser(AskUser):
    """Single questions in temporary chats (nothing stored)."""

    weight = 2

    @task
    def one_question(self) -> None:
        """Asks one held-out question."""
        self.ask(
            random.choice(QUESTIONS)["question"], True, "temporary question"
        )


class BurstUser(AskUser):
    """No think time: drives throughput and overload (NFR-3, FR-6, FR-9)."""

    wait_time = constant(0)

    @task
    def hammer(self) -> None:
        """Asks back to back."""
        self.ask(random.choice(ANSWERABLE)["question"], True, "burst question")


W1_THINK_MEAN_S = 10.0
W1_ZIPF_ALPHA = 1.1
W1_REPEAT = 0.6
W1_TENANTS = ("csa", "cds", "institute")
W1_QUESTIONS = ZipfQuestions(
    QUESTIONS, W1_ZIPF_ALPHA, W1_REPEAT, random.Random(252)
)
"""One generator for the whole run, so repeats are across users."""
DRAINING = threading.Event()
"""Set by the shape during a drain stage: no new questions start."""
_next_tenant = itertools.count()


def exponential(mean_s: float) -> Any:
    """A Locust wait_time with exponential think time of mean `mean_s`."""
    return lambda user: random.expovariate(1 / mean_s)


class W1User(AskUser):
    """PRD W1: 3 tenants, Exp(10 s) think, Zipf(1.1) held-out questions.

    Users are spread evenly over the three departments, and each asks
    single questions (temporary chats). W2 is this user at 1x, 2x and 3x;
    W3 replays a day of it.
    """

    wait_time = exponential(W1_THINK_MEAN_S)

    def on_start(self) -> None:
        """Logs in as an account of the next department in turn."""
        tenant = W1_TENANTS[next(_next_tenant) % len(W1_TENANTS)]
        self.account_filter = lambda u: u["department"] == tenant
        super().on_start()

    @task
    def question(self) -> None:
        """Asks one question, unless the run is draining."""
        if DRAINING.is_set():
            return
        self.ask(str(W1_QUESTIONS.draw()["question"]), True, "w1 question")


class RepeatUser(AskUser):
    """A few popular questions again and again (FR-7 cache, prefix cache)."""

    weight = 1
    wait_time = between(1, 3)
    POPULAR = [q["question"] for q in ANSWERABLE[:10]]

    @task
    def popular(self) -> None:
        """Asks a popular question."""
        self.ask(random.choice(self.POPULAR), True, "popular question")


def _department_of(n: int) -> str:
    counts = Counter(u["department"] for u in USERS)
    ranked = [d for d, _ in counts.most_common()]
    return ranked[min(n, len(ranked) - 1)] if ranked else ""


class HeavyTenantUser(AskUser):
    """The noisy department: 10 users per light one, no think (FR-11)."""

    weight = 10
    wait_time = constant(0)

    def account_filter(self, u: dict[str, Any]) -> bool:  # type: ignore[override]
        """Only the busiest department."""
        return bool(u["department"] == _department_of(0))

    @task
    def hammer(self) -> None:
        """Back-to-back questions."""
        self.ask(random.choice(ANSWERABLE)["question"], True, "heavy tenant")


class LightTenantUser(HeavyTenantUser):
    """The quiet department sharing the GPU (FR-11)."""

    weight = 1

    def account_filter(self, u: dict[str, Any]) -> bool:  # type: ignore[override]
        """Only the second department."""
        return bool(u["department"] == _department_of(1))

    @task
    def hammer(self) -> None:
        """Back-to-back questions."""
        self.ask(random.choice(ANSWERABLE)["question"], True, "light tenant")


class ProbeUser(AskUser):
    """Asks about restricted documents as all kinds of users (FR-2, NFR-10).

    Every answer's citations are checked against the independent oracle;
    a forbidden citation fails the request as a LEAK.
    """

    weight = 2
    wait_time = between(1, 4)

    @task
    def probe(self) -> None:
        """One permission question as this user."""
        item = random.choice(PERMISSION or QUESTIONS)
        answer = self.ask(item["question"], True, "permission probe")
        if answer is None or item.get("doc_id") in (None, "None"):
            return
        cited = item["doc_id"] in {c["doc_id"] for c in answer["citations"]}
        may = allowed(self.user_id, item["doc_id"])
        RESULTS.count("probe_allowed_cited" if may and cited else "probe_other")


class RevocationUser(AskUser):
    """Revokes a document and watches a reader lose it (FR-3: within 60 s).

    As an admin: save a restricted document's policy, check a reader gets
    it cited, delete the policy, then ask as that reader every 3 s for
    90 s. A citation more than 60 s after the revocation is a leak. The
    policy is restored afterwards.
    """

    fixed_count = 1
    wait_time = constant(30)

    def account_filter(self, u: dict[str, Any]) -> bool:  # type: ignore[override]
        """An institute admin (may manage every document)."""
        return bool(u["role"] == "admin" and u["department"] == "institute")

    @task
    def revoke_and_watch(self) -> None:
        """One revocation cycle."""
        cases = [
            (q, u)
            for q in PERMISSION
            for u in USERS
            if q.get("doc_id") in RESTRICTED
            and allowed(u["user_id"], q["doc_id"])
        ]
        if not cases:
            return
        item, reader = random.choice(cases)
        doc = item["doc_id"]
        saved = self.client.get(f"/api/admin/policies/{doc}", name="policy get")
        if not saved.ok:
            return
        session = HttpSession(
            self.host, self.environment.events.request, user=self
        )
        login(session, reader["user_id"])
        before = self.ask(
            item["question"],
            True,
            "revocation reader",
            session,
            reader["user_id"],
        )
        if before is None or doc not in {
            c["doc_id"] for c in before["citations"]
        }:
            RESULTS.count("revocation_skipped")
            return
        policy = {k: v for k, v in saved.json().items() if k != "doc_id"}
        self.client.delete(
            f"/api/admin/policies/{doc}", headers=HDR, name="policy revoke"
        )
        revoked = time.time()
        last_seen = 0.0
        try:
            while time.time() - revoked < 90:  # noqa: PLR2004
                answer = self.ask(
                    item["question"],
                    True,
                    "revocation reader",
                    session,
                    reader["user_id"],
                )
                if answer and doc in {c["doc_id"] for c in answer["citations"]}:
                    last_seen = time.time() - revoked
                gevent.sleep(3)
        finally:
            self.client.put(
                f"/api/admin/policies/{doc}",
                json=policy,
                headers=HDR,
                name="policy restore",
            )
        RESULTS.sample("revocation_last_seen_s", last_seen)
        RESULTS.count("revocations")
        if last_seen > 60:  # noqa: PLR2004
            RESULTS.count("revocation_leaks")


class RevisionUser(AskUser):
    """Publishes a revision every 72 s (50/h) and times freshness (NFR-8).

    Uses its own test document, readable only by the synthetic readers,
    so real users never see it.
    """

    fixed_count = 1
    wait_time = constant(72)
    DOC = "loadtest-freshness"

    def account_filter(self, u: dict[str, Any]) -> bool:  # type: ignore[override]
        """An institute admin."""
        return bool(u["role"] == "admin" and u["department"] == "institute")

    def on_start(self) -> None:
        """Creates the test document's policy."""
        super().on_start()
        self.readers = [u["user_id"] for u in USERS if u["role"] == "student"][
            :3
        ]
        policy = {"owner": "institute", "allow_users": self.readers}
        EXTRA_POLICIES[self.DOC] = {"doc_id": self.DOC, **policy}
        self.client.put(
            f"/api/admin/policies/{self.DOC}",
            json=policy,
            headers=HDR,
            name="policy put",
        )
        self.reader = HttpSession(
            self.host, self.environment.events.request, user=self
        )
        if self.readers:
            login(self.reader, self.readers[0])
        self.revision = 0

    @task
    def revise(self) -> None:
        """Publish, then ask until the answer quotes the new code."""
        if not self.readers:
            return
        self.revision += 1
        code = f"R{self.revision}-{random.randint(1000, 9999)}"
        text = (
            "Load-test freshness notice.\n"
            "The current revision code of the load-test freshness notice "
            f"is {code}.\n"
            f"Published at {time.ctime()}."
        )
        r = self.client.post(
            f"/api/admin/documents/{self.DOC}/revise",
            json={"text": text},
            headers=HDR,
            name="document revise",
        )
        if not r.ok:
            return
        published = time.time()
        question = (
            "What is the current revision code of the load-test "
            "freshness notice?"
        )
        while time.time() - published < 900:  # noqa: PLR2004 - NFR-8: 15 min
            answer = self.ask(
                question, True, "freshness reader", self.reader, self.readers[0]
            )
            if answer and code in answer["text"]:
                RESULTS.sample("freshness_s", time.time() - published)
                RESULTS.count("revisions_current")
                return
            gevent.sleep(5)
        RESULTS.count("revisions_stale")


class ScriptedRevocationUser(RevocationUser):
    """E1: 100 scripted revocations, five at a time on different documents.

    The same cycle as RevocationUser, probing every second for 75 s, until
    REVOCATIONS_TOTAL cycles have run. Each cycle's time from the revoke to
    the last answer that still cited the document goes to samples.json
    (FR-3: at most 60 s).
    """

    fixed_count = 5
    wait_time = constant(2)
    PROBE_S = 1.0
    WATCH_S = 75.0

    @task
    def revoke_and_watch(self) -> None:
        """One revocation cycle on a document no other user is revoking."""
        if RESULTS.counts["revocation_cycles"] >= REVOCATIONS_TOTAL:
            gevent.sleep(5)
            return
        cases = [
            (q, u)
            for q in PERMISSION
            for u in USERS
            if q.get("doc_id") in RESTRICTED
            and q["doc_id"] not in REVOKING
            and allowed(u["user_id"], q["doc_id"])
        ]
        if not cases:
            gevent.sleep(1)
            return
        item, reader = random.choice(cases)
        REVOKING.add(item["doc_id"])
        RESULTS.count("revocation_cycles")
        try:
            self._cycle(item, reader)
        finally:
            REVOKING.discard(item["doc_id"])

    def _cycle(self, item: dict[str, Any], reader: dict[str, Any]) -> None:
        doc = item["doc_id"]
        saved = self.client.get(f"/api/admin/policies/{doc}", name="policy get")
        if not saved.ok:
            return
        session = HttpSession(
            self.host, self.environment.events.request, user=self
        )
        login(session, reader["user_id"])

        def cites() -> bool:
            answer = self.ask(
                item["question"], True, "revocation reader", session,
                reader["user_id"],
            )  # fmt: skip
            return bool(answer) and doc in {
                c["doc_id"] for c in answer["citations"]
            }

        if not cites():
            RESULTS.count("revocation_skipped")
            return
        policy = {k: v for k, v in saved.json().items() if k != "doc_id"}
        self.client.delete(
            f"/api/admin/policies/{doc}", headers=HDR, name="policy revoke"
        )
        revoked, last_seen = time.time(), 0.0
        try:
            while time.time() - revoked < self.WATCH_S:
                if cites():
                    last_seen = time.time() - revoked
                gevent.sleep(self.PROBE_S)
        finally:
            self.client.put(
                f"/api/admin/policies/{doc}", json=policy, headers=HDR,
                name="policy restore",
            )  # fmt: skip
        RESULTS.sample("revocation_last_seen_s", last_seen)
        RESULTS.event(
            {"kind": "revocation", "doc": doc, "last_seen_s": last_seen}
        )
        RESULTS.count("revocations")
        if last_seen > 60:  # noqa: PLR2004 - FR-3
            RESULTS.count("revocation_leaks")


REVOCATIONS_TOTAL = 100
REVOKING: set[str] = set()
"""Documents a ScriptedRevocationUser is revoking now."""
FRESH_DOCS = tuple(f"loadtest-fresh-{i:02d}" for i in range(1, 21))
FRESH_REVISIONS = 60
FRESH_EVERY_S = 20.0
PUBLISHED: dict[str, str] = {}
"""Freshness document to the code of its newest published revision."""


def fresh_question(doc: str) -> str:
    """The question whose answer is a freshness document's code."""
    number = doc.rsplit("-", 1)[1]
    return f"What is the current revision code of freshness notice {number}?"


class FreshnessReplayUser(AskUser):
    """E5 / FR-5: 60 revisions across 20 documents, one every 20 s.

    It does not wait for answers (FreshnessReader asks). Each publish is an
    event (document, version, code), so eval/freshness_check.py can time
    demotion and count unflagged stale answers.
    """

    fixed_count = 1
    wait_time = constant(FRESH_EVERY_S)

    def account_filter(self, u: dict[str, Any]) -> bool:  # type: ignore[override]
        """An institute admin."""
        return bool(u["role"] == "admin" and u["department"] == "institute")

    def on_start(self) -> None:
        """Gives each document a policy readable by the student readers."""
        super().on_start()
        readers = [u["user_id"] for u in USERS if u["role"] == "student"]
        policy = {"owner": "institute", "allow_users": readers}
        for doc in FRESH_DOCS:
            EXTRA_POLICIES[doc] = {"doc_id": doc, **policy}
            self.client.put(
                f"/api/admin/policies/{doc}", json=policy, headers=HDR,
                name="policy put",
            )  # fmt: skip
        self.done = 0

    @task
    def publish(self) -> None:
        """Publishes the next revision, round robin over the documents."""
        if self.done >= FRESH_REVISIONS:
            return
        doc = FRESH_DOCS[self.done % len(FRESH_DOCS)]
        self.done += 1
        code = f"F{self.done}-{random.randint(1000, 9999)}"
        number = doc.rsplit("-", 1)[1]
        text = (
            f"Load-test freshness notice {number}.\n"
            f"The current revision code of freshness notice {number} is "
            f"{code}.\n"
        )
        r = self.client.post(
            f"/api/admin/documents/{doc}/revise", json={"text": text},
            headers=HDR, name="document revise",
        )  # fmt: skip
        if r.ok:
            PUBLISHED[doc] = code
            RESULTS.event(
                {"kind": "publish", "doc": doc, "code": code,
                 "version": r.json().get("version")}
            )  # fmt: skip


class FreshnessReader(AskUser):
    """Asks for a freshness document's code, about every 2 s (FR-5)."""

    wait_time = constant(2)

    def account_filter(self, u: dict[str, Any]) -> bool:  # type: ignore[override]
        """A student (the documents' readers)."""
        return bool(u["role"] == "student")

    @task
    def read(self) -> None:
        """Asks about one published document and logs what came back."""
        if not PUBLISHED:
            return
        doc = random.choice(sorted(PUBLISHED))
        sent = time.time()
        answer = self.ask(fresh_question(doc), True, "freshness reader")
        if answer is None:
            return
        RESULTS.event(
            {
                "kind": "answer",
                "doc": doc,
                "sent": sent,
                "request_id": answer.get("request_id"),
                "codes": quoted_codes(answer["text"]),
                "staleness_flag": answer.get("staleness_flag"),
                "citations": [
                    {k: c.get(k) for k in ("doc_id", "version", "is_stale",
                                           "superseded_by")}
                    for c in answer["citations"]
                ],
            }
        )  # fmt: skip


def quoted_codes(text: str) -> list[str]:
    """Freshness codes quoted in an answer."""
    return re.findall(r"F\d+-\d{4}", text)


class RedTeamUser(AskUser):
    """Asks each of the 90 red-team cases once (NFR-9), then ends the run.

    As an institute admin it first publishes the planted documents
    (eval/datasets/redteam/planted), readable only by the synthetic
    accounts; afterwards their policies are deleted so nobody can read
    them. Each case is asked in a temporary chat as its `as` account.
    Counts: injections and injection_successes (override, extraction,
    indirect, exfiltration), leaks (any category), and per category.
    """

    fixed_count = 1
    wait_time = constant(0)

    def account_filter(self, u: dict[str, Any]) -> bool:  # type: ignore[override]
        """An institute admin (publishes the planted documents)."""
        return bool(u["role"] == "admin" and u["department"] == "institute")

    def on_start(self) -> None:
        """Publishes the planted documents and queues the cases."""
        super().on_start()
        readers = [u["user_id"] for u in USERS if u["role"] != "admin"]
        self.planted = planted_documents()
        for doc, text in self.planted.items():
            policy = {"owner": "institute", "allow_users": readers}
            EXTRA_POLICIES[doc] = {"doc_id": doc, **policy}
            self.client.put(
                f"/api/admin/policies/{doc}",
                json=policy,
                headers=HDR,
                name="policy put",
            )
            self.client.post(
                f"/api/admin/documents/{doc}/revise",
                json={"text": text},
                headers=HDR,
                name="document revise",
            )
        self.queue = load_cases()
        self.sessions: dict[str, HttpSession] = {}

    def on_stop(self) -> None:
        """Makes the planted documents unreadable again."""
        for doc in self.planted:
            self.client.delete(
                f"/api/admin/policies/{doc}", headers=HDR, name="policy revoke"
            )
        super().on_stop()

    def _session(self, user_id: str) -> HttpSession:
        if user_id not in self.sessions:
            session = HttpSession(
                self.host, self.environment.events.request, user=self
            )
            login(session, user_id)
            self.sessions[user_id] = session
        return self.sessions[user_id]

    @task
    def next_case(self) -> None:
        """Asks the next case; quits the run when none are left."""
        if not self.queue:
            if self.environment.runner:
                self.environment.runner.quit()
            return
        case = self.queue.pop(0)
        answer = self.ask(
            case["question"],
            True,
            f"red-team {case['category']}",
            self._session(case["as"]),
            case["as"],
        )
        if answer is None:
            RESULTS.count("redteam_unanswered")
            return
        forbidden = check_leaks_quiet(case["as"], answer)
        result = verdict(case, answer["text"], forbidden)
        category = case["category"]
        RESULTS.count(f"redteam_{category}")
        if category != "cross_user":
            RESULTS.count("injections")
            if result == "success":
                RESULTS.count("injection_successes")
        if result != "blocked":
            RESULTS.count(f"redteam_{category}_{result}")
        if result == "leak" and not forbidden:
            RESULTS.count("leaks")  # a phrase leak; cited ones are counted
        with RESULTS.lock:
            RESULTS.answers.append(
                {"id": case["id"], "category": category, "verdict": result}
            )


def check_leaks_quiet(user_id: str, answer: dict[str, Any]) -> bool:
    """Whether a cited document is forbidden (ask() already counted it)."""
    return any(
        not allowed(user_id, c["doc_id"]) for c in answer.get("citations", [])
    )


class ChaosUser(AskUser):
    """Injects one fault mid-run (serve job needs ASKIISC_CHAOS=1).

    kill_worker for soak (NFR-5), restart_access for failover (FR-4); the
    fault is chosen from the workload, the time by --chaos-at-s.
    """

    fixed_count = 1
    wait_time = constant(5)

    def account_filter(self, u: dict[str, Any]) -> bool:  # type: ignore[override]
        """An institute admin."""
        return bool(u["role"] == "admin" and u["department"] == "institute")

    def on_start(self) -> None:
        """Logs in and remembers when the run started."""
        super().on_start()
        self.done = False

    @task
    def maybe_inject(self) -> None:
        """Injects the fault once its time has come."""
        options = self.environment.parsed_options
        at = (
            getattr(options, "chaos_at_s", 0)
            or plan(
                options.workload, options.peak_users, options.duration_scale
            ).seconds
            / 2
        )
        if self.done or time.time() - RESULTS.started < at:
            return
        fault = (
            "restart_access"
            if options.workload == "failover"
            else "kill_worker"
        )
        r = self.client.post(
            f"/api/admin/chaos/{fault}", headers=HDR, name=f"chaos {fault}"
        )
        RESULTS.count(f"chaos_{fault}_{'ok' if r.ok else 'refused'}")
        self.done = True


class QualityUser(AskUser):
    """Asks every held-out question once as its own user (FR-1, NFR-7).

    Answers go to answers.json in the run folder, in the shape the
    graders read (eval/grader/rubric.py, judge.py).
    """

    wait_time = constant(0)
    queue = list(QUESTIONS)
    in_flight = 0

    @task
    def next_question(self) -> None:
        """Takes the next question; ends the run once all are answered.

        The run ends only when no other user still has a question in
        flight, so the last answers are not lost.
        """
        if not QualityUser.queue:
            if QualityUser.in_flight == 0 and self.environment.runner:
                self.environment.runner.quit()
            gevent.sleep(1)
            return
        item = QualityUser.queue.pop()
        QualityUser.in_flight += 1
        try:
            self._answer(item)
        finally:
            QualityUser.in_flight -= 1

    def _answer(self, item: dict[str, Any]) -> None:
        asker = next((u for u in USERS if u["user_id"] == item["user"]), None)
        if asker is None:
            RESULTS.count("quality_skipped_unknown_user")
            return
        session = HttpSession(
            self.host, self.environment.events.request, user=self
        )
        login(session, asker["user_id"])
        answer = self.ask(
            item["question"],
            True,
            "held-out question",
            session,
            asker["user_id"],
        )
        with RESULTS.lock:
            RESULTS.answers.append(
                {**item, "status": 200 if answer else 0, "answer": answer}
            )


class AuthAbuseUser(HttpUser):
    """Wrong passwords, a lockout and forged sessions (all must fail)."""

    wait_time = between(1, 2)

    @task
    def abuse(self) -> None:
        """One round of bad requests; any success is a failure."""
        for _ in range(6):
            with self.client.post(
                "/api/auth/login",
                json={"user_id": ABUSE_ACCOUNT, "password": "wrong-password-1"},
                headers=HDR,
                catch_response=True,
                name="bad login",
            ) as r:
                if r.status_code in (401, 423):
                    RESULTS.count(f"bad_login_{r.status_code}")
                    r.success()
                else:
                    r.failure(f"bad login got {r.status_code}")
        for path, headers in (("/api/chats", {}), ("/api/admin/overview", {})):
            self.client.cookies.set("askiisc_session", "forged-session-value")
            with self.client.get(
                path,
                headers=headers,
                catch_response=True,
                name="forged session",
            ) as r:
                if r.status_code == 401:  # noqa: PLR2004
                    r.success()
                else:
                    r.failure(f"forged session got {r.status_code}")
        with self.client.post(
            "/api/chats",
            json={},
            catch_response=True,
            name="missing CSRF header",
        ) as r:
            if r.status_code == 403:  # noqa: PLR2004
                r.success()
            else:
                r.failure(f"no CSRF header got {r.status_code}")


# ------------------------------------------------------------- the shape

CLASSES = {
    c.__name__: c
    for c in (
        ChatUser, TemporaryUser, BurstUser, RepeatUser, HeavyTenantUser,
        LightTenantUser, ProbeUser, RevocationUser, RevisionUser, RedTeamUser,
        ChaosUser, QualityUser, AuthAbuseUser, W1User, ScriptedRevocationUser,
        FreshnessReplayUser, FreshnessReader,
    )
}  # fmt: skip


class WorkloadShape(LoadTestShape):
    """Walks through the chosen workload's stages, then stops."""

    def tick(self) -> tuple[int, float, list[type[HttpUser]]] | None:
        """Users, spawn rate and user classes for now."""
        options = self.runner.environment.parsed_options
        work = plan(
            options.workload, options.peak_users, options.duration_scale
        )
        stage = stage_at(work, self.get_run_time())
        if stage is None:
            return None
        if stage.drain:
            DRAINING.set()
        else:
            DRAINING.clear()
        classes = [CLASSES[name] for name in stage.classes]
        fixed = sum(1 for c in classes if getattr(c, "fixed_count", 0))
        users = stage.users + fixed if stage.users else 0
        return users, stage.spawn_rate, classes


@events.init_command_line_parser.add_listener
def _arguments(parser: Any) -> None:
    parser.add_argument(
        "--workload",
        choices=sorted(WORKLOADS),
        default="steady",
        help="which requirement to load-test (see eval/locust/workloads.py)",
    )
    parser.add_argument(
        "--peak-users", type=int, default=0, help="0 = workload default"
    )
    parser.add_argument(
        "--duration-scale", type=float, default=1.0, help="0.1 = 10x shorter"
    )
    parser.add_argument(
        "--chaos-at-s", type=float, default=0, help="0 = half-way"
    )


# --------------------------------------------------- Prometheus exporter


def _exporter(environment: Any) -> None:
    """Serves Locust's live numbers to Prometheus on a free port."""
    registry = CollectorRegistry()
    gauges = {
        name: Gauge(f"locust_{name}", text, registry=registry)
        for name, text in (
            ("users", "Simulated users running."),
            ("requests_per_second", "Current requests per second."),
            ("failures_per_second", "Current failures per second."),
            ("response_p50_ms", "Current median response time."),
            ("response_p95_ms", "Current 95th percentile response time."),
            ("requests_total", "Requests so far in this run."),
            ("failures_total", "Failures so far in this run."),
            ("leaks_total", "Forbidden citations seen (must stay 0)."),
        )
    }
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    start_http_server(port, addr="127.0.0.1", registry=registry)
    folder = Path(
        os.environ.get("ASKIISC_PROM_TARGETS", DATA / "run" / "prom_targets")
    )
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"locust_{os.getpid()}.json"
    target.write_text(
        json.dumps(
            [{"targets": [f"127.0.0.1:{port}"], "labels": {"job": "locust"}}]
        )
    )

    def update() -> None:
        while True:
            stats = environment.stats.total
            runner = environment.runner
            gauges["users"].set(runner.user_count if runner else 0)
            gauges["requests_per_second"].set(stats.current_rps)
            gauges["failures_per_second"].set(stats.current_fail_per_sec)
            gauges["response_p50_ms"].set(
                stats.get_current_response_time_percentile(0.5) or 0
            )
            gauges["response_p95_ms"].set(
                stats.get_current_response_time_percentile(0.95) or 0
            )
            gauges["requests_total"].set(stats.num_requests)
            gauges["failures_total"].set(stats.num_failures)
            gauges["leaks_total"].set(RESULTS.counts["leaks"])
            gevent.sleep(2)

    gevent.spawn(update)
    events.quitting.add_listener(lambda **_: target.unlink(missing_ok=True))


class UIUser(UserMixin):  # type: ignore[misc]
    """The one account of the Locust web UI."""

    def __init__(self, username: str) -> None:
        self.username = username

    def get_id(self) -> str:
        """Flask-Login's session id."""
        return self.username


def _protect_web_ui(web_ui: Any) -> None:
    """Puts a login page in front of the Locust UI (--web-login).

    The UI can start load tests, and 127.0.0.1 on a shared server is
    reachable by every logged-in user, so it needs a password. The user
    name and password file come from deploy/serve.py.
    """
    user = os.environ.get("ASKIISC_LOCUST_UI_USER", "admin")
    path = os.environ.get("ASKIISC_LOCUST_UI_PASSWORD_FILE", "")
    password = Path(path).read_text().strip() if path else ""
    if not password:
        raise RuntimeError("--web-login needs ASKIISC_LOCUST_UI_PASSWORD_FILE")
    web_ui.app.config["SECRET_KEY"] = secrets.token_hex(32)
    web_ui.auth_args = {"username_password_callback": "/login_submit"}
    web_ui.login_manager.user_loader(
        lambda uid: UIUser(uid) if uid == user else None
    )

    @web_ui.app.route("/login_submit", methods=["POST"])
    def login_submit() -> Any:
        name = request.form.get("username", "")
        given = request.form.get("password", "")
        ok_user = hmac.compare_digest(name.encode(), user.encode())
        ok_pass = hmac.compare_digest(given.encode(), password.encode())
        if ok_user and ok_pass:
            login_user(UIUser(user))
            session.pop("auth_error", None)
            return redirect(url_for("locust.index"))
        session["auth_error"] = "Wrong user name or password"
        return redirect(url_for("locust.login"))


@events.init.add_listener
def _on_init(environment: Any, **_: Any) -> None:
    if not USERS:
        print("WARNING: no synthetic accounts; run askiisc.web.cli seed-locust")
    _exporter(environment)
    options = environment.parsed_options
    if environment.web_ui and getattr(options, "web_login", False):
        _protect_web_ui(environment.web_ui)


@events.test_start.add_listener
def _on_start(environment: Any, **_: Any) -> None:
    RESULTS.reset()
    QualityUser.queue = list(QUESTIONS)


def _row(entry: Any) -> dict[str, Any]:
    return {
        "name": entry.name,
        "method": entry.method,
        "requests": entry.num_requests,
        "failures": entry.num_failures,
        "rps": round(entry.total_rps, 3),
        "p50_ms": entry.get_response_time_percentile(0.5),
        "p95_ms": entry.get_response_time_percentile(0.95),
        "p99_ms": entry.get_response_time_percentile(0.99),
        "max_ms": entry.max_response_time,
    }


@events.test_stop.add_listener
def _on_stop(environment: Any, **_: Any) -> None:
    """Writes summary.json (and answers.json for the quality workload)."""
    if environment.stats.total.num_requests == 0:
        return  # nothing ran (for example the UI was closed at shutdown)
    options = environment.parsed_options
    root = Path(
        os.environ.get("ASKIISC_LOCUST_RESULTS", DATA / "results" / "locust")
    )
    name = os.environ.get("ASKIISC_LOCUST_RUN") or time.strftime(
        "%Y%m%d-%H%M%S"
    )
    folder = root / f"{name}-{options.workload}"
    folder.mkdir(parents=True, exist_ok=True)
    total = environment.stats.total
    samples = {
        k: {"n": len(v), "max": max(v), "mean": sum(v) / len(v)}
        for k, v in RESULTS.samples.items()
        if v
    }
    summary = {
        "name": folder.name,
        "workload": options.workload,
        "requirements": WORKLOADS[options.workload][0],
        "peak_users": options.peak_users,
        "duration_scale": options.duration_scale,
        "started": RESULTS.started,
        "finished": time.time(),
        **{k: v for k, v in _row(total).items() if k not in ("name", "method")},
        "checks": dict(RESULTS.counts),
        "measurements": samples,
        "w1_questions": {
            "draws": W1_QUESTIONS.draws,
            "repeat_rate": round(W1_QUESTIONS.repeat_rate(), 4),
        },
        "by_request": [_row(e) for e in environment.stats.entries.values()],
        "errors": [
            {"name": e.name, "error": str(e.error), "count": e.occurrences}
            for e in environment.stats.errors.values()
        ],
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=1))
    if RESULTS.events:
        (folder / "events.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in RESULTS.events)
        )
    raw = {k: v for k, v in RESULTS.samples.items() if v}
    if raw:
        (folder / "samples.json").write_text(json.dumps(raw))
    if RESULTS.answers:
        (folder / "answers.json").write_text(
            json.dumps(RESULTS.answers, indent=1)
        )
    if len(RESULTS.client):
        RESULTS.client.write(folder / "client_requests.jsonl")
    print(f"summary written to {folder}")
