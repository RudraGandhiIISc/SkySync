"""Per-request logs: the request id travels to every service and is logged;
requests.log lines; the Loki auth proxy; Loki and Alloy configuration."""

import base64
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import httpx
import pytest
from deploy import logs

from askiisc.access.client import HttpAccessClient
from askiisc.access.service import create_app as create_access
from askiisc.common.requestid import HEADER, REQUEST_ID, headers
from askiisc.common.servicekey import bearer
from askiisc.gateway.app import RequestEvent
from askiisc.llm.client import LLMClient, StreamState
from askiisc.observability.authproxy import check_basic, create_proxy
from askiisc.retrieval.embed_server import create_app as create_embed
from askiisc.retrieval.embedder import HttpEmbedder
from askiisc.web.requestlog import RequestLog
from tests.access_fixtures import world
from tests.gateway_fixtures import build, final

SECRET = "Is the director's salary 12345?"


def _recorder() -> tuple[list[httpx.Request], Any]:
    seen: list[httpx.Request] = []

    async def hook(request: httpx.Request) -> None:
        seen.append(request)

    return seen, hook


def test_headers_follow_the_context() -> None:
    token = REQUEST_ID.set(None)
    assert headers() == {}
    REQUEST_ID.set("abc")
    assert headers() == {HEADER: "abc"}
    REQUEST_ID.reset(token)


async def test_gateway_request_id_reaches_the_llm_and_the_answer() -> None:
    stack = build()
    seen, hook = _recorder()
    stack.deps.rag._llm._client.event_hooks["request"].append(hook)  # type: ignore[attr-defined]  # noqa: SLF001
    events: list[RequestEvent] = []
    stack.deps.on_request = events.append
    answer = final(await stack.ask("u_cds_stu", "hostel fees rules?"))
    rid = str(answer["request_id"])
    assert seen and seen[0].headers[HEADER] == rid
    assert events[0].request_id == rid
    assert stack.audit.records[-1].request_id == rid
    second = final(await stack.ask("u_cds_stu", "library hours?"))
    assert second["request_id"] != rid


async def test_clients_send_the_request_id() -> None:
    REQUEST_ID.set("rid-1")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/embed":
            return httpx.Response(200, json=[[0.0] * 4])
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, text="data: [DONE]\n\n")
        return httpx.Response(
            200, json={"public": False, "doc_ids": [], "policy_version": 1}
        )

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await HttpAccessClient(http, "http://access", world().clock).filter("t")
    await HttpEmbedder(http, "http://embed", dim=4).embed(["x"])
    llm = LLMClient(http, "http://llm", "m")
    async for _ in llm.stream(
        [{"role": "user", "content": "q"}], StreamState()
    ):
        pass
    assert [r.headers.get(HEADER) for r in seen] == ["rid-1"] * 3


async def test_access_and_embed_services_log_one_line_per_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    w = world()
    access = create_access(
        w.directory, w.policies, w.issuer.verifier(w.clock), w.clock
    )
    embed = create_embed(lambda texts: [[1.0, 0.0] for _ in texts], key="k")
    caplog.set_level(logging.INFO)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=access), base_url="http://a"
    ) as c:
        await c.get("/healthz")
        await c.post(
            "/v1/filter",
            json={},
            headers={
                "Authorization": f"Bearer {w.token('u_cds_stu')}",
                HEADER: "rid-2",
            },
        )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=embed), base_url="http://e"
    ) as c:
        await c.post(
            "/embed",
            json={"inputs": [SECRET]},
            headers={HEADER: "rid-2", **bearer("k")},
        )
    lines = [
        json.loads(r.getMessage())
        for r in caplog.records
        if "service" in r.getMessage()
    ]
    assert [(x["service"], x["path"], x["status"]) for x in lines] == [
        ("access", "/v1/filter", 200),
        ("embed", "/embed", 200),
    ]
    assert all(x["request_id"] == "rid-2" for x in lines)
    assert SECRET not in caplog.text


def _event(**kw: Any) -> RequestEvent:
    base: dict[str, Any] = dict(
        request_id="r1", time=100.0, user_id="u1", tenant="cds", source="chat",
        synthetic=False, decision="answered", status=200, latency_ms=12.0,
        ttft_ms=5.0, prompt_tokens=1, completion_tokens=2, cache_hit=False,
        stale=False, citations=1, documents=("d@v1",), model="m",
        policy_version=1, cold_hold=False,
    )  # fmt: skip
    base.update(kw)
    return RequestEvent(**base)


def test_requests_log_lines(tmp_path: Path) -> None:
    log = RequestLog(
        tmp_path / "r.db",
        role_of=lambda u: "student",
        lines=tmp_path / "requests.log",
    )
    log.record(_event())
    log.record(_event(request_id="r2", user_id="u2"))
    log.flush()
    lines = [
        json.loads(x)
        for x in (tmp_path / "requests.log").read_text().splitlines()
    ]
    assert [(x["request_id"], x["user_id"]) for x in lines] == [
        ("r1", "u1"),
        ("r2", "u2"),
    ]
    assert lines[0]["documents"] == ["d@v1"] and lines[0]["role"] == "student"
    assert set(lines[0]) == set(
        log.rows(None)[0]
    )  # same fields as the admin log


async def test_auth_proxy() -> None:
    seen: list[httpx.Request] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(204, headers={"X-Up": "1"})

    proxy = create_proxy(
        httpx.AsyncClient(
            transport=httpx.MockTransport(upstream), base_url="http://up"
        ),
        "loki",
        "pw",
    )
    good = "Basic " + base64.b64encode(b"loki:pw").decode()
    bad = "Basic " + base64.b64encode(b"loki:nope").decode()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=proxy), base_url="http://p"
    ) as c:
        r = await c.get("/ready")
        assert r.status_code == 401 and "Basic" in r.headers["WWW-Authenticate"]
        assert (
            await c.get("/ready", headers={"Authorization": bad})
        ).status_code == 401
        r = await c.post(
            "/loki/api/v1/push?x=1",
            content=b"{}",
            headers={"Authorization": good},
        )
        assert r.status_code == 204 and r.headers["X-Up"] == "1"
    assert len(seen) == 1 and seen[0].url.path == "/loki/api/v1/push"
    assert (
        seen[0].url.params["x"] == "1"
        and "authorization" not in seen[0].headers
    )
    assert not check_basic("Basic !!!", "loki", "pw")
    assert not check_basic("Bearer x", "loki", "pw")


def test_loki_and_alloy_configs(tmp_path: Path) -> None:
    certs = {n: tmp_path / n for n in ("ca.crt", "tls.crt", "tls.key")}
    config, socket = logs.loki_config(tmp_path, tmp_path / "data", 4321, certs)
    data = json.loads(config.read_text())
    assert data["server"]["http_listen_network"] == "unix"
    assert data["server"]["grpc_listen_address"] == "127.0.0.1"
    tls = data["server"]["grpc_tls_config"]
    assert tls["client_auth_type"] == "RequireAndVerifyClientCert"
    assert socket.name == "loki.sock:0"
    assert oct((tmp_path / "loki").stat().st_mode)[-3:] == "700"
    cmd = logs.loki_command(Path("/bin/loki"), config, certs)
    for client in logs.GRPC_CLIENTS:
        assert f"-{client}.tls-enabled=true" in cmd
    password_file = tmp_path / "pw"
    password_file.write_text("s3cret-value")
    alloy = logs.alloy_config(tmp_path, 9999, password_file, "1461").read_text()
    assert "s3cret-value" not in alloy and str(password_file) in alloy
    for job in logs.LOG_JOBS:
        assert f'"job" = "{job}"' in alloy
    assert "127.0.0.1:9999/loki/api/v1/push" in alloy
    assert "user_id" not in alloy and "request_id" not in alloy  # not labels
    cmd = logs.alloy_command(Path("/bin/alloy"), Path("c"), tmp_path, 7777)
    assert "--server.http.listen-addr=127.0.0.1:7777" in cmd


def test_certificates_are_made_once_and_private(tmp_path: Path) -> None:
    if shutil.which("openssl") is None:
        pytest.skip("openssl not installed")
    folder = tmp_path / "tls"
    certs = logs.make_certs(folder)
    assert all(p.exists() for p in certs.values())
    assert oct(certs["tls.key"].stat().st_mode)[-3:] == "600"
    before = certs["tls.crt"].read_bytes()
    assert logs.make_certs(folder)["tls.crt"].read_bytes() == before


def test_grafana_gets_the_loki_source_and_dashboard(tmp_path: Path) -> None:
    from deploy.serve import Paths, grafana_provisioning

    paths = Paths(tmp_path / "run", tmp_path, tmp_path, tmp_path / "data")
    root = grafana_provisioning(paths, {"prometheus": 1}, "pp", (2, "lp"))
    sources = json.loads((root / "datasources" / "prometheus.yaml").read_text())
    loki = sources["datasources"][1]
    assert loki["uid"] == "askiisc-loki" and loki["url"] == "http://127.0.0.1:2"
    assert loki["secureJsonData"]["basicAuthPassword"] == "lp"
    root = grafana_provisioning(paths, {"prometheus": 1}, "pp")
    sources = json.loads((root / "datasources" / "prometheus.yaml").read_text())
    assert len(sources["datasources"]) == 1
    dash = json.loads(Path("deploy/grafana/requests.json").read_text())
    assert dash["uid"] == "askiisc-requests"
    names = {v["name"] for v in dash["templating"]["list"]}
    assert names == {"request_id", "user_id"}
    assert all(p["datasource"]["uid"] == "askiisc-loki" for p in dash["panels"])


def test_vllm_logs_request_ids_without_prompts() -> None:
    from tests.unit.test_pool_observability import _pool

    pool, _, _ = _pool([200])
    cmd = pool.command(30000, 0.5)
    assert "--enable-log-requests" in cmd
    assert cmd[cmd.index("--max-log-len") + 1] == "0"
