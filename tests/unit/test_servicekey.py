"""The per-job service key in front of vLLM and the embedding server."""

import httpx
import pytest
from fastapi import FastAPI

from askiisc.common.servicekey import (
    EMBED_OPEN_PATHS,
    KEY_ENV,
    RequireKey,
    bearer,
    new_key,
)


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> str:
        return "requests 1"

    @app.post("/invocations")
    async def invocations() -> dict[str, str]:
        return {"answer": "generated"}

    @app.post("/v1/chat/completions")
    async def chat() -> dict[str, str]:
        return {"answer": "generated"}

    return app


def _client(app: object) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    return httpx.AsyncClient(transport=transport, base_url="http://w")


async def test_every_generating_path_needs_the_key() -> None:
    app = _app()
    app.add_middleware(RequireKey, key="k1")
    async with _client(app) as c:
        # vLLM's own --api-key leaves /invocations open; this does not.
        for path in ("/invocations", "/v1/chat/completions"):
            refused = await c.post(path)
            assert refused.status_code == 401
            assert refused.json() == {"error": "unauthorized"}
            assert (
                await c.post(path, headers=bearer("wrong"))
            ).status_code == 401
            assert (await c.post(path, headers=bearer("k1"))).status_code == 200
        lowercase = {"authorization": "bearer k1"}
        assert (
            await c.post("/invocations", headers=lowercase)
        ).status_code == 200
        basic = {"Authorization": "Basic k1"}
        assert (await c.post("/invocations", headers=basic)).status_code == 401


async def test_health_and_counters_stay_open_for_workers() -> None:
    app = _app()
    app.add_middleware(RequireKey, key="k1")
    async with _client(app) as c:
        assert (await c.get("/health")).status_code == 200
        assert (await c.get("/metrics")).status_code == 200


async def test_embed_open_paths_leave_only_health_open() -> None:
    app = _app()
    app.add_middleware(RequireKey, key="k1", open_paths=EMBED_OPEN_PATHS)
    async with _client(app) as c:
        assert (await c.get("/health")).status_code == 200
        assert (await c.get("/metrics")).status_code == 401


async def test_key_comes_from_the_environment_as_vllm_loads_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # vLLM's --middleware adds the class with no arguments.
    monkeypatch.setenv(KEY_ENV, "from-env")
    app = _app()
    app.add_middleware(RequireKey)
    async with _client(app) as c:
        assert (await c.post("/invocations")).status_code == 401
        ok = await c.post("/invocations", headers=bearer("from-env"))
        assert ok.status_code == 200


def test_a_service_never_runs_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)
    with pytest.raises(ValueError, match=KEY_ENV):
        RequireKey(_app())
    with pytest.raises(ValueError, match=KEY_ENV):
        RequireKey(_app(), key="")


async def test_lifespan_events_pass_through() -> None:
    seen = []

    async def inner(scope: dict, receive: object, send: object) -> None:  # type: ignore[type-arg]
        seen.append(scope["type"])

    middleware = RequireKey(inner, key="k")  # type: ignore[arg-type]
    await middleware({"type": "lifespan"}, None, None)  # type: ignore[arg-type]
    assert seen == ["lifespan"]


async def test_websocket_without_the_key_is_closed() -> None:
    sent: list[dict[str, object]] = []

    async def inner(scope: object, receive: object, send: object) -> None:
        raise AssertionError("must not be reached")

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    middleware = RequireKey(inner, key="k")  # type: ignore[arg-type]
    scope = {"type": "websocket", "path": "/ws", "headers": []}
    await middleware(scope, None, send)  # type: ignore[arg-type]
    assert sent == [{"type": "websocket.close", "code": 1008}]


def test_keys_are_random_and_the_header_is_empty_without_one() -> None:
    assert new_key() != new_key()
    assert len(new_key()) >= 40
    assert bearer("") == {}
    assert bearer("x") == {"Authorization": "Bearer x"}
