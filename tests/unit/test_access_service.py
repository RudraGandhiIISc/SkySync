"""FR-4: the service, the HTTP client and failing closed."""

import httpx
import pytest

from askiisc.access.client import HttpAccessClient
from askiisc.access.service import create_app
from askiisc.common.errors import AccessDenied, AuthError
from askiisc.common.types import DocId
from tests.access_fixtures import AccessWorld, world


def _app_client(w: AccessWorld) -> httpx.AsyncClient:
    app = create_app(
        w.directory, w.policies, w.issuer.verifier(w.clock), w.clock
    )
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://access"
    )


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_service_endpoints() -> None:
    w = world()
    async with _app_client(w) as client:
        assert (await client.get("/healthz")).json() == {"status": "ok"}
        token = w.token("u_cds_stu")
        r = await client.post(
            "/v1/authorize", json={"doc_id": "d02"}, headers=_auth(token)
        )
        assert r.json()["allowed"] is True
        r = await client.post("/v1/filter", json={}, headers=_auth(token))
        assert r.json()["doc_ids"] == ["d01", "d02", "d05", "d09", "d12", "d14"]
        r = await client.post(
            "/v1/filter", json={"at": 3000.0}, headers=_auth(token)
        )
        assert "d10" in r.json()["doc_ids"]


async def test_user_id_never_comes_from_the_body() -> None:
    w = world()
    async with _app_client(w) as client:
        r = await client.post(
            "/v1/authorize",
            json={"doc_id": "d07", "user_id": "u_admin"},
            headers=_auth(w.token("u_guest")),
        )
    assert r.status_code == 422


@pytest.mark.parametrize("header", ["", "Basic abc", "Bearer not-a-jwt"])
async def test_bad_tokens_get_401(header: str) -> None:
    w = world()
    async with _app_client(w) as client:
        r = await client.post(
            "/v1/filter", json={}, headers={"Authorization": header}
        )
    assert r.status_code == 401


async def test_expired_token_gets_401() -> None:
    w = world()
    token = w.token("u_cds_stu", ttl=10.0)
    w.clock.advance(11.0)
    async with _app_client(w) as client:
        r = await client.post("/v1/filter", json={}, headers=_auth(token))
    assert r.status_code == 401


async def test_policy_changes_need_admin() -> None:
    w = world()
    async with _app_client(w) as client:
        body = {"groups": ["students"]}
        r = await client.put(
            "/v1/policy/new", json=body, headers=_auth(w.token("u_cds_stu"))
        )
        assert r.status_code == 403
        r = await client.put(
            "/v1/policy/new", json=body, headers=_auth(w.token("u_admin"))
        )
        assert r.json() == {"policy_version": 2}
        r = await client.delete(
            "/v1/policy/new", headers=_auth(w.token("u_admin"))
        )
        assert r.json() == {"policy_version": 3}


async def test_http_client_against_service() -> None:
    w = world()
    async with _app_client(w) as http:
        client = HttpAccessClient(http, "http://access", w.clock)
        token = w.token("u_ece_fac")
        assert await client.authorize(token, DocId("d04"))
        assert not await client.authorize(token, DocId("d02"))
        result = await client.filter(token)
        assert result.doc_ids == {"d01", "d03", "d04", "d17"}
        with pytest.raises(AccessDenied):
            await client.filter("garbage")


async def test_revocation_takes_effect_within_sixty_seconds() -> None:
    """Revoke, then probe for 60 s of fake time: no leak after the TTL."""
    w = world()
    async with _app_client(w) as http:
        client = HttpAccessClient(
            http, "http://access", w.clock, cache_ttl_s=10.0
        )
        token = w.token("u_cds_stu")
        assert "d02" in (await client.filter(token)).doc_ids
        w.policies.delete(DocId("d02"))
        revoked_at = w.clock.now()
        leaks = 0
        while w.clock.now() < revoked_at + 60.0:
            w.clock.advance(1.0)
            visible = "d02" in (await client.filter(token)).doc_ids
            if visible and w.clock.now() >= revoked_at + 10.0:
                leaks += 1
            assert not await client.authorize(token, DocId("d02"))
        assert leaks == 0
        assert "d02" not in (await client.filter(token)).doc_ids


class _Transport(httpx.AsyncBaseTransport):
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls = 0

    async def handle_async_request(
        self, request: httpx.Request
    ) -> httpx.Response:
        item = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, httpx.Response)
        return item


def _client(responses: list[object]) -> tuple[HttpAccessClient, _Transport]:
    transport = _Transport(responses)
    http = httpx.AsyncClient(transport=transport)
    return HttpAccessClient(http, "http://access", world().clock), transport


async def test_unreachable_service_denies_after_one_retry() -> None:
    client, transport = _client([httpx.ConnectError("down")])
    with pytest.raises(AccessDenied, match="unreachable"):
        await client.filter("t")
    assert transport.calls == 2


async def test_connection_error_then_success_is_retried() -> None:
    ok = httpx.Response(200, json={"allowed": True, "policy_version": 1})
    client, transport = _client([httpx.ConnectError("blip"), ok])
    assert await client.authorize("t", DocId("d"))
    assert transport.calls == 2


@pytest.mark.parametrize(
    "response",
    [
        httpx.ReadTimeout("slow"),
        httpx.Response(500),
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json=[1, 2]),
        httpx.Response(200, json={"doc_ids": "d01", "policy_version": 1}),
    ],
)
async def test_any_failure_denies(response: object) -> None:
    client, transport = _client([response])
    with pytest.raises(AccessDenied):
        await client.filter("t")
    assert transport.calls == 1


async def test_deny_is_not_retried() -> None:
    deny = httpx.Response(200, json={"allowed": False, "policy_version": 1})
    client, transport = _client([deny])
    assert not await client.authorize("t", DocId("d"))
    assert transport.calls == 1


def test_verifier_rejects_wrong_audience_and_missing_subject() -> None:
    w = world()
    verifier = w.issuer.verifier(w.clock)
    other = w.issuer.issue("u", w.clock.now() + 60, aud="someone-else")
    with pytest.raises(AuthError):
        verifier.verify(other)
    blank = w.issuer.issue("", w.clock.now() + 60)
    with pytest.raises(AuthError):
        verifier.verify(blank)
