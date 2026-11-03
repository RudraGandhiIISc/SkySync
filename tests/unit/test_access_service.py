"""FR-4: the service, the HTTP client and failing closed."""

from pathlib import Path

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
        assert r.json()["doc_ids"] == [
            "d01",
            "d02",
            "d05",
            "d09",
            "d12",
            "d14",
            "d21",
            "d22",
        ]
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
        assert result.doc_ids == {"d01", "d03", "d04", "d17", "d21", "d22"}
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


async def test_department_admins_manage_only_their_documents() -> None:
    from askiisc.access.directory import Directory
    from askiisc.access.policy import DocPolicy, PolicyStore

    w = world()
    directory = Directory.from_records(
        [
            {"user_id": "adm_cds", "department": "cds", "role": "admin"},
            {"user_id": "adm_csa", "department": "csa", "role": "admin"},
            {"user_id": "adm_inst", "department": "institute", "role": "admin"},
        ]
    )
    store = PolicyStore(
        [
            DocPolicy(
                doc_id=DocId("cds-doc"),
                owner="cds",
                departments=frozenset({"cds"}),
            )
        ]
    )
    app = create_app(directory, store, w.issuer.verifier(w.clock), w.clock)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://a") as c:

        async def put(
            user: str, doc: str, owner: str, co: tuple[str, ...] = ()
        ) -> int:
            body = {"owner": owner, "co_owners": co, "departments": [owner]}
            r = await c.put(
                f"/v1/policy/{doc}", json=body, headers=_auth(w.token(user))
            )
            return r.status_code

        async def delete(user: str, doc: str) -> int:
            r = await c.delete(
                f"/v1/policy/{doc}", headers=_auth(w.token(user))
            )
            return r.status_code

        assert await put("adm_csa", "cds-doc", "cds") == 403
        assert await put("adm_csa", "cds-doc", "csa") == 403  # no takeover
        assert await put("adm_cds", "cds-doc", "cds") == 200
        assert await put("adm_csa", "csa-new", "csa") == 200
        assert await put("adm_cds", "csa-new", "csa") == 403
        assert await put("adm_inst", "csa-new", "csa") == 200
        assert await delete("adm_cds", "csa-new") == 403
        assert await delete("adm_csa", "csa-new") == 200
        assert await delete("adm_cds", "never-existed") == 403
        assert await delete("adm_inst", "never-existed") == 200
        # Co-owners: CDS shares its document with CSA; CSA then manages it
        # too, but cannot drop CDS from it.
        assert await put("adm_cds", "cds-doc", "cds", ("csa",)) == 200
        assert await put("adm_csa", "cds-doc", "cds", ("csa",)) == 200
        assert await put("adm_csa", "cds-doc", "csa") == 403
        r = await c.get("/v1/policy/cds-doc", headers=_auth(w.token("adm_csa")))
        assert r.status_code == 200 and r.json()["co_owners"] == ["csa"]
        assert await delete("adm_csa", "cds-doc") == 200


async def test_compact_filter_lists_only_the_non_public_documents() -> None:
    w = world()
    async with _app_client(w) as client:
        token = w.token("u_cds_stu")
        full = await client.post("/v1/filter", json={}, headers=_auth(token))
        compact = await client.post(
            "/v1/filter", json={"compact": True}, headers=_auth(token)
        )
        public = await client.get("/v1/public", headers=_auth(token))
        body = compact.json()
        assert body["public"] is True
        union = set(body["doc_ids"]) | set(public.json()["doc_ids"])
        assert union == set(full.json()["doc_ids"])
        assert public.json()["policy_version"] == body["policy_version"]
        stranger = w.issuer.issue("u_not_in_directory", w.clock.now() + 60)
        r = await client.post(
            "/v1/filter", json={"compact": True}, headers=_auth(stranger)
        )
        assert r.json()["public"] is False and r.json()["doc_ids"] == []
        assert (await client.get("/v1/public")).status_code == 401


async def test_client_fetches_the_public_list_once_per_version() -> None:
    w = world()
    paths: list[str] = []
    app = create_app(
        w.directory, w.policies, w.issuer.verifier(w.clock), w.clock
    )

    async def record(request: httpx.Request) -> None:
        paths.append(request.url.path)

    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://access",
        event_hooks={"request": [record]},
    )
    async with http:
        client = HttpAccessClient(http, "http://access", w.clock, cache_ttl_s=0)
        a = await client.filter(w.token("u_cds_stu"))
        b = await client.filter(w.token("u_ece_fac"))
        assert paths.count("/v1/public") == 1
        assert a.doc_ids != b.doc_ids
        w.policies.delete(DocId("d21"))
        c = await client.filter(w.token("u_ece_fac"))
        assert paths.count("/v1/public") == 2
        assert "d21" not in c.doc_ids


async def test_public_list_from_another_version_denies() -> None:
    stale = httpx.Response(200, json={"doc_ids": ["d1"], "policy_version": 1})
    compact = httpx.Response(
        200, json={"public": True, "doc_ids": [], "policy_version": 2}
    )
    client, _ = _client([compact, stale])
    with pytest.raises(AccessDenied, match="policy changed"):
        await client.filter("t")


async def test_users_with_nothing_extra_share_one_set() -> None:
    compact = httpx.Response(
        200, json={"public": True, "doc_ids": [], "policy_version": 1}
    )
    public = httpx.Response(200, json={"doc_ids": ["d1"], "policy_version": 1})
    client, _ = _client([compact, public, compact, compact])
    first = await client.filter("t1")
    second = await client.filter("t2")
    assert first.doc_ids is second.doc_ids == {"d1"}


async def test_metrics_get_policy_and_change_listener() -> None:
    w = world()
    changes: list[tuple[str, object]] = []
    app = create_app(
        w.directory,
        w.policies,
        w.issuer.verifier(w.clock),
        w.clock,
        on_change=lambda kind, what: changes.append((kind, what)),
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://a") as c:
        admin = _auth(w.token("u_admin"))
        await c.post("/v1/authorize", json={"doc_id": "d02"}, headers=admin)
        await c.put("/v1/policy/n1", json={"roles": ["faculty"]}, headers=admin)
        r = await c.get("/v1/policy/n1", headers=admin)
        assert r.json()["roles"] == ["faculty"]
        r = await c.get("/v1/policy/n1", headers=_auth(w.token("u_cds_stu")))
        assert r.status_code == 403
        assert (await c.get("/v1/policy/zz", headers=admin)).status_code == 404
        await c.delete("/v1/policy/n1", headers=admin)
        text = (await c.get("/metrics")).text
    assert 'askiisc_access_seconds_count{call="authorize"} 1.0' in text
    assert 'askiisc_access_policy_changes_total{kind="put"} 1.0' in text
    assert [k for k, _ in changes] == ["put", "delete"]
    assert changes[1][1] == "n1"


def test_overrides_survive_a_restart(tmp_path: Path) -> None:
    from askiisc.access.policy import DocPolicy, PolicyStore
    from askiisc.access.server import OverrideLog, load_policies

    log = OverrideLog(tmp_path / "private" / "overrides.jsonl")
    corpus = Path("eval/datasets/sample/corpus")
    first = load_policies(corpus, log)
    some = sorted(first.snapshot()[1])[0]
    log.record("put", DocPolicy(doc_id=DocId("added"), roles=frozenset({"x"})))
    log.record("delete", DocId(some))
    (tmp_path / "private" / "overrides.jsonl").open("a").write("\n")
    again = load_policies(corpus, log)
    policies = again.snapshot()[1]
    assert "added" in policies and some not in policies
    assert log.replay(PolicyStore()) == 2
    assert OverrideLog(tmp_path / "none.jsonl").replay(PolicyStore()) == 0
