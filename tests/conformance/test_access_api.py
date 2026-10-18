"""Conformance suite for any AccessAPI implementation (FR-4).

A new implementation passes when it is added to IMPLEMENTATIONS and every
test here is green.
"""

from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from askiisc.access.client import AccessAPI, HttpAccessClient, LocalAccess
from askiisc.access.policy import PolicyEngine
from askiisc.access.service import create_app
from askiisc.common.errors import AccessDenied
from askiisc.common.types import DocId
from tests.access_fixtures import MATRIX, AccessWorld, expected_cases, world


async def _local(w: AccessWorld) -> AsyncIterator[AccessAPI]:
    engine = PolicyEngine(w.directory, w.policies)
    yield LocalAccess(engine, w.issuer.verifier(w.clock), w.clock)


async def _http(w: AccessWorld) -> AsyncIterator[AccessAPI]:
    app = create_app(
        w.directory, w.policies, w.issuer.verifier(w.clock), w.clock
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport) as http:
        yield HttpAccessClient(http, "http://access", w.clock, cache_ttl_s=0.0)


IMPLEMENTATIONS: dict[
    str, Callable[[AccessWorld], AsyncIterator[AccessAPI]]
] = {
    "local": _local,
    "http": _http,
}


@pytest.fixture(params=sorted(IMPLEMENTATIONS))
async def api(
    request: pytest.FixtureRequest,
) -> AsyncIterator[tuple[AccessAPI, AccessWorld]]:
    w = world()
    factory = IMPLEMENTATIONS[request.param]
    async for impl in factory(w):
        yield impl, w


async def test_matrix(api: tuple[AccessAPI, AccessWorld]) -> None:
    impl, w = api
    tokens = {u: w.token(u) for u in MATRIX["expected"]}
    for user, doc, allowed in expected_cases():
        assert await impl.authorize(tokens[user], DocId(doc)) is allowed


async def test_filter_matches_matrix(
    api: tuple[AccessAPI, AccessWorld],
) -> None:
    impl, w = api
    for user, bits in MATRIX["expected"].items():
        want = {
            d
            for d, b in zip(MATRIX["documents"], bits, strict=True)
            if b == "1"
        }
        got = await impl.filter(w.token(user))
        assert got.doc_ids == want


async def test_as_of_time(api: tuple[AccessAPI, AccessWorld]) -> None:
    impl, w = api
    token = w.token("u_cds_stu")
    assert not await impl.authorize(token, DocId("d10"))
    assert await impl.authorize(token, DocId("d10"), at=2500.0)
    assert "d11" in (await impl.filter(token, at=100.0)).doc_ids


async def test_revocation_and_version(
    api: tuple[AccessAPI, AccessWorld],
) -> None:
    impl, w = api
    token = w.token("u_cds_stu")
    before = await impl.filter(token)
    w.policies.delete(DocId("d02"))
    after = await impl.filter(token)
    assert "d02" in before.doc_ids
    assert "d02" not in after.doc_ids
    assert after.policy_version > before.policy_version


async def test_bad_token_denies(api: tuple[AccessAPI, AccessWorld]) -> None:
    impl, w = api
    with pytest.raises(AccessDenied):
        await impl.authorize("not-a-token", DocId("d01"))
    expired = w.token("u_admin", ttl=-1.0)
    with pytest.raises(AccessDenied):
        await impl.filter(expired)
