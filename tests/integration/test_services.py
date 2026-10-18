"""Real Redis and a real HTTP access service that is killed mid-run (FR-4)."""

import subprocess

import httpx
import pytest

from askiisc.access.client import HttpAccessClient
from askiisc.access.service import create_app
from askiisc.common.errors import AccessDenied
from askiisc.common.types import Answer, DocId
from askiisc.gateway.cache import CachedAnswer, RedisAnswerCache
from tests.access_fixtures import world
from tests.integration.conftest import ThreadServer


async def test_redis_cache_against_real_server(redis_url: str) -> None:
    from redis import asyncio as aioredis

    client = aioredis.from_url(redis_url)
    cache = RedisAnswerCache(client, ttl_s=60)
    value = CachedAnswer(
        answer=Answer(text="x"),
        documents=(("d01", 1),),
        chunk_ids=("d01@v1#0",),
    )
    await cache.set("k", value)
    assert await cache.get("k") == value
    assert 0 < await client.ttl("k") <= 60
    await client.aclose()


async def test_redis_down_is_a_cache_miss(redis_url: str) -> None:
    from redis import asyncio as aioredis

    client = aioredis.from_url(redis_url)
    await client.shutdown(nosave=True)
    cache = RedisAnswerCache(client)
    assert await cache.get("k") is None
    await client.aclose()


async def test_killing_the_access_service_denies_everything() -> None:
    w = world()
    app = create_app(
        w.directory, w.policies, w.issuer.verifier(w.clock), w.clock
    )
    server = ThreadServer(app).start()
    token = w.token("u_cds_stu")
    async with httpx.AsyncClient() as http:
        client = HttpAccessClient(
            http, server.url, w.clock, cache_ttl_s=0.0, timeout_s=2.0
        )
        assert "d02" in (await client.filter(token)).doc_ids
        assert await client.authorize(token, DocId("d02"))
        server.stop()
        for _ in range(20):
            with pytest.raises(AccessDenied):
                await client.filter(token)
            with pytest.raises(AccessDenied):
                await client.authorize(token, DocId("d01"))


def test_redis_binary_version() -> None:
    import shutil

    binary = shutil.which("redis-server")
    assert binary is not None
    out = subprocess.run(  # noqa: S603
        [binary, "--version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "v=7.2" in out
