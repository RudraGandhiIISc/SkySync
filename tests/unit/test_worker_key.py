"""The serving job's vLLM workers and embedding server need its key."""

import httpx
import pytest

from askiisc.autoscaler.pool import (
    KEY_MIDDLEWARE,
    GpuInfo,
    NoGpuRoom,
    ProcessPool,
    ProcessPoolConfig,
    Worker,
)
from askiisc.common.clock import FakeClock
from askiisc.common.errors import LLMError
from askiisc.common.servicekey import KEY_ENV, RequireKey, bearer
from askiisc.common.targets import MIN_GPU_FRACTION, MIN_GPU_FRACTION_4BIT
from askiisc.llm.client import LLMClient
from askiisc.llm.fake_server import FakeLLMConfig, create_app
from askiisc.llm.models import select_model
from askiisc.retrieval.embed_server import create_app as create_embed
from askiisc.retrieval.embedder import EmbeddingError, HttpEmbedder
from askiisc.retrieval.rerank import HttpReranker, RerankError

GPUS = [GpuInfo(0, 0, 140_000, 143_771)]


class _Proc:
    pid = 4242
    returncode = None


def _pool(
    api_key: str, gpus: list[GpuInfo] = GPUS
) -> tuple[ProcessPool, list[list[str]], list[dict[str, str]]]:
    commands: list[list[str]] = []
    envs: list[dict[str, str]] = []

    async def launcher(cmd: list[str], env: dict[str, str]) -> _Proc:
        commands.append(cmd)
        envs.append(env)
        return _Proc()

    config = ProcessPoolConfig(
        vllm_bin="/venv/bin/vllm",
        model_args=["Qwen/Qwen3-8B"],
        log_dir="/tmp",
        api_key=api_key,
        source_dir="/repo/src",
    )
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200))
    )
    pool = ProcessPool(
        config, FakeClock(), http, lambda: gpus, lambda: 30_000, launcher
    )  # type: ignore[arg-type]
    return pool, commands, envs


async def test_workers_get_the_key_by_environment_not_command_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    pool, commands, envs = _pool("secret-key")
    await pool.request_worker()
    command, env = commands[0], envs[0]
    # Other users of the server can read every command line (ps).
    assert "secret-key" not in " ".join(command)
    assert "--api-key" not in command
    assert command[-2:] == ["--middleware", KEY_MIDDLEWARE]
    assert env[KEY_ENV] == "secret-key"
    assert env["PYTHONPATH"].split(":") == ["/repo/src", "/elsewhere"]
    assert pool.auth_headers() == {"Authorization": "Bearer secret-key"}
    assert "secret-key" not in repr(pool._config)


async def test_unkeyed_pools_start_vllm_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)
    pool, commands, envs = _pool("")
    await pool.request_worker()
    assert "--middleware" not in commands[0]
    assert KEY_ENV not in envs[0]
    assert pool.auth_headers() == {}


async def test_no_room_on_any_card_is_its_own_error() -> None:
    pool, _, _ = _pool("k", gpus=[GpuInfo(0, 0, 10_000, 143_771)])
    with pytest.raises(NoGpuRoom, match="no GPU"):
        await pool.request_worker()


def test_four_bit_models_start_in_a_smaller_share_of_a_card() -> None:
    assert select_model("qwen3-8b").min_gpu_fraction == MIN_GPU_FRACTION
    assert (
        select_model("qwen3-8b-awq").min_gpu_fraction == MIN_GPU_FRACTION_4BIT
    )
    gptq = select_model("granite-3.3-8b-instruct-gptq")
    assert gptq.min_gpu_fraction == MIN_GPU_FRACTION_4BIT
    # A 4-bit worker fits in about 21 GiB free; an 8B BF16 one does not.
    crowded = [GpuInfo(0, 100, 22_000, 143_771)]
    pool, _, _ = _pool("k", gpus=crowded)
    with pytest.raises(NoGpuRoom):
        pool._pick(Worker("w"))  # the default share: a BF16 8B model
    pool._config.min_memory_fraction = MIN_GPU_FRACTION_4BIT
    assert pool._pick(Worker("w")) == crowded[0]


async def test_llm_client_sends_the_key_the_fake_worker_requires() -> None:
    app = create_app(FakeLLMConfig(model="m"), FakeClock())
    app.add_middleware(RequireKey, key="k1")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport) as http:
        keyed = LLMClient(http, "http://w", model="m", api_key="k1")
        text, _ = await keyed.complete([{"role": "user", "content": "hi"}])
        assert text
        unkeyed = LLMClient(http, "http://w", model="m")
        with pytest.raises(LLMError, match="401"):
            await unkeyed.complete([{"role": "user", "content": "hi"}])
        # vLLM's SageMaker route is closed too.
        assert (await http.post("http://w/invocations")).status_code == 401
        assert (await http.get("http://w/health")).status_code == 200


def _encode(texts: object) -> list[list[float]]:
    return [[1.0, 0.0, 0.0] for _ in texts]  # type: ignore[attr-defined]


async def test_embedding_server_refuses_calls_without_the_key() -> None:
    app = create_embed(_encode, lambda q, ts: [0.5] * len(ts), key="k2")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://e") as c:
        assert (await c.get("/health")).status_code == 200
        body = {"inputs": ["a"]}
        assert (await c.post("/embed", json=body)).status_code == 401
        ok = await c.post("/embed", json=body, headers=bearer("k2"))
        assert ok.status_code == 200
        keyed = HttpEmbedder(c, "http://e", dim=3, api_key="k2")
        assert await keyed.embed(["a"]) == [(1.0, 0.0, 0.0)]
        with pytest.raises(EmbeddingError):
            await HttpEmbedder(c, "http://e", dim=3).embed(["a"])
        reranker = HttpReranker(c, "http://e", api_key="k2")
        assert await reranker.scores("q", ["a", "b"]) == [0.5, 0.5]
        with pytest.raises(RerankError):
            await HttpReranker(c, "http://e").scores("q", ["a"])


def test_embedding_server_never_starts_without_a_key() -> None:
    with pytest.raises(ValueError, match="key"):
        create_embed(_encode, key="")
