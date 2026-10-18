"""ProcessPool with fake processes, the cost model, tracing and GPU sampling."""

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from askiisc.autoscaler.pool import (
    GpuInfo,
    PoolError,
    ProcessPool,
    ProcessPoolConfig,
    WorkerState,
    choose_gpu,
    memory_fraction,
)
from askiisc.common.clock import FakeClock
from askiisc.observability.cost import (
    WorkerTimeline,
    cost_report,
    gpu_hours,
    per_thousand,
    worker_cost,
)
from askiisc.observability.gpu import GpuSampler, read_gpus
from askiisc.observability.tracing import make_tracer, span

GPUS = [
    GpuInfo(0, 0, 140_000, 143_771),
    GpuInfo(1, 100, 90_000, 143_771),
    GpuInfo(2, 100, 90_000, 143_771),
]


def test_choose_gpu_prefers_idle_then_free_memory() -> None:
    assert choose_gpu(GPUS, set()) == GPUS[0]
    assert choose_gpu(GPUS, {0}) == GPUS[1]
    assert choose_gpu(GPUS, {0, 1, 2}) is None
    assert memory_fraction(GPUS[0]) == 0.9
    assert memory_fraction(GPUS[1]) == 0.6


class FakeProcess:
    _next_pid = 10_000

    def __init__(self, exits: bool = False) -> None:
        FakeProcess._next_pid += 1
        self.pid = FakeProcess._next_pid
        self.returncode: int | None = 1 if exits else None

    async def wait(self) -> int:
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode


def _pool(
    health: list[int], exits: bool = False, gpus: list[GpuInfo] | None = None
) -> tuple[ProcessPool, list[list[str]], list[dict[str, str]]]:
    calls = iter(health)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(next(calls, 200))

    commands: list[list[str]] = []
    envs: list[dict[str, str]] = []

    async def launcher(cmd: list[str], env: dict[str, str]) -> FakeProcess:
        commands.append(cmd)
        envs.append(env)
        return FakeProcess(exits)

    ports = iter(range(30_000, 30_100))
    config = ProcessPoolConfig(
        vllm_bin="/venv/bin/vllm",
        model_args=["Qwen/Qwen3-4B-Instruct-2507"],
        log_dir="/tmp",
        load_timeout_s=10.0,
        poll_s=2.0,
    )
    pool = ProcessPool(
        config,
        FakeClock(),
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        lambda: gpus or GPUS,
        lambda: next(ports),
        launcher,
    )  # type: ignore[arg-type]
    return pool, commands, envs


async def test_process_pool_starts_pinned_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    killed: list[int] = []
    monkeypatch.setattr("os.killpg", lambda pid, sig: killed.append(pid))
    pool, commands, envs = _pool([503, 200, 200])
    first = await pool.request_worker()
    second = await pool.request_worker()
    assert (first.gpu, second.gpu) == (0, 1)
    assert envs[0]["CUDA_VISIBLE_DEVICES"] == "0"
    assert envs[0]["HF_HUB_OFFLINE"] == "1"
    assert commands[0][:3] == [
        "/venv/bin/vllm",
        "serve",
        "Qwen/Qwen3-4B-Instruct-2507",
    ]
    assert "--host" in commands[0] and "127.0.0.1" in commands[0]
    assert commands[1][-1] == "0.60"
    assert first.state == WorkerState.READY and first.url.endswith(":30000")
    with pytest.raises(PoolError, match="maximum"):
        await pool.request_worker()
    await pool.close()
    assert len(killed) == 2 and pool.list_workers() == []
    assert first.released_at is not None


async def test_process_pool_failures() -> None:
    pool, _, _ = _pool([], exits=True)
    with pytest.raises(PoolError, match="exited"):
        await pool.request_worker()
    pool, _, _ = _pool([503] * 20)
    with pytest.raises(PoolError, match="not healthy"):
        await pool.request_worker()
    full = [GpuInfo(0, 0, 10_000, 143_771)]
    pool, _, _ = _pool([], gpus=full)
    with pytest.raises(PoolError, match="free memory"):
        await pool.request_worker()
    with pytest.raises(ValueError):
        ProcessPool(
            ProcessPoolConfig("v", [], "/tmp", max_workers=3),
            FakeClock(),
            httpx.AsyncClient(),
            list,
            int,
        )


async def test_release_escalates_to_sigkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals: list[int] = []
    monkeypatch.setattr("os.killpg", lambda pid, sig: signals.append(sig))
    pool, _, _ = _pool([200])
    pool._config.stop_grace_s = 0.01
    worker = await pool.request_worker()
    process = pool._workers[worker.worker_id][1]
    calls = 0

    async def slow_wait() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(3600)  # ignores SIGTERM
        return 0

    process.wait = slow_wait  # type: ignore[method-assign]
    await pool.release_worker(worker.worker_id)
    assert len(signals) == 2 and calls == 2


def test_cost_hand_computed_example() -> None:
    t = WorkerTimeline(
        "w1", requested_at=0, acquired_at=10, ready_at=70, released_at=3670
    )
    cost = worker_cost(t, [(100, 700), (500, 1300), (5000, 6000)])
    assert (cost.acquire_s, cost.load_s, cost.serve_s) == (10, 60, 1200)
    assert cost.idle_s == 3600 - 1200
    assert gpu_hours([cost]) == pytest.approx(3670 / 3600)
    assert per_thousand(1.0, 2000) == 0.5
    assert per_thousand(1.0, 0) == float("inf")
    with pytest.raises(ValueError):
        per_thousand(-1, 1)


def test_idle_only_night_costs_everything_held() -> None:
    night = WorkerTimeline("w", 0, 0, 0, 8 * 3600)
    report = cost_report([night], {}, completed=0, correct=0, tenant_tokens={})
    assert report.gpu_hours == 8.0
    assert report.per_1k_queries == float("inf")
    assert report.tenant_token_share == {}


def test_report_shares_and_failed_start() -> None:
    failed = WorkerTimeline("f", 0, 5, None, 5)
    ok = WorkerTimeline("w", 0, 5, 35, 3600)
    report = cost_report(
        [failed, ok],
        {"w": [(35, 3600)]},
        completed=1000,
        correct=800,
        tenant_tokens={"cds": 3, "ece": 1},
    )
    assert report.per_1k_correct == pytest.approx(report.per_1k_queries * 1.25)
    assert report.tenant_token_share == {"cds": 0.75, "ece": 0.25}
    with pytest.raises(ValueError):
        worker_cost(WorkerTimeline("x", 10, 5, 6, 7), [])


@given(
    acquire=st.floats(0, 100),
    load=st.floats(0, 300),
    held=st.floats(0, 1e5),
    extra_idle=st.floats(1, 1e4),
    busy=st.lists(st.tuples(st.floats(0, 1e5), st.floats(0, 1e4)), max_size=10),
)
def test_cost_never_negative_and_grows_with_idle(
    acquire: float,
    load: float,
    held: float,
    extra_idle: float,
    busy: list[tuple[float, float]],
) -> None:
    ready = acquire + load
    intervals = [(ready + s, ready + s + d) for s, d in busy]
    short = worker_cost(
        WorkerTimeline("w", 0, acquire, ready, ready + held), intervals
    )
    longer = worker_cost(
        WorkerTimeline("w", 0, acquire, ready, ready + held + extra_idle),
        intervals,
    )
    for c in (short, longer):
        assert min(c.acquire_s, c.load_s, c.serve_s, c.idle_s) >= -1e-6
    assert longer.total_s > short.total_s


def test_tracing_refuses_text_attributes() -> None:
    exporter = InMemorySpanExporter()
    tracer = make_tracer(exporter)
    with span(tracer, "ask", tenant="cds", top_k=8):
        pass
    assert exporter.get_finished_spans()[0].attributes == {
        "tenant": "cds",
        "top_k": 8,
    }
    with pytest.raises(ValueError), span(tracer, "ask", query_text="secret"):
        pass
    with span(make_tracer(), "noop"):
        pass


def _fake_nvml() -> SimpleNamespace:
    mib = 1024 * 1024
    devices = [(0, 140_000, 143_771), (55, 90_000, 143_771)]
    return SimpleNamespace(
        nvmlDeviceGetCount=lambda: len(devices),
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetUtilizationRates=lambda h: SimpleNamespace(
            gpu=devices[h][0]
        ),
        nvmlDeviceGetMemoryInfo=lambda h: SimpleNamespace(
            free=devices[h][1] * mib, total=devices[h][2] * mib
        ),
    )


async def test_gpu_sampler_reads_only_pinned_cards() -> None:
    nvml = _fake_nvml()
    assert read_gpus(nvml)[1] == GpuInfo(1, 55, 90_000, 143_771)
    clock = FakeClock()
    sampler = GpuSampler(lambda: read_gpus(nvml), clock, lambda: [1])
    count = 0

    def until() -> bool:
        nonlocal count
        count += 1
        return count > 3

    await sampler.run(interval_s=1.0, until=until)
    assert len(sampler.samples) == 3
    assert {s.index for s in sampler.samples} == {1}
    assert sampler.samples[0].used_mib == 143_771 - 90_000
    assert clock.now() == 3.0
