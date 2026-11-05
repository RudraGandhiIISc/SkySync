"""Whether a worker's card was shared at its start (R-3: every run records it).

Without root, NVML may hide other users' processes. Then the card counts as
shared when its used memory exceeds what our own processes account for.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from askiisc.autoscaler.pool import (
    GpuInfo,
    GpuProcess,
    ProcessPool,
    ProcessPoolConfig,
    occupancy,
)
from askiisc.common.clock import FakeClock
from askiisc.common.targets import GPU_SHARED_SLACK_MIB
from askiisc.observability.gpu import descends_from, read_processes

TOTAL = 143_771


def _gpu(used: int, util: int = 0, index: int = 0) -> GpuInfo:
    return GpuInfo(index, util, TOTAL - used, TOTAL)


def test_a_visible_foreign_process_means_shared() -> None:
    procs = [GpuProcess(10, 40_000), GpuProcess(99, 50_000)]
    occ = occupancy(_gpu(90_000), procs, is_ours=lambda pid: pid == 10)
    assert occ.shared and occ.method == "processes"
    assert occ.foreign_processes == 1 and occ.own_mib == 40_000  # noqa: PLR2004


def test_hidden_processes_are_found_by_memory() -> None:
    occ = occupancy(_gpu(56_620), [], is_ours=lambda pid: True)
    assert occ.shared and occ.method == "memory"
    unreadable = occupancy(_gpu(56_620), None, is_ours=lambda pid: True)
    assert unreadable.shared and unreadable.method == "memory"


def test_memory_our_processes_explain_is_not_shared() -> None:
    procs = [GpuProcess(10, 40_000)]
    used = 40_000 + GPU_SHARED_SLACK_MIB - 1
    occ = occupancy(_gpu(used), procs, is_ours=lambda pid: pid == 10)
    assert not occ.shared and occ.method == "memory"
    over = occupancy(_gpu(used + 2), procs, is_ours=lambda pid: pid == 10)
    assert over.shared and over.method == "memory"


def test_an_idle_card_is_not_shared() -> None:
    occ = occupancy(_gpu(500), [], is_ours=lambda pid: False)
    assert not occ.shared and occ.used_mib == 500  # noqa: PLR2004


def test_unknown_process_memory_counts_as_zero() -> None:
    procs = [GpuProcess(10, None)]
    occ = occupancy(_gpu(30_000), procs, is_ours=lambda pid: True)
    assert occ.own_mib == 0 and occ.shared and occ.method == "memory"


class _Process:
    pid = 4242
    returncode: int | None = None

    async def wait(self) -> int:
        self.returncode = 0
        return 0


async def test_pool_records_the_card_state_on_each_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("os.killpg", lambda pid, sig: None)

    async def launcher(cmd: list[str], env: dict[str, str]) -> _Process:
        return _Process()

    ports = iter(range(31_000, 31_010))
    gpus = [_gpu(500, index=0), _gpu(56_620, util=100, index=1)]
    pool = ProcessPool(
        ProcessPoolConfig(vllm_bin="v", model_args=["m"], log_dir="/tmp"),
        FakeClock(),
        httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200))
        ),
        lambda: gpus,
        lambda: next(ports),
        launcher,  # type: ignore[arg-type]
        processes=lambda index: [] if index == 0 else None,
        is_ours=lambda pid: False,
    )
    idle = await pool.request_worker()
    busy = await pool.request_worker()
    assert (idle.gpu, idle.gpu_shared, idle.gpu_free_mib) == (
        0,
        False,
        TOTAL - 500,
    )
    assert idle.gpu_total_mib == TOTAL and idle.gpu_shared_method == "memory"
    assert (busy.gpu, busy.gpu_shared, busy.gpu_utilization) == (1, True, 100)
    await pool.release_worker(idle.worker_id)
    assert [w.worker_id for w in pool.history] == [
        idle.worker_id,
        busy.worker_id,
    ]
    assert pool.history[0].released_at is not None
    await pool.close()


def _nvml(fail: bool = False) -> SimpleNamespace:
    class NVMLError(Exception):
        pass

    def running(handle: int) -> list[SimpleNamespace]:
        if fail:
            raise NVMLError("no permission")
        mib = 1024 * 1024
        return [
            SimpleNamespace(pid=10, usedGpuMemory=40_000 * mib),
            SimpleNamespace(pid=11, usedGpuMemory=None),
        ]

    return SimpleNamespace(
        NVMLError=NVMLError,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetComputeRunningProcesses=running,
    )


def test_read_processes_from_nvml() -> None:
    assert read_processes(_nvml(), 0) == [
        GpuProcess(10, 40_000),
        GpuProcess(11, None),
    ]
    assert read_processes(_nvml(fail=True), 0) is None


def test_descends_from_walks_the_parent_chain() -> None:
    parents = {30: 20, 20: 10, 10: 1, 7: 7, 8: None}
    ours = descends_from(10, parents.get)
    assert ours(10) and ours(20) and ours(30)
    assert not ours(1) and not ours(8) and not ours(99)
    assert not ours(7)  # a parent loop ends the walk


def test_workers_json_keeps_lifecycle_and_card_state(tmp_path: Path) -> None:
    from deploy.stack import write_workers

    from askiisc.autoscaler.pool import Worker, WorkerState

    worker = Worker(
        "vllm-1",
        gpu=1,
        state=WorkerState.READY,
        requested_at=1.0,
        acquired_at=1.5,
        ready_at=30.0,
        gpu_free_mib=87_151,
        gpu_total_mib=TOTAL,
        gpu_utilization=100,
        gpu_shared=True,
        gpu_shared_method="memory",
    )
    path = tmp_path / "workers.json"
    write_workers(path, [worker])
    rows = json.loads(path.read_text())
    assert rows[0]["state"] == "ready" and rows[0]["gpu_shared"] is True
    assert rows[0]["gpu_shared_method"] == "memory"
    assert rows[0]["ready_at"] == 30.0  # noqa: PLR2004
    assert not (tmp_path / "workers.tmp").exists()
