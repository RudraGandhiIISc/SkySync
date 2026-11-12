"""The serving job's watchdog for the access and embedding servers."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from deploy.watchdog import (
    MAX_BACKOFF_S,
    STABLE_S,
    ChildSpec,
    Watchdog,
)

from askiisc.common.clock import FakeClock


@dataclass
class FakeProc:
    returncode: int | None = None


@dataclass
class FakeChildren:
    procs: dict[str, FakeProc] = field(default_factory=dict)
    started: list[tuple[str, list[str], dict[str, str]]] = field(
        default_factory=list
    )
    fail: bool = False

    async def start(
        self, name: str, cmd: list[str], log: Path, env: dict[str, str]
    ) -> FakeProc:
        if self.fail:
            raise FileNotFoundError(cmd[0])
        self.started.append((name, cmd, env))
        self.procs[name] = FakeProc()
        return self.procs[name]


def _setup() -> tuple[FakeChildren, Watchdog, FakeClock, list[str]]:
    clock = FakeClock(1000.0)
    children = FakeChildren(procs={"access": FakeProc(), "embed": FakeProc()})
    specs = [
        ChildSpec("access", ["python", "-m", "access"], Path("a.log"), {}),
        ChildSpec(
            "embed", ["python", "-m", "embed"], Path("e.log"), {"K": "key"}
        ),
    ]
    restarts: list[str] = []
    dog = Watchdog(children, specs, clock, on_restart=restarts.append)
    return children, dog, clock, restarts


async def test_a_killed_child_is_restarted_with_its_own_command() -> None:
    children, dog, _, restarts = _setup()
    assert await dog.check() == []
    children.procs["embed"].returncode = -9  # SIGKILL
    assert await dog.check() == ["embed"]
    # Same command and environment, so the same port and key.
    assert children.started == [
        ("embed", ["python", "-m", "embed"], {"K": "key"})
    ]
    assert restarts == ["embed"]
    assert children.procs["embed"].returncode is None


async def test_quick_deaths_back_off_exponentially() -> None:
    children, dog, clock, _ = _setup()
    times = []
    for _ in range(12):
        children.procs["access"].returncode = 1
        while not await dog.check():
            clock.advance(0.5)
        times.append(clock.now())
    gaps = [b - a for a, b in zip(times, times[1:], strict=False)]
    assert gaps[:4] == [1.0, 2.0, 4.0, 8.0]
    assert max(gaps) == MAX_BACKOFF_S


async def test_backoff_resets_after_a_stable_run() -> None:
    children, dog, clock, _ = _setup()
    for _ in range(4):
        children.procs["access"].returncode = 1
        while not await dog.check():
            clock.advance(0.5)
    clock.advance(STABLE_S)
    await dog.check()  # still running: counts as healthy again
    children.procs["access"].returncode = 1
    assert await dog.check() == ["access"]
    children.procs["access"].returncode = 1
    clock.advance(1.0)
    assert await dog.check() == ["access"]  # first backoff again: 1 s


async def test_a_child_stopped_on_purpose_is_left_alone() -> None:
    children, dog, _, _ = _setup()
    del children.procs["access"]  # Children.stop pops it (shutdown, chaos)
    assert await dog.check() == []
    assert children.started == []


async def test_a_failed_restart_is_retried_after_the_backoff() -> None:
    children, dog, clock, restarts = _setup()
    children.procs["embed"].returncode = 1
    children.fail = True
    assert await dog.check() == []
    children.fail = False
    assert await dog.check() == []  # still inside the backoff
    clock.advance(1.0)
    assert await dog.check() == ["embed"]
    assert restarts == ["embed"]


async def test_run_checks_until_cancelled() -> None:
    children, dog, clock, restarts = _setup()
    children.procs["access"].returncode = 1

    class Stop(Exception):
        pass

    calls = 0
    real_sleep = clock.sleep

    async def sleep(seconds: float) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise Stop
        await real_sleep(seconds)

    clock.sleep = sleep  # type: ignore[method-assign]
    with pytest.raises(Stop):
        await dog.run(interval_s=2.0)
    assert restarts == ["access"]
