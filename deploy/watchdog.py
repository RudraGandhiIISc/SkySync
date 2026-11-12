"""Restarts the serving job's access and embedding servers when they die.

The vLLM workers are replaced by the autoscaler (`ProcessPool.reap`), but
the access service and the embedding server are single processes the
supervisor starts once. Without this, killing either one (red-team failure
injection) would leave every question denied (FR-4 fails closed) or failed
until someone restarted the whole job (NFR-5). A dead child is started
again with the same command, port and key, after a backoff that doubles
with each quick death (1 s, 2 s, 4 s, ... at most 60 s) and resets once
the child has stayed up for two minutes. A child stopped on purpose (job
shutdown, the admin's access-restart fault) is not in `procs` and is left
alone.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from askiisc.common.clock import Clock

LOG = logging.getLogger("askiisc.watchdog")
CHECK_INTERVAL_S = 2.0
FIRST_BACKOFF_S = 1.0
MAX_BACKOFF_S = 60.0
STABLE_S = 120.0
"""A child up this long counts as healthy again: its backoff resets."""


class Proc(Protocol):
    """What the watchdog reads from a running child."""

    @property
    def returncode(self) -> int | None:
        """None while the process runs."""
        ...


class Supervisor(Protocol):
    """Starts named children and keeps them (deploy/serve.py Children)."""

    @property
    def procs(self) -> Mapping[str, Proc]:
        """The children it is running, by name."""
        ...

    async def start(
        self, name: str, cmd: list[str], log: Path, env: dict[str, str]
    ) -> object:
        """Starts one child."""
        ...


@dataclass(frozen=True)
class ChildSpec:
    """How to start one watched child again."""

    name: str
    cmd: list[str]
    log: Path
    env: dict[str, str] = field(repr=False)


@dataclass
class _State:
    started_at: float = 0.0
    quick_deaths: int = 0
    next_try: float = 0.0


class Watchdog:
    """Restarts dead watched children with exponential backoff.

    Args:
        children: The supervisor holding the running children.
        specs: The children to watch and how to start them.
        clock: The clock.
        on_restart: Called with a child's name after each restart.
    """

    def __init__(
        self,
        children: Supervisor,
        specs: Sequence[ChildSpec],
        clock: Clock,
        on_restart: Callable[[str], None] = lambda name: None,
    ) -> None:
        self._children = children
        self._specs = list(specs)
        self._clock = clock
        self._on_restart = on_restart
        now = clock.now()
        self._state = {s.name: _State(started_at=now) for s in self._specs}

    async def check(self) -> list[str]:
        """Restarts every watched child that has died and may be retried.

        Returns:
            The names restarted in this check.
        """
        restarted = []
        for spec in self._specs:
            if await self._check_one(spec):
                restarted.append(spec.name)
        return restarted

    async def _check_one(self, spec: ChildSpec) -> bool:
        proc = self._children.procs.get(spec.name)
        if proc is None:
            return False  # stopped on purpose
        state, now = self._state[spec.name], self._clock.now()
        if proc.returncode is None:
            if now - state.started_at >= STABLE_S:
                state.quick_deaths = 0
            return False
        if now < state.next_try:
            return False
        LOG.warning("%s exited with %s; restarting", spec.name, proc.returncode)
        delay = min(MAX_BACKOFF_S, FIRST_BACKOFF_S * 2**state.quick_deaths)
        state.quick_deaths += 1
        state.started_at, state.next_try = now, now + delay
        try:
            await self._children.start(spec.name, spec.cmd, spec.log, spec.env)
        except OSError as err:  # retried after the backoff
            LOG.error("could not restart %s: %s", spec.name, err)
            return False
        self._on_restart(spec.name)
        return True

    async def run(self, interval_s: float = CHECK_INTERVAL_S) -> None:
        """Checks every interval until cancelled."""
        while True:
            await self.check()
            await self._clock.sleep(interval_s)
