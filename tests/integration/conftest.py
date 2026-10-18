"""Real services for integration tests, on free 127.0.0.1 ports.

Binaries come from $ASKIISC_SCRATCH/bin (slurm/fetch_services.sh). A missing
binary is a failure, not a skip: the gate must not pass silently.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import uvicorn


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_port(port: int, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError(f"port {port} never opened")


@pytest.fixture
def redis_url(tmp_path: Path) -> Iterator[str]:
    binary = shutil.which("redis-server")
    if binary is None:
        pytest.fail("redis-server missing: run sbatch slurm/fetch_services.sh")
    port = free_port()
    proc = subprocess.Popen(  # noqa: S603
        [
            binary,
            "--port",
            str(port),
            "--bind",
            "127.0.0.1",
            "--save",
            "",
            "--appendonly",
            "no",
            "--dir",
            str(tmp_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_port(port)
        yield f"redis://127.0.0.1:{port}/0"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


class ThreadServer:
    """Runs an ASGI app with uvicorn in a thread; `stop` kills it."""

    def __init__(self, app: Any) -> None:
        self.port = free_port()
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> ThreadServer:
        self._thread.start()
        wait_port(self.port)
        return self

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)
