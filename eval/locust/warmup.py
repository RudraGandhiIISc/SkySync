"""Gets a GPU worker running before a measured run starts.

The pool starts a worker only when a request arrives and a card has room;
other users' jobs on the shared cards come and go within minutes. This
asks one question every `every_s` seconds, as a synthetic account, until
the gateway reports a ready worker, so the measurement starts warm.

Run (inside the Slurm job, against the stack in current.json):
    python -m eval.locust.warmup <base url> [--wait-s 5400]
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence

import httpx

from askiisc.common.config import data_dir

HDR = {"Content-Type": "application/json", "X-AskIISc": "1"}
QUESTION = "What are the hostel fees rules?"


def ready(client: httpx.Client) -> bool:
    """Whether the gateway has a ready worker."""
    r = client.get("/healthz")
    return r.status_code == 200 and bool(r.json().get("workers_ready"))  # noqa: PLR2004


def ask(client: httpx.Client) -> int:
    """Asks one question (this is what starts a worker); the status."""
    with client.stream(
        "POST",
        "/api/temporary",
        json={"question": QUESTION, "history": []},
        headers=HDR,
    ) as r:
        for _ in r.iter_bytes():
            pass
        return r.status_code


def warm_up(
    client: httpx.Client,
    user: str,
    password: str,
    wait_s: float,
    every_s: float = 60.0,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Asks until a worker is ready or `wait_s` passes.

    Returns:
        True if a worker became ready.
    """
    login = client.post(
        "/api/auth/login",
        json={"user_id": user, "password": password},
        headers=HDR,
    )
    login.raise_for_status()
    deadline = now() + wait_s
    while now() < deadline:
        if ready(client):
            return True
        ask(client)
        if ready(client):
            return True
        sleep(every_s)
    return False


def main(argv: Sequence[str] | None = None) -> int:
    """Warms the stack at `base`; exit 0 once a worker is ready."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("--wait-s", type=float, default=5400.0)
    args = parser.parse_args(argv)
    data = data_dir()
    accounts = json.loads(
        (data / "private" / "app" / "locust_accounts.json").read_text()
    )
    user = accounts["users"][0]["user_id"]
    with httpx.Client(base_url=args.base, timeout=300.0) as client:
        ok = warm_up(client, user, accounts["password"], args.wait_s)
    print("worker ready" if ok else f"no worker after {args.wait_s:.0f} s")  # noqa: T201
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
