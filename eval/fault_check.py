"""Failure injection on a live serving stack: worker keys and the watchdog.

Run inside the GPU job, against the stack in current.json (slurm/v3_stack.sh
part `faults`):

    python -m eval.fault_check <serve run dir> <base url> <out dir>

1. A question through the chat app is answered (the keyed path works).
2. Every vLLM worker and the embedding server refuse calls without the
   serving job's key, on /v1, on vLLM's SageMaker route /invocations and on
   /embed and /rerank; health checks stay open.
3. The embedding server, then the access service, is killed (SIGKILL to
   its process group). The watchdog must start it again; the report gives
   the seconds until a new process runs, until it is healthy and until a
   question is answered again (red-team failure-injection tests 9 and 10,
   NFR-5).

Writes `<out dir>/fault_check.json`; exit 0 only if every check passed.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx

from askiisc.common.config import data_dir
from eval.locust.warmup import HDR, QUESTION

EMBED = ("askiisc.retrieval.embed_server",)
ACCESS = ("askiisc.access.server",)
RECOVERY_LIMIT_S = 600.0
WORKERS_WAIT_S = 60.0
POLL_S = 0.5
UNAUTHORIZED = 401
OK = 200


def own_processes(match: Sequence[str]) -> list[int]:
    """This user's processes whose argv contains every string in `match`."""
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            if os.stat(f"/proc/{entry}").st_uid != os.getuid():
                continue
            argv = Path(f"/proc/{entry}/cmdline").read_bytes().split(b"\0")
        except OSError:
            continue  # the process ended while we looked
        words = [a.decode(errors="replace") for a in argv]
        if all(m in words for m in match):
            pids.append(int(entry))
    return sorted(pids)


def port_of(argv: Sequence[str]) -> int:
    """The value after --port in a command line."""
    return int(argv[list(argv).index("--port") + 1])


def argv_of(pid: int) -> list[str]:
    """A process's command line."""
    raw = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    return [a.decode(errors="replace") for a in raw if a]


def answered(events: str) -> bool:
    """Whether an SSE stream from the chat app carried an answer."""
    return any(
        block.startswith("event: answer") for block in events.split("\n\n")
    )


def ask(client: httpx.Client) -> bool:
    """Asks one question through the chat app; True if it was answered."""
    try:
        r = client.post(
            "/api/temporary",
            json={"question": QUESTION, "history": []},
            headers=HDR,
        )
    except httpx.HTTPError:
        return False
    return r.status_code == OK and answered(r.text)


def refusals(http: httpx.Client, workers: Sequence[str], embed: str) -> dict:
    """Status codes of calls made without the key, and of health checks."""
    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    out: dict[str, Any] = {"workers": {}, "embed": {}}
    for url in workers:
        out["workers"][url] = {
            "chat_without_key": http.post(
                url + "/v1/chat/completions", json=body
            ).status_code,
            "invocations_without_key": http.post(
                url + "/invocations", json=body
            ).status_code,
            "health": http.get(url + "/health").status_code,
        }
    out["embed"] = {
        "embed_without_key": http.post(
            embed + "/embed", json={"inputs": ["x"]}
        ).status_code,
        "rerank_without_key": http.post(
            embed + "/rerank", json={"query": "q", "texts": ["x"]}
        ).status_code,
        "health": http.get(embed + "/health").status_code,
    }
    return out


def refusals_pass(found: dict) -> bool:
    """Every unkeyed call got 401 and every health check 200."""
    checks = [*found["workers"].values(), found["embed"]]
    return bool(found["workers"]) and all(
        (v == OK) if k == "health" else (v == UNAUTHORIZED)
        for check in checks
        for k, v in check.items()
    )


def wait_for(
    test: Callable[[], bool],
    limit_s: float,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> float | None:
    """Seconds until `test()` holds, or None after `limit_s`."""
    start = now()
    while now() - start < limit_s:
        if test():
            return round(now() - start, 1)
        sleep(POLL_S)
    return None


def kill_and_recover(
    name: str, match: Sequence[str], health: str, chat: httpx.Client
) -> dict:
    """Kills one child and times the watchdog's recovery."""
    old = own_processes(match)
    if len(old) != 1:
        return {"error": f"expected one {name} process, found {old}"}
    url = f"http://127.0.0.1:{port_of(argv_of(old[0]))}{health}"
    os.killpg(old[0], signal.SIGKILL)
    started = time.monotonic()

    def restarted() -> bool:
        return any(p not in old for p in own_processes(match))

    def healthy() -> bool:
        try:
            return httpx.get(url, timeout=2.0).status_code == OK
        except httpx.HTTPError:
            return False

    report = {
        "killed_pid": old[0],
        "new_process_s": wait_for(restarted, RECOVERY_LIMIT_S),
        "healthy_s": wait_for(healthy, RECOVERY_LIMIT_S),
        "answered_s": wait_for(lambda: ask(chat), RECOVERY_LIMIT_S),
    }
    report["total_s"] = round(time.monotonic() - started, 1)
    return report


def live_workers(run_dir: Path) -> list[str]:
    """URLs of the ready workers in the run's workers.json."""
    try:
        rows = json.loads((run_dir / "workers.json").read_text())
    except (OSError, ValueError):
        return []  # not written yet, or being replaced
    return [r["url"] for r in rows if r.get("state") == "ready" and r["url"]]


def wait_for_workers(
    run_dir: Path,
    limit_s: float = WORKERS_WAIT_S,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> list[str]:
    """The ready workers, waiting for workers.json to list one.

    The file is rewritten every 5 s, so a worker that just became ready may
    not be in it yet.
    """
    start = now()
    while True:
        workers = live_workers(run_dir)
        if workers or now() - start >= limit_s:
            return workers
        sleep(POLL_S)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    """Runs the checks against a live stack; exit 0 if all pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("base")
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    accounts = json.loads(
        (data_dir() / "private" / "app" / "locust_accounts.json").read_text()
    )
    with httpx.Client(base_url=args.base, timeout=300.0) as chat:
        chat.post(
            "/api/auth/login",
            json={
                "user_id": accounts["users"][0]["user_id"],
                "password": accounts["password"],
            },
            headers=HDR,
        ).raise_for_status()
        report: dict[str, Any] = {"answered_before": ask(chat)}
        embed = own_processes(EMBED)
        with httpx.Client(timeout=30.0) as http:
            report["refusals"] = refusals(
                http,
                wait_for_workers(args.run_dir),
                f"http://127.0.0.1:{port_of(argv_of(embed[0]))}",
            )
        report["faults"] = {
            "embed": kill_and_recover("embed", EMBED, "/health", chat),
            "access": kill_and_recover("access", ACCESS, "/healthz", chat),
        }
    report["passed"] = verdict(report)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "fault_check.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))  # noqa: T201
    return 0 if report["passed"] else 1


def verdict(report: dict) -> bool:
    """Answered before, every refusal right, both children recovered."""
    faults = report.get("faults", {}).values()
    return (
        bool(report.get("answered_before"))
        and refusals_pass(report["refusals"])
        and bool(faults)
        and all(f.get("answered_s") is not None for f in faults)
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
