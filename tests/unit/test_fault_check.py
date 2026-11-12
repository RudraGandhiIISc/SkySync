"""The live failure-injection check's verdict logic (eval/fault_check.py)."""

import os
import subprocess
import sys

from eval.fault_check import (
    answered,
    argv_of,
    own_processes,
    port_of,
    refusals_pass,
    verdict,
    wait_for,
)

GOOD = {
    "workers": {
        "http://w:1": {
            "chat_without_key": 401,
            "invocations_without_key": 401,
            "health": 200,
        }
    },
    "embed": {
        "embed_without_key": 401,
        "rerank_without_key": 401,
        "health": 200,
    },
}


def test_port_and_answer_parsing() -> None:
    argv = ["python", "-m", "askiisc.access.server", "--port", "4321"]
    assert port_of(argv) == 4321
    stream = (
        'event: warming\ndata: {}\n\nevent: answer\ndata: {"text": "x"}\n\n'
    )
    assert answered(stream)
    assert not answered('event: error\ndata: {"error": "GpusFull"}\n\n')


def test_refusals_need_401_everywhere_and_open_health() -> None:
    assert refusals_pass(GOOD)
    leaky = {**GOOD, "embed": {**GOOD["embed"], "embed_without_key": 200}}
    assert not refusals_pass(leaky)
    worker = {**GOOD["workers"]["http://w:1"], "invocations_without_key": 200}
    open_route = {"workers": {"http://w:1": worker}, "embed": GOOD["embed"]}
    assert not refusals_pass(open_route)
    assert not refusals_pass({"workers": {}, "embed": GOOD["embed"]})


def test_verdict_needs_every_recovery() -> None:
    faults = {"embed": {"answered_s": 41.0}, "access": {"answered_s": 3.5}}
    report = {"answered_before": True, "refusals": GOOD, "faults": faults}
    assert verdict(report)
    assert not verdict({**report, "answered_before": False})
    failed = {**faults, "access": {"answered_s": None}}
    assert not verdict({**report, "faults": failed})
    assert not verdict({**report, "faults": {}})


def test_wait_for_times_out_and_measures() -> None:
    clock = [0.0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    calls = iter([False, False, True])
    got = wait_for(lambda: next(calls), 10.0, lambda: clock[0], sleep)
    assert got == 1.0
    assert wait_for(lambda: False, 2.0, lambda: clock[0], sleep) is None


def test_own_processes_finds_this_users_process_by_argv() -> None:
    child = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", "import time; time.sleep(30)", "marker-xyz"]
    )
    try:
        found = own_processes(("marker-xyz",))
        assert found == [child.pid]
        assert argv_of(child.pid)[-1] == "marker-xyz"
        assert os.getpid() not in found
    finally:
        child.kill()
        child.wait()
