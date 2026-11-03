"""Every Locust workload is well formed and names existing user classes."""

import ast
from pathlib import Path

import pytest
from eval.locust.workloads import WORKLOADS, plan, stage_at

CLASSES = {
    node.name
    for node in ast.walk(
        ast.parse(Path("eval/locust/locustfile.py").read_text())
    )
    if isinstance(node, ast.ClassDef)
}


@pytest.mark.parametrize("name", sorted(WORKLOADS))
def test_workload_plans(name: str) -> None:
    work = plan(name)
    assert work.stages and work.seconds > 0 and work.requirements
    for stage in work.stages:
        assert set(stage.classes) <= CLASSES
        assert stage.seconds > 0 and stage.users >= 0 and stage.spawn_rate > 0
    assert stage_at(work, 0) == work.stages[0]
    assert stage_at(work, work.seconds + 1) is None
    short = plan(name, peak=3, scale=0.1)
    assert short.seconds <= work.seconds
    peak = max(s.users for s in work.stages)
    assert max(s.users for s in short.stages) <= max(3, peak)


def test_workloads_cover_every_requirement() -> None:
    covered = " ".join(r for r, _, _ in WORKLOADS.values())
    for fr in (1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12):
        assert f"FR-{fr}" in covered
    for nfr in range(1, 11):
        assert f"NFR-{nfr}" in covered
    steps = [s.users for s in plan("ramp", peak=16).stages]
    assert steps == [1, 2, 4, 8, 16]
    with pytest.raises(KeyError):
        plan("nope")
