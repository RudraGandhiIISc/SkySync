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


def test_w1_has_the_prd_ramp_steady_and_drain() -> None:
    stages = plan("w1").stages
    assert [(s.seconds, s.users, s.drain) for s in stages] == [
        (60, 60, False),
        (600, 60, False),
        (60, 60, True),
    ]
    assert all(s.classes == ("W1User",) for s in stages)


def test_w2_is_w1_at_one_two_and_three_times() -> None:
    stages = plan("w2").stages
    steady = [s.users for s in stages if s.seconds == 600]  # noqa: PLR2004
    assert steady == [60, 120, 180]
    assert stages[-1].drain and stages[-1].users == 180  # noqa: PLR2004


def test_w3_has_a_20x_peak_and_two_quiet_gaps() -> None:
    users = [s.users for s in plan("w3").stages]
    assert len(users) == 24 and plan("w3").seconds == 3600  # noqa: PLR2004
    assert max(users) / sorted(users)[11] == 20  # noqa: PLR2004
    zero = "".join("0" if n == 0 else "x" for n in users)
    assert [len(g) for g in zero.split("x") if g] == [6, 2]


def _registered() -> set[str]:
    tree = ast.parse(Path("eval/locust/locustfile.py").read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "CLASSES"
        ):
            return {
                n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)
            }
    return set()


@pytest.mark.parametrize("name", sorted(WORKLOADS))
def test_every_workload_class_is_registered_for_the_shape(name: str) -> None:
    # Regression (FR-6 run 2126): W1User was defined but missing from
    # CLASSES, so the shape failed with KeyError and w2 never ran.
    for stage in plan(name).stages:
        assert set(stage.classes) <= _registered()


def test_scripted_revocations_do_not_inherit_the_uncoordinated_cycle() -> None:
    # Regression (run 2143): Locust runs a base class's tasks too, so a
    # RevocationUser base revoked the same documents at once and left five
    # policies deleted when the run ended.
    tree = ast.parse(Path("eval/locust/locustfile.py").read_text())
    cls = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "ScriptedRevocationUser"
    )  # fmt: skip
    assert [b.id for b in cls.bases if isinstance(b, ast.Name)] == ["AskUser"]
    assert any(
        isinstance(f, ast.FunctionDef) and f.name == "on_stop" for f in cls.body
    )
