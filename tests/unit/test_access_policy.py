"""FR-3 and NFR-10: the policy engine, directory and oracle agree."""

from pathlib import Path

import pytest
from eval.oracle.oracle import LogEntry, check_run, oracle_allows
from hypothesis import given
from hypothesis import strategies as st

from askiisc.access.directory import Directory, Person, parse_export
from askiisc.access.policy import (
    DocPolicy,
    PolicyEngine,
    PolicyStore,
    evaluate,
)
from askiisc.common.errors import ConfigError
from askiisc.common.targets import FR3_MIN_MATRIX_CASES
from askiisc.common.types import DocId, UserId
from tests.access_fixtures import MATRIX, expected_cases, world

CASES = expected_cases()


def test_matrix_is_big_enough() -> None:
    assert len(CASES) >= FR3_MIN_MATRIX_CASES


@pytest.mark.parametrize(("user", "doc", "allowed"), CASES)
def test_engine_matches_matrix(user: str, doc: str, allowed: bool) -> None:
    w = world()
    engine = PolicyEngine(w.directory, w.policies)
    decision = engine.authorize(UserId(user), DocId(doc), MATRIX["at"])
    assert decision.allowed is allowed, decision.reason
    assert decision.policy_version == 1


@pytest.mark.parametrize(("user", "doc", "allowed"), CASES)
def test_oracle_matches_matrix(user: str, doc: str, allowed: bool) -> None:
    got = oracle_allows(
        MATRIX["directory"], MATRIX["policies"], user, doc, MATRIX["at"]
    )
    assert got is allowed


def test_filter_agrees_with_authorize() -> None:
    w = world()
    engine = PolicyEngine(w.directory, w.policies)
    for user in MATRIX["expected"]:
        docs = engine.filter(UserId(user), MATRIX["at"]).doc_ids
        for doc in MATRIX["documents"]:
            decision = engine.authorize(UserId(user), DocId(doc), MATRIX["at"])
            assert (doc in docs) is decision.allowed


def test_unknown_user_and_document_are_denied() -> None:
    w = world()
    engine = PolicyEngine(w.directory, w.policies)
    assert not engine.authorize(UserId("ghost"), DocId("d01"), 1000.0).allowed
    assert not engine.authorize(UserId("u_admin"), DocId("nope"), 1.0).allowed
    assert engine.filter(UserId("ghost"), 1000.0).doc_ids == frozenset()


groups = st.frozensets(st.sampled_from(["a", "b", "c", "d"]), max_size=4)


@given(person_groups=groups, policy_groups=groups, at=st.floats(0, 1e6))
def test_deny_always_wins(
    person_groups: frozenset[str], policy_groups: frozenset[str], at: float
) -> None:
    person = Person(UserId("u"), "dept", "role", person_groups)
    policy = DocPolicy(
        doc_id=DocId("d"),
        groups=policy_groups,
        departments=frozenset({"dept"}),
        roles=frozenset({"role"}),
        allow_users=frozenset({UserId("u")}),
        deny_users=frozenset({UserId("u")}),
    )
    assert evaluate(person, policy, at) == (False, "user denied")


@given(user=st.text(max_size=12), doc=st.text(max_size=12))
def test_unknown_inputs_are_denied(user: str, doc: str) -> None:
    w = world()
    engine = PolicyEngine(w.directory, w.policies)
    known_user = w.directory.get(UserId(user)) is not None
    known_doc = doc in {p["doc_id"] for p in MATRIX["policies"]}
    if not (known_user and known_doc):
        assert not engine.authorize(UserId(user), DocId(doc), 1000.0).allowed


class _Exploding:
    def __getattr__(self, name: str) -> object:
        raise RuntimeError("corrupted policy")


def test_corrupted_policy_fails_closed() -> None:
    w = world()
    engine = PolicyEngine(w.directory, w.policies)
    w.policies._policies[DocId("d01")] = _Exploding()  # type: ignore[assignment]
    decision = engine.authorize(UserId("u_cds_stu"), DocId("d01"), 1000.0)
    assert decision == decision.__class__(False, 1, "error")
    assert engine.filter(UserId("u_cds_stu"), 1000.0).doc_ids == frozenset()


def test_policy_versions_bump_on_change_and_revocation() -> None:
    store = PolicyStore()
    assert store.version == 1
    assert store.put(DocPolicy(doc_id=DocId("x"), groups=frozenset({"g"}))) == 2
    assert store.delete(DocId("x")) == 3
    assert store.snapshot() == (3, {})


def test_directory_formats_and_reload(tmp_path: Path) -> None:
    csv_text = "user_id,department,role,groups\nu1,cds,student,a;b\n"
    people = parse_export(csv_text, "csv")
    assert people[UserId("u1")].groups == frozenset({"a", "b"})
    path = tmp_path / "dir.json"
    path.write_text('[{"user_id": "u2", "groups": ["x"]}]')
    directory = Directory()
    assert directory.load(path) == 1
    assert directory.get(UserId("u2")) is not None
    assert directory.get(UserId("u1")) is None


@pytest.mark.parametrize(
    ("text", "fmt"),
    [
        ("[]", "xml"),
        ('[{"department": "cds"}]', "json"),
        ('[{"user_id": " "}]', "json"),
        ('[{"user_id": "u", "groups": 5}]', "json"),
    ],
)
def test_bad_directory_exports_are_rejected(text: str, fmt: str) -> None:
    with pytest.raises(ConfigError):
        parse_export(text, fmt)


def test_check_run_catches_injected_leak_and_false_deny() -> None:
    log = [
        LogEntry("u_cds_stu", 1000.0, retrieved=(("c1", "d05"),)),
        LogEntry("u_guest", 1000.0, retrieved=(("leak", "d02"),)),
        LogEntry("u_admin", 1000.0, denied_docs=("d07", "d02")),
    ]
    violations = check_run(log, MATRIX["directory"], MATRIX["policies"])
    kinds = [(v.kind, v.user_id, v.doc_id) for v in violations]
    assert kinds == [
        ("false_allow", "u_guest", "d02"),
        ("false_deny", "u_admin", "d07"),
    ]
