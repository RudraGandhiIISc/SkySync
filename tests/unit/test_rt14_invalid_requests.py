"""Issue #14: a question refused by validation (422) is audited and does not
use up the department's admission tokens.

Failed on rt-baseline (78c72ab) before the fix.
"""

from __future__ import annotations

from pathlib import Path

from tests.gateway_fixtures import build
from tests.rt_helpers import (
    CDS,
    QUESTION,
    student,
)
from tests.web_fixtures import HDR


async def test_issue14_a_malformed_question_is_audited_and_costs_nothing(
    tmp_path: Path,
) -> None:
    web = await student(tmp_path, rate=0.001, burst=3.0)
    long_question = {"question": "x" * 9000}
    bad = [
        long_question,
        {"question": ""},
        {"question": QUESTION, "unknown": 1},
    ]
    for body in bad:
        r = await web.client.post("/api/temporary", json=body, headers=HDR)
        assert r.status_code == 422  # noqa: PLR2004
        record = web.stack.audit.records[-1]
        assert (record.decision, record.reason) == (
            "rejected",
            "invalid_request",
        )
        assert record.tenant == "cds"
        assert r.headers["X-Request-Id"] == record.request_id
    malformed = await web.client.post(
        "/api/temporary",
        content=b"{not json",
        headers={**HDR, "Content-Type": "application/json"},
    )
    assert malformed.status_code == 422  # noqa: PLR2004
    assert web.stack.audit.records[-1].reason == "invalid_request"
    assert len(web.stack.audit.records) == 4  # noqa: PLR2004
    # The bucket holds 3 tokens and never refills: malformed requests took
    # none, so a colleague is still answered.
    web.account("u_cds_fac")
    colleague = web.browser()
    await web.login("u_cds_fac", colleague)
    ok = await colleague.post(
        "/api/temporary", json={"question": QUESTION}, headers=HDR
    )
    assert ok.status_code == 200  # noqa: PLR2004
    assert web.stack.deps.admission.depth(CDS) == 0


async def test_issue14_an_api_question_over_the_limit_is_audited() -> None:
    stack = build()
    r = await stack.ask("u_cds_stu", "x" * 9000)
    assert r.status_code == 422  # noqa: PLR2004
    record = stack.audit.records[-1]
    assert (record.decision, record.reason, record.tenant) == (
        "rejected",
        "invalid_request",
        "cds",
    )
    assert r.headers["X-Request-Id"] == record.request_id
    anonymous = await stack.ask(None, "")
    assert anonymous.status_code == 422  # noqa: PLR2004
    assert stack.audit.records[-1].tenant == "unknown"


async def test_issue14_unencodable_or_unparsable_bodies_are_audited(
    tmp_path: Path,
) -> None:
    web = await student(tmp_path)
    json_headers = {**HDR, "Content-Type": "application/json"}
    bodies = {
        '{"question": "hi", "\\ud800": 1}': 422,  # lone surrogate in a field
        '{"question": "' + "\\ud800" * 9000 + '"}': 422,  # ... and over-long
        b'{"question": "\xff"}': 400,  # not UTF-8: FastAPI cannot parse it
    }
    for body, status in bodies.items():
        r = await web.client.post(
            "/api/temporary", content=body, headers=json_headers
        )
        assert r.status_code == status
        record = web.stack.audit.records[-1]
        assert (record.reason, record.tenant) == ("invalid_request", "cds")
        assert r.headers["X-Request-Id"] == record.request_id
    assert len(web.stack.audit.records) == len(bodies)
    assert web.stack.deps.admission.depth(CDS) == 0
