"""Issues #6 and #12: access is decided at the time of the request; `as_of`
only picks document versions and is never later than now.

Failed on rt-baseline (78c72ab) before the fix.
"""

from __future__ import annotations

import httpx
import pytest

from askiisc.access.service import create_app as create_access
from tests.access_fixtures import ISSUER, world
from tests.gateway_fixtures import build, final
from tests.rt_helpers import (
    QUESTION,
    retrieved,
)


async def test_issue6_a_future_as_of_never_opens_a_document_early() -> None:
    stack = build()
    question = "What does document d10 say about convocation dress?"
    now = await stack.ask("u_cds_stu", question)
    assert "d10" not in retrieved(now)  # control: opens at 2000, now 1000
    future = await stack.ask("u_cds_stu", question, as_of=2500.0)
    assert future.status_code == 200  # noqa: PLR2004
    assert "d10" not in retrieved(future)
    record = stack.audit.records[-1]
    assert record.as_of == stack.w.clock.now()  # clamped to the present
    assert "d10" not in {d.doc_id for d in record.documents}


@pytest.mark.parametrize("as_of", ["NaN", "Infinity"])
async def test_issue6_a_non_finite_as_of_is_refused(as_of: str) -> None:
    stack = build()
    r = await stack.client.post(
        "/v1/ask",
        content=f'{{"question": "{QUESTION}", "as_of": {as_of}}}',
        headers={
            "Authorization": f"Bearer {stack.w.token('u_cds_stu')}",
            "Content-Type": "application/json",
        },
    )
    assert r.status_code == 422  # noqa: PLR2004


async def test_issue12_an_expired_window_stays_closed_for_past_questions() -> (
    None
):
    stack = build()
    question = "What does document d11 section 0 say about hostel fees?"
    past = await stack.ask("u_cds_stu", question, as_of=100.0)
    assert past.status_code == 200  # noqa: PLR2004
    # d11 was readable until 500; now (1000) it is not, so its text must
    # not reach the model, whatever time the question is about.
    assert "d11" not in retrieved(past)
    assert "d11" not in {
        c["doc_id"]
        for c in final(past)["citations"]  # type: ignore[attr-defined]
    }


@pytest.mark.parametrize("path", ["/v1/authorize", "/v1/filter"])
@pytest.mark.parametrize("at", ["NaN", "Infinity", "-Infinity"])
async def test_issue6_the_access_service_refuses_a_non_finite_time(
    path: str, at: str
) -> None:
    w = world()
    app = create_access(
        w.directory, w.policies, ISSUER.verifier(w.clock), w.clock
    )
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://access"
    )
    doc = '"doc_id": "d10", ' if path == "/v1/authorize" else ""
    r = await client.post(
        path,
        content=f'{{{doc}"at": {at}}}',
        headers={
            "Authorization": f"Bearer {w.token('u_cds_stu')}",
            "Content-Type": "application/json",
        },
    )
    assert r.status_code == 422  # noqa: PLR2004 - NaN passed valid_from
