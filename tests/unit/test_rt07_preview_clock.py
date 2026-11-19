"""Issue #7: the no-GPU preview's clock follows wall time, so a department
that used its burst is served again.

Failed on rt-baseline (78c72ab) before the fix.
"""

from __future__ import annotations

from pathlib import Path

from tools.ui_preview import catch_up

from tests.rt_helpers import (
    CDS,
    QUESTION,
    student,
)
from tests.web_fixtures import HDR


async def test_issue7_a_department_recovers_once_the_preview_clock_moves(
    tmp_path: Path,
) -> None:
    web = await student(tmp_path, rate=1.0, burst=3.0)  # the preview's stack
    url, body = "/api/temporary", {"question": QUESTION}
    for _ in range(3):
        ok = await web.client.post(url, json=body, headers=HDR)
        assert ok.status_code == 200  # noqa: PLR2004
    refused = await web.client.post(url, json=body, headers=HDR)
    assert refused.status_code == 429  # noqa: PLR2004 - the bucket is empty
    assert web.stack.deps.admission.depth(CDS) == 0  # nothing in flight leaked
    clock = web.stack.w.clock
    catch_up(clock, lambda: clock.now() + 5.0)  # what the preview now does
    again = await web.client.post(url, json=body, headers=HDR)
    assert again.status_code == 200  # noqa: PLR2004
