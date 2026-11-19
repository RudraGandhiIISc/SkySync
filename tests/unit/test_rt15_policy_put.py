"""Issue #15: PUT /api/admin/policies/{id} checks the admin first and answers
422 for a body that is not JSON.

Failed on rt-baseline (78c72ab) before the fix.
"""

from __future__ import annotations

from pathlib import Path

from tests.rt_helpers import (
    student,
)
from tests.web_fixtures import HDR, build_web


async def test_issue15_policy_put_checks_admin_then_the_body(
    tmp_path: Path,
) -> None:
    url = "/api/admin/policies/d02"
    json_headers = {**HDR, "Content-Type": "application/json"}
    for sub in ("student", "admin"):
        (tmp_path / sub).mkdir()
    stu = await student(tmp_path / "student")
    for body in (b"{bad json", b'{"owner": "nosuchdept"}'):
        r = await stu.client.put(url, content=body, headers=json_headers)
        assert r.status_code == 403  # noqa: PLR2004 - was 500 and 400
    web = build_web(tmp_path / "admin")
    await web.institute_admin(tmp_path / "admin" / "users")
    for body in (b"{bad json", b""):
        r = await web.client.put(url, content=body, headers=json_headers)
        assert r.status_code == 422  # noqa: PLR2004 - was 500
    assert (await web.client.get(url)).status_code == 200  # noqa: PLR2004
