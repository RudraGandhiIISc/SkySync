"""Issue #11: one client spraying sign-ins must not lock out a browser that
has signed in to the account before.

Failed on rt-baseline (78c72ab) before the fix.
"""

from __future__ import annotations

from pathlib import Path

from askiisc.web.store import AppStore, load_key
from tests.rt_helpers import (
    student,
)
from tests.web_fixtures import HDR


async def test_issue11_a_spray_does_not_lock_out_a_known_browser(
    tmp_path: Path,
) -> None:
    web = await student(tmp_path)  # this browser has signed in before
    await web.client.post("/api/auth/logout", headers=HDR)
    throttle = web.deps.throttle
    assert throttle is not None
    for i in range(61):  # one client sprays distinct ids
        throttle.failed(f"nobody_{i}@iisc.ac.in")
    fresh = web.browser()
    assert (await web.login("u_cds_stu", fresh)).status_code == 429  # noqa: PLR2004 - spray protection kept
    assert (await web.login("u_cds_stu")).status_code == 200  # noqa: PLR2004
    device = web.client.cookies["askiisc_device"]
    web.account("u_cds_fac")
    other = await web.browser().post(  # the device cookie names one user
        "/api/auth/login",
        json={"user_id": "u_cds_fac", "password": "Correct-horse-9"},
        headers={**HDR, "Cookie": f"askiisc_device={device}"},
    )
    assert other.status_code == 429  # noqa: PLR2004


async def test_issue11_a_targeted_spray_does_not_lock_out_a_known_browser(
    tmp_path: Path,
) -> None:
    web = await student(tmp_path)  # this browser has signed in before
    await web.client.post("/api/auth/logout", headers=HDR)
    throttle = web.deps.throttle
    assert throttle is not None
    for _ in range(10):  # another client guesses this account's password
        throttle.failed("u_cds_stu")
    fresh = web.browser()
    assert (await web.login("u_cds_stu", fresh)).status_code == 429  # noqa: PLR2004 - per-address limit kept
    assert (await web.login("u_cds_stu")).status_code == 200  # noqa: PLR2004
    await web.client.post("/api/auth/logout", headers=HDR)
    wrong = {"user_id": "u_cds_stu", "password": "not-the-password"}
    for _ in range(6):  # the known browser's own failures still count
        r = await web.client.post("/api/auth/login", json=wrong, headers=HDR)
        assert r.status_code == 401  # noqa: PLR2004
    assert (await web.login("u_cds_stu")).status_code == 429  # noqa: PLR2004


def test_device_tokens_name_one_user_and_expire(tmp_path: Path) -> None:
    from askiisc.web.store import DEVICE_TTL_S  # new in the fix

    store = AppStore(tmp_path / "app.db", load_key(tmp_path / "chat.key"))
    token = store.device_token("u_cds_stu", 1000.0)
    assert store.device_user(token, 2000.0) == "u_cds_stu"
    assert store.device_user(token, 1000.0 + DEVICE_TTL_S + 1) is None
    assert store.device_user("not-a-token", 2000.0) is None
    other = AppStore(tmp_path / "b.db", load_key(tmp_path / "other.key"))
    assert other.device_user(token, 2000.0) is None  # another server key
