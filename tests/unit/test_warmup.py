"""The warm-up asks until a worker is ready, then stops."""

import httpx
from eval.locust.warmup import warm_up


def client(ready_after: int, calls: list[str]) -> httpx.Client:
    asked = [0]

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/api/auth/login":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/api/temporary":
            asked[0] += 1
            return httpx.Response(200, content=b"event: answer\n\n")
        return httpx.Response(
            200, json={"workers_ready": asked[0] >= ready_after}
        )

    return httpx.Client(
        transport=httpx.MockTransport(handle), base_url="http://t"
    )


def test_it_asks_until_a_worker_is_ready() -> None:
    calls: list[str] = []
    t = [0.0]
    ok = warm_up(
        client(2, calls), "u", "p", wait_s=600,
        now=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s),
    )  # fmt: skip
    assert ok
    assert calls.count("/api/temporary") == 2  # noqa: PLR2004


def test_it_gives_up_after_the_wait() -> None:
    calls: list[str] = []
    t = [0.0]
    ok = warm_up(
        client(99, calls), "u", "p", wait_s=150,
        now=lambda: t[0], sleep=lambda s: t.__setitem__(0, t[0] + s),
    )  # fmt: skip
    assert not ok
    assert calls.count("/api/temporary") == 3  # noqa: PLR2004 - t=0, 60, 120
