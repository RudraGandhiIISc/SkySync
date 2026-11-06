"""FR-5: the answer cache forgets answers made before a revision shows."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from deploy.serve import make_publisher

from askiisc.common.clock import FakeClock


class FakeIngestor:
    """Ingests at once; the version becomes visible `delay` s later."""

    def __init__(self, clock: FakeClock, delay: float) -> None:
        self.clock, self.delay = clock, delay

    async def ingest(self, doc: Any, data: bytes, fp: str, now: float) -> Any:
        return SimpleNamespace(
            doc_id=doc.doc_id, version=2, chunks=1,
            visible_from=now + self.delay, retired=1,
        )  # fmt: skip


async def test_the_generation_moves_again_when_the_revision_is_visible(
    tmp_path: Path,
) -> None:
    # Regression (run 2143): answers cached in the commit delay still used
    # the old version and were served unflagged for minutes.
    clock = FakeClock()
    live = SimpleNamespace(generation=[0])
    scheduled: list[tuple[float, Any]] = []
    publish = make_publisher(
        FakeIngestor(clock, 5.0),  # type: ignore[arg-type]
        live,  # type: ignore[arg-type]
        clock,
        {},
        tmp_path,
        later=lambda delay, fn: scheduled.append((delay, fn)),
    )
    answer = await publish("doc-a", b"text", "text/plain", "")
    assert answer["version"] == 2  # noqa: PLR2004
    assert live.generation == [1]
    ((delay, bump),) = scheduled
    assert delay == 5.0  # noqa: PLR2004
    bump()
    assert live.generation == [2]  # noqa: PLR2004
