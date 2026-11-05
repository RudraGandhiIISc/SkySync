"""The serving job's FR-6 timing files."""

import json
from pathlib import Path

from deploy.serve import NodeTiming

from askiisc.observability.timing import ResponseTiming


def test_timing_files_are_written_and_closed(tmp_path: Path) -> None:
    timing = NodeTiming(tmp_path, profile=True)
    timing.response(ResponseTiming(1.0, "temporary", 429, "r", 0.5, 0.75))
    timing.close()
    row = json.loads((tmp_path / "responses.jsonl").read_text())
    assert (row["status"], row["done_ms"]) == (429, 0.75)
    assert (tmp_path / "loop_lag.jsonl").exists()
    assert (tmp_path / "profile.folded").exists()


def test_no_profile_file_without_the_switch(tmp_path: Path) -> None:
    NodeTiming(tmp_path, profile=False).close()
    assert not (tmp_path / "profile.folded").exists()


def test_gc_pauses_are_logged_while_timing_runs(tmp_path: Path) -> None:
    import gc

    before = len(gc.callbacks)
    timing = NodeTiming(tmp_path, profile=False)
    assert len(gc.callbacks) == before + 1
    gc.collect()
    timing.close()
    assert len(gc.callbacks) == before
    assert (tmp_path / "gc.jsonl").exists()
