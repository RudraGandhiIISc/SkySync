"""FR-8: weight memory and KV-cache room from vLLM start-up logs."""

import json
from pathlib import Path

import pytest

from askiisc.observability.vllm_startup import kv_ratio, main, parse, record
from askiisc.web.models import comparison

FIXTURES = Path("tests/fixtures/vllm")


def test_the_8b_log_parses() -> None:
    s = parse((FIXTURES / "qwen3-8b-startup.log").read_text())
    assert s is not None
    assert (s.weights_gib, s.kv_cache_gib) == (15.2683, 58.66)
    assert s.kv_cache_tokens == 427_184  # noqa: PLR2004
    assert s.kv_cache_blocks == 26_699  # noqa: PLR2004
    assert (s.context_tokens, s.max_concurrency) == (16_384, 26.07)


def test_the_awq_log_parses_and_compares() -> None:
    awq = parse((FIXTURES / "qwen3-8b-awq-startup.log").read_text())
    bf16 = parse((FIXTURES / "qwen3-8b-startup.log").read_text())
    assert awq is not None and bf16 is not None
    assert awq.weights_gib == 5.7086  # noqa: PLR2004
    assert awq.gpu_memory_utilization == 0.61  # noqa: PLR2004
    # These two came from different free memory, so this ratio is not the
    # FR-8 verdict (that needs one fixed budget); it only shows the sum.
    assert kv_ratio(awq, bf16) == pytest.approx(537_504 / 427_184)


def test_an_incomplete_log_gives_nothing() -> None:
    assert parse("Model loading took 5.0 GiB and 1 seconds") is None
    s = parse(
        "Model loading took 5.0 GiB\nAvailable KV cache memory: 1.5 GiB\n"
        "GPU KV cache size: 1,000 tokens\n"
    )
    assert s is not None and s.max_concurrency is None
    assert s.kv_cache_blocks is None and s.gpu_memory_utilization is None


def test_record_keeps_other_fields_and_the_tab_shows_them(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "qwen3-8b"
    folder.mkdir()
    (folder / "metrics.json").write_text(json.dumps({"accuracy": 0.8}))
    s = parse((FIXTURES / "qwen3-8b-startup.log").read_text())
    assert s is not None
    record(folder, s, "vllm-1.log")
    data = json.loads((folder / "metrics.json").read_text())
    assert data["accuracy"] == 0.8  # noqa: PLR2004
    assert data["kv_cache_tokens"] == 427_184  # noqa: PLR2004
    assert data["startup"]["source"] == "vllm-1.log"
    row = next(
        r for r in comparison(tmp_path, "qwen3-8b") if r["key"] == "qwen3-8b"
    )
    assert row["metrics"]["weights_gib"]["value"] == 15.2683  # noqa: PLR2004
    assert row["metrics"]["kv_cache_tokens"]["value"] == 427_184  # noqa: PLR2004


def test_main(tmp_path: Path) -> None:
    log = FIXTURES / "qwen3-8b-startup.log"
    assert main([str(log), str(tmp_path), "qwen3-8b"]) == 0
    empty = tmp_path / "empty.log"
    empty.write_text("nothing")
    assert main([str(empty), str(tmp_path), "qwen3-8b"]) == 1
