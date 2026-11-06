"""FR-7 ablation: vLLM prefix caching can be switched off."""

import pytest

from askiisc.autoscaler.pool import ProcessPool, ProcessPoolConfig
from askiisc.common.clock import FakeClock
from askiisc.common.config import Settings


def command(prefix: bool) -> list[str]:
    config = ProcessPoolConfig("v", ["m"], "/tmp", prefix_caching=prefix)
    pool = ProcessPool(config, FakeClock(), None, list, lambda: 1)  # type: ignore[arg-type]
    return pool.command(9000, 0.5)


def test_prefix_caching_flag_follows_the_config() -> None:
    assert "--enable-prefix-caching" in command(True)
    off = command(False)
    assert "--no-enable-prefix-caching" in off
    assert "--enable-prefix-caching" not in off


def test_the_setting_reads_on_and_off(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings().prefix_cache is True
    monkeypatch.setenv("ASKIISC_PREFIX_CACHE", "off")
    assert Settings().prefix_cache is False
