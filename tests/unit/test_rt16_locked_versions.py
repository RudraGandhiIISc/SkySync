"""Issue #16: no locked dependency with a known advisory.

Failed on rt-baseline (78c72ab) before the fix.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_issue16_opentelemetry_is_past_cve_2026_126360() -> None:
    lock = tomllib.loads((REPO / "uv.lock").read_text())
    versions = {p["name"]: p["version"] for p in lock["package"]}
    for name in ("opentelemetry-api", "opentelemetry-sdk"):
        parts = tuple(int(x) for x in versions[name].split(".")[:3])
        assert parts >= (1, 45, 2), name
