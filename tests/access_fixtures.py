"""Builders for access-control tests, from tests/fixtures/access_matrix.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from askiisc.access.directory import Directory
from askiisc.access.policy import DocPolicy, PolicyStore
from askiisc.common.clock import FakeClock
from askiisc.common.tokens import TokenIssuer

MATRIX: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "access_matrix.json").read_text()
)
ISSUER = TokenIssuer()  # one RSA key per test session: keygen is slow


@dataclass
class AccessWorld:
    clock: FakeClock
    directory: Directory
    policies: PolicyStore
    issuer: TokenIssuer

    def token(self, user: str, ttl: float = 3600.0) -> str:
        return self.issuer.issue(user, self.clock.now() + ttl)


def world(at: float | None = None) -> AccessWorld:
    clock = FakeClock(start=MATRIX["at"] if at is None else at)
    directory = Directory.from_records(MATRIX["directory"])
    policies = PolicyStore([DocPolicy(**p) for p in MATRIX["policies"]])
    return AccessWorld(clock, directory, policies, ISSUER)


def expected_cases() -> list[tuple[str, str, bool]]:
    docs = MATRIX["documents"]
    return [
        (user, docs[i], bit == "1")
        for user, bits in MATRIX["expected"].items()
        for i, bit in enumerate(bits)
    ]
