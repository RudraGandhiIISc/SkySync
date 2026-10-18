"""Marks each test by the directory it lives in."""

from __future__ import annotations

import pytest

_MARKS = {
    "integration": "integration",
    "e2e": "e2e",
    "perf": "perf",
    "security": "security",
    "regression": "regress",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        parts = item.path.parts
        for directory, mark in _MARKS.items():
            if directory in parts:
                item.add_marker(getattr(pytest.mark, mark))
                break
        else:
            item.add_marker(pytest.mark.unit)
