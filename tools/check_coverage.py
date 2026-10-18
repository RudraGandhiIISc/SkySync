"""Fails unless the named files or packages have 100% line coverage.

Usage: python tools/check_coverage.py build/coverage.json PATH...
A PATH is a file or a directory under src/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def uncovered(report: dict[str, object], prefixes: list[str]) -> list[str]:
    """Lists files under the prefixes whose line coverage is below 100%.

    Args:
        report: Parsed coverage.py JSON report.
        prefixes: File or directory paths, relative to the repository root.

    Returns:
        One "path: missing lines" entry per file below 100%.
    """
    files = report["files"]
    if not isinstance(files, dict):
        raise TypeError("not a coverage.py JSON report")
    problems = []
    for path, data in sorted(files.items()):
        if not any(path.startswith(p) for p in prefixes):
            continue
        missing = data["missing_lines"]
        if missing:
            problems.append(f"{path}: missing lines {missing}")
    return problems


def main(argv: list[str]) -> int:
    """Runs the check.

    Args:
        argv: The report path followed by the paths that need 100%.

    Returns:
        The process exit code.
    """
    report = json.loads(Path(argv[0]).read_text())
    prefixes = argv[1:]
    files = report["files"]
    known = [p for p in prefixes if any(f.startswith(p) for f in files)]
    for p in sorted(set(prefixes) - set(known)):
        print(f"not measured yet (no file): {p}")
    problems = uncovered(report, known)
    for line in problems:
        print(line)
    if problems:
        print("100% coverage required on the files above")
        return 1
    print("100% coverage files: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
