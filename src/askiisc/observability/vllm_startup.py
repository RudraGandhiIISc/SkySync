"""Model memory and KV-cache room from a vLLM start-up log (FR-8).

FR-8 asks each 4-bit model to have at least 2x the KV-cache room of its
8B parent. vLLM 0.11 prints what it loaded and what is left, once per
start:

    Model loading took 15.2683 GiB and 3.36 seconds
    Available KV cache memory: 58.66 GiB
    GPU KV cache size: 427,184 tokens
    Maximum concurrency for 16,384 tokens per request: 26.07x
    ... num_gpu_blocks is: 26699

`parse` reads those lines (the first start in the log) and `record` adds
them to the model's `metrics.json` (the admin Models tab), keeping every
other field. Run:

    python -m askiisc.observability.vllm_startup <vllm log> <models dir> <key>
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_NUMBER = r"([0-9][0-9,]*(?:\.[0-9]+)?)"
PATTERNS = {
    "weights_gib": re.compile(rf"Model loading took {_NUMBER} GiB"),
    "kv_cache_gib": re.compile(rf"Available KV cache memory: {_NUMBER} GiB"),
    "kv_cache_tokens": re.compile(rf"GPU KV cache size: {_NUMBER} tokens"),
    "max_concurrency": re.compile(
        rf"Maximum concurrency for {_NUMBER} tokens per request: {_NUMBER}x"
    ),
    "kv_cache_blocks": re.compile(rf"num_gpu_blocks is: {_NUMBER}"),
    "gpu_memory_utilization": re.compile(
        rf"'gpu_memory_utilization': {_NUMBER}"
    ),
}


@dataclass(frozen=True)
class Startup:
    """What one vLLM start loaded and kept for the KV cache."""

    weights_gib: float
    kv_cache_gib: float
    kv_cache_tokens: int
    kv_cache_blocks: int | None
    context_tokens: int | None
    max_concurrency: float | None
    gpu_memory_utilization: float | None


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def parse(log: str) -> Startup | None:
    """The first start-up's numbers in a vLLM log.

    Args:
        log: The log text (colour codes are fine).

    Returns:
        The numbers, or None if the log has no complete start-up.
    """
    found: dict[str, re.Match[str]] = {}
    for name, pattern in PATTERNS.items():
        match = pattern.search(log)
        if match is not None:
            found[name] = match
    if not {"weights_gib", "kv_cache_gib", "kv_cache_tokens"} <= set(found):
        return None
    conc = found.get("max_concurrency")
    blocks = found.get("kv_cache_blocks")
    util = found.get("gpu_memory_utilization")
    return Startup(
        weights_gib=_num(found["weights_gib"].group(1)),
        kv_cache_gib=_num(found["kv_cache_gib"].group(1)),
        kv_cache_tokens=int(_num(found["kv_cache_tokens"].group(1))),
        kv_cache_blocks=None if blocks is None else int(_num(blocks.group(1))),
        context_tokens=None if conc is None else int(_num(conc.group(1))),
        max_concurrency=None if conc is None else _num(conc.group(2)),
        gpu_memory_utilization=None if util is None else _num(util.group(1)),
    )


def kv_ratio(child: Startup, parent: Startup) -> float:
    """How many times its parent's KV-cache tokens a model holds."""
    return child.kv_cache_tokens / parent.kv_cache_tokens


def record(folder: Path, startup: Startup, source: str) -> Path:
    """Adds the start-up numbers to a model's metrics.json.

    Args:
        folder: `<results>/models/<key>`.
        startup: The parsed numbers.
        source: The log they came from.

    Returns:
        The file written.
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "metrics.json"
    data: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    data["weights_gib"] = startup.weights_gib
    data["kv_cache_tokens"] = startup.kv_cache_tokens
    data["startup"] = {**asdict(startup), "source": source}
    path.write_text(json.dumps(data, indent=1))
    return path


def main(argv: Sequence[str] | None = None) -> int:
    """Parses one log into one model's metrics.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("models", type=Path)
    parser.add_argument("key")
    args = parser.parse_args(argv)
    startup = parse(args.log.read_text(errors="replace"))
    if startup is None:
        print(f"no vLLM start-up in {args.log}")  # noqa: T201
        return 1
    path = record(args.models / args.key, startup, str(args.log))
    print(f"wrote {path}: {asdict(startup)}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
