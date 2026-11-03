"""Load tests of the GPU node alone (E0 / NFR-3, FR-7).

The prompts are the ones the gateway would send: the fixed system prompt,
the real chunks the held-out run retrieved for each question (trimmed to
the same context budget) and the question. Sending them straight to vLLM
measures the model server without the retrieval stage in front of it (on
this server Milvus Lite caps the full system near 1 request/s).

Usage (inside slurm/llm_job.sh):
    python -m eval.llm_bench sweep RESULTS.json STATE OUT [--steps 1,2,4]
    python -m eval.llm_bench fixed RESULTS.json STATE OUT --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

from askiisc.common.targets import LLM_MAX_TOKENS, NFR3_CONCURRENCY
from askiisc.common.types import Chunk, ChunkId, DocId, Version
from askiisc.llm.prompts import build_messages, fit_context
from eval.grader.rubric import ChunkTexts
from eval.measurements import write_measurement

Message = dict[str, str]


def _retrieved(item: dict[str, Any]) -> list[tuple[str, str]]:
    """(chunk id, doc id) pairs from either results format.

    Held-out runs keep `answer.retrieved` as pairs; edition-aware runs
    (eval/versioned) keep a top-level `retrieved` list of chunk ids, whose
    document is the part before "@".
    """
    answer = item.get("answer")
    if isinstance(answer, dict) and answer.get("retrieved"):
        return [(str(c), str(d)) for c, d in answer["retrieved"]]
    return [
        (str(c), str(c).split("@", 1)[0]) for c in item.get("retrieved") or []
    ]


def prompts(results: Path, state: Path) -> list[list[Message]]:
    """Gateway-identical prompts for every held-out question with chunks."""
    texts = ChunkTexts(state)
    out = []
    for item in json.loads(results.read_text()):
        chunks = []
        for chunk_id, doc_id in _retrieved(item):
            m = re.match(r"^.*@v(\d+)#\d+$", chunk_id)
            chunks.append(
                Chunk(
                    id=ChunkId(chunk_id),
                    doc_id=DocId(doc_id),
                    version=Version(int(m.group(1)) if m else 1),
                    text=texts.text(chunk_id) or " ",
                    page=1,
                    valid_from=0.0,
                )
            )
        if chunks:
            out.append(build_messages(fit_context(chunks), item["question"]))
    return out


def pct(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile."""
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, math.ceil(q / 100 * len(s)) - 1)]


async def metrics(http: httpx.AsyncClient, url: str) -> dict[str, float]:
    """VLLM counters."""
    out: dict[str, float] = {}
    text = (await http.get(url + "/metrics", timeout=10)).text
    for line in text.splitlines():
        m = re.match(
            r"^(vllm:[a-z_]+_total)(?:\{[^}]*\})? ([0-9.eE+-]+)$", line
        )
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


async def one(
    http: httpx.AsyncClient, url: str, model: str, messages: list[Message]
) -> dict[str, Any]:
    """Streams one completion; returns timings and token counts."""
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": LLM_MAX_TOKENS,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft, tokens, ok = None, 0, False
    try:
        async with http.stream(
            "POST", url + "/v1/chat/completions", json=body, timeout=600
        ) as r:
            ok = r.status_code == httpx.codes.OK
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line.endswith("[DONE]"):
                    continue
                event = json.loads(line[6:])
                if event.get("usage"):
                    tokens = int(event["usage"]["completion_tokens"])
                choices = event.get("choices") or []
                if (
                    ttft is None
                    and choices
                    and choices[0]["delta"].get("content")
                ):
                    ttft = time.perf_counter() - t0
    except httpx.HTTPError:
        ok = False
    return {
        "ok": ok,
        "ttft_s": ttft,
        "total_s": time.perf_counter() - t0,
        "tokens": tokens,
    }


async def closed_loop(
    url: str,
    model: str,
    pool: list[list[Message]],
    concurrency: int,
    seconds: float,
) -> dict[str, Any]:
    """`concurrency` clients, no think time, for `seconds`."""
    limits = httpx.Limits(max_connections=concurrency + 8)
    async with httpx.AsyncClient(limits=limits) as http:
        before = await metrics(http, url)
        end = time.monotonic() + seconds
        rows: list[dict[str, Any]] = []
        counter = [0]

        async def client() -> None:
            while time.monotonic() < end:
                i = counter[0]
                counter[0] += 1
                rows.append(await one(http, url, model, pool[i % len(pool)]))

        t0 = time.monotonic()
        await asyncio.gather(*(client() for _ in range(concurrency)))
        took = time.monotonic() - t0
        after = await metrics(http, url)
    delta = {k: after[k] - before.get(k, 0.0) for k in after}
    ok = [r for r in rows if r["ok"]]
    hits = delta.get("vllm:prefix_cache_hits_total", 0.0)
    queries = delta.get("vllm:prefix_cache_queries_total", 0.0)
    return {
        "concurrency": concurrency,
        "seconds": round(took, 1),
        "requests": len(rows),
        "ok": len(ok),
        "req_per_s": round(len(ok) / took, 3),
        "output_tokens_per_s": round(
            delta.get("vllm:generation_tokens_total", 0.0) / took, 1
        ),
        "client_tokens_per_s": round(sum(r["tokens"] for r in ok) / took, 1),
        "prompt_tokens_per_s": round(
            delta.get("vllm:prompt_tokens_total", 0.0) / took, 1
        ),
        "prefix_cache_hit_rate": round(hits / queries, 4) if queries else None,
        "total_p50_s": pct([r["total_s"] for r in ok], 50),
        "total_p95_s": pct([r["total_s"] for r in ok], 95),
        "total_p99_s": pct([r["total_s"] for r in ok], 99),
        "ttft_p50_s": pct([r["ttft_s"] for r in ok if r["ttft_s"]], 50),
        "ttft_p95_s": pct([r["ttft_s"] for r in ok if r["ttft_s"]], 95),
    }


def main() -> None:
    """Runs a sweep or one fixed-concurrency step and writes JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["sweep", "fixed"])
    parser.add_argument("results", type=Path)
    parser.add_argument("state", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--steps", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--measurements", type=Path, help="send NFR-3 to the scorecard"
    )
    args = parser.parse_args()
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    url, model = os.environ["ASKIISC_LLM_URL"], os.environ["ASKIISC_LLM_MODEL"]
    pool = prompts(args.results, args.state)
    print(f"{len(pool)} prompts")
    steps = (
        [int(s) for s in args.steps.split(",")]
        if args.mode == "sweep"
        else [args.concurrency]
    )
    rows = []
    asyncio.run(closed_loop(url, model, pool, min(8, steps[0]), 20.0))  # warm
    for c in steps:
        row = asyncio.run(closed_loop(url, model, pool, c, args.seconds))
        row["label"] = args.label
        row["vllm_extra_args"] = os.environ.get("VLLM_EXTRA_ARGS", "")
        rows.append(row)
        print(json.dumps(row))
        name = (
            "sweep.json" if args.mode == "sweep" else f"fixed_{args.label}.json"
        )
        (args.out / name).write_text(json.dumps(rows, indent=1))
    at_64 = [r for r in rows if r.get("concurrency") == NFR3_CONCURRENCY]
    if args.measurements and at_64:
        write_measurement(
            args.measurements, "throughput", "NFR-3", at_64[0], "Benchmark",
            started, model=model,
        )  # fmt: skip


if __name__ == "__main__":
    main()
