"""Recall of the serving index under the permission filter (FR-2).

The serving job searches with `DenseIndex`. This check probes it with
query vectors drawn from the real stored chunk vectors (a random chunk's
vector plus noise), each as a random user from the users tree with the
filter from the policy engine, and compares the top k with a separate
brute-force search:

- the reference reads the snapshot files itself (not through DenseIndex);
- which documents a user may read comes from the oracle (independent code);
- recall@k = returned chunks the user may read whose score reaches the
  reference k-th score (ties count as hits), divided by k (or by the number
  of readable chunks, if fewer), averaged over the probes;
- a returned chunk the oracle forbids, or one not valid now, is a leak.

Usage (inside Slurm):
    python -m eval.probes.recall --state DIR --corpus DIR --users DIR \
        --results DATA/results/measurements

It writes <results>/<date>-recall/summary.json, which the admin scorecard
reads for FR-2 (no user ids in it).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from askiisc.common.types import DocId
from askiisc.retrieval.dense import DenseIndex
from eval.measurements import write_measurement
from eval.oracle.oracle import oracle_allows

TIE_EPS = 1e-5
NOISE = 0.5
Floats = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Snapshot:
    """The snapshot files, read independently of DenseIndex."""

    vectors: Floats
    row_doc: list[str]
    row_of: dict[str, int]
    valid_from: npt.NDArray[np.float64]
    valid_to: npt.NDArray[np.float64]


def read_snapshot(folder: Path) -> Snapshot:
    """Reads vectors, documents and validity from a dense snapshot."""
    vectors = np.load(folder / "vectors.npy")
    meta = np.load(folder / "meta.npz")
    docs = json.loads((folder / "docs.json").read_text())
    with (folder / "rows.jsonl").open() as f:
        ids = [json.loads(line)[0] for line in f]
    return Snapshot(
        vectors=vectors.astype(np.float32, copy=False),
        row_doc=[docs[d] for d in meta["doc"].tolist()],
        row_of={c: i for i, c in enumerate(ids)},
        valid_from=meta["valid_from"],
        valid_to=meta["valid_to"],
    )


def readable_docs(
    directory: Sequence[Mapping[str, Any]],
    policies: Sequence[Mapping[str, Any]],
    user: str,
    at: float,
) -> set[str]:
    """Every document the oracle lets the user read."""
    return {
        str(p["doc_id"])
        for p in policies
        if oracle_allows(directory, [p], user, str(p["doc_id"]), at)
    }


def probe_vector(snap: Snapshot, rng: random.Random) -> Floats:
    """A stored chunk vector plus noise, unit length."""
    base = snap.vectors[rng.randrange(len(snap.vectors))]
    noise = np.asarray([rng.gauss(0, 1) for _ in range(base.shape[0])])
    noise = noise / np.linalg.norm(noise) * NOISE
    q = (base + noise).astype(np.float32)
    return q / np.linalg.norm(q)


def _row_mask(snap: Snapshot, docs: set[str], at: float) -> Any:
    mask = np.fromiter((d in docs for d in snap.row_doc), bool)
    return mask & (snap.valid_from <= at) & (snap.valid_to > at)


def one_probe(
    snap: Snapshot,
    mask: Any,
    readable: set[str],
    returned: Sequence[str],
    q: Floats,
    k: int,
) -> tuple[float, int]:
    """Recall of one search and its number of leaked chunks."""
    scores = snap.vectors @ q
    allowed = np.flatnonzero(mask)
    need = min(k, len(allowed))
    if need == 0:
        return (1.0 if not returned else 0.0), len(returned)
    kth = float(np.sort(scores[allowed])[-need])
    hits = leaks = 0
    for chunk_id in returned:
        row = snap.row_of.get(chunk_id)
        if row is None or not mask[row] or snap.row_doc[row] not in readable:
            leaks += 1
        elif scores[row] >= kth - TIE_EPS:
            hits += 1
    return min(1.0, hits / need), leaks


def run_probes(
    index: DenseIndex,
    snap: Snapshot,
    filters: Mapping[str, frozenset[DocId]],
    readable: Mapping[str, set[str]],
    n: int,
    k: int,
    rng: random.Random,
    now: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Runs n probes and returns the FR-2 summary checks."""
    at = now()
    users = sorted(filters)
    masks = {u: _row_mask(snap, readable[u], at) for u in users}
    recalls: list[float] = []
    leaks = 0
    search_ms: list[float] = []
    for _ in range(n):
        user = rng.choice(users)
        q = probe_vector(snap, rng)
        started = time.perf_counter()
        hits = index.search(q.tolist(), filters[user], at, k)
        search_ms.append((time.perf_counter() - started) * 1e3)
        recall, leaked = one_probe(
            snap, masks[user], readable[user], [h.chunk.id for h in hits], q, k
        )
        recalls.append(recall)
        leaks += leaked
    search_ms.sort()
    return {
        "probes": n,
        "recall_at_k": sum(recalls) / max(1, n),
        "min_recall": min(recalls, default=1.0),
        "leaks": leaks,
        "users": len(users),
        "search_ms_p95": search_ms[math.ceil(0.95 * n) - 1] if n else 0.0,
    }


def main() -> None:  # pragma: no cover - runs on the real data in Slurm
    """Writes <results>/<date>-recall/summary.json (workload "recall")."""
    from askiisc.access.directory import Directory  # noqa: PLC0415
    from askiisc.access.policy import PolicyEngine, PolicyStore  # noqa: PLC0415
    from askiisc.common.targets import FR2_ACCESS_PROBES, RETRIEVAL_TOP_K  # noqa: PLC0415
    from askiisc.common.types import UserId  # noqa: PLC0415
    from askiisc.ingest.corpus import load_corpus  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("--state", "--corpus", "--users", "--results"):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument("--probes", type=int, default=FR2_ACCESS_PROBES)
    args = parser.parse_args()
    started = time.time()
    directory = Directory()
    directory.load(args.users)
    docs = load_corpus(args.corpus)
    engine = PolicyEngine(directory, PolicyStore([d.policy for d in docs]))
    people = [
        {"user_id": p.user_id, "department": p.department, "role": p.role,
         "groups": sorted(p.groups)}
        for p in directory._people.values()  # noqa: SLF001
    ]  # fmt: skip
    policies = [d.policy.model_dump(mode="json") for d in docs]
    users = [str(p["user_id"]) for p in people]
    filters = {u: engine.filter(UserId(u), started).doc_ids for u in users}
    readable = {u: readable_docs(people, policies, u, started) for u in users}
    index = DenseIndex()
    index.load(args.state / "dense")
    snap = read_snapshot(args.state / "dense")
    checks = run_probes(
        index, snap, filters, readable, args.probes, RETRIEVAL_TOP_K,
        random.Random(252),
    )  # fmt: skip
    path = write_measurement(
        args.results, "recall", "FR-2", checks, "Probe", started,
        chunks=len(snap.row_doc),
    )  # fmt: skip
    print(path.read_text())


if __name__ == "__main__":  # pragma: no cover
    main()
