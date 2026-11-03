"""The access service as its own process (FR-4).

It used to share the gateway's event loop, so under load its 50 ms client
timeout fired while the gateway was busy. Run on its own, it answers from
its own loop. Policy changes made through the API are appended to an
overrides file and replayed at start-up, so they survive restarts. A users
tree is re-read within RELOAD_S of an admin's edit (DirectoryWatcher).

Usage (started by deploy/serve.py):
    python -m askiisc.access.server --port P --jwks JWKS --corpus CORPUS \
        --overrides FILE [--directory USERS]
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from pathlib import Path

from askiisc.access.directory import Directory, DirectoryWatcher
from askiisc.access.policy import DocPolicy, PolicyStore
from askiisc.common.types import DocId

RELOAD_S = 2.0


class OverrideLog:
    """Policy changes made at run time, one JSON line each."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def record(self, kind: str, what: DocPolicy | DocId) -> None:
        """Appends one change (the access service's on_change hook)."""
        if isinstance(what, DocPolicy):
            line = {"kind": kind, "policy": what.model_dump(mode="json")}
        else:
            line = {"kind": kind, "doc_id": str(what)}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a") as f:
            f.write(json.dumps(line) + "\n")

    def replay(self, store: PolicyStore) -> int:
        """Applies every recorded change in order; returns the count."""
        if not self._path.exists():
            return 0
        count = 0
        for line in self._path.read_text().splitlines():
            if not line.strip():
                continue
            change = json.loads(line)
            if change["kind"] == "put":
                store.put(DocPolicy(**change["policy"]))
            else:
                store.delete(DocId(change["doc_id"]))
            count += 1
        return count


def load_policies(corpus: Path, overrides: OverrideLog) -> PolicyStore:
    """The corpus policies with the run-time changes applied."""
    from askiisc.ingest.corpus import load_corpus  # noqa: PLC0415

    store = PolicyStore([d.policy for d in load_corpus(corpus)])
    overrides.replay(store)
    return store


def main() -> None:  # pragma: no cover - a process entry point
    """Loads the directory and policies and serves on 127.0.0.1."""
    import uvicorn  # noqa: PLC0415

    from askiisc.access.service import create_app  # noqa: PLC0415
    from askiisc.common.clock import SystemClock  # noqa: PLC0415
    from askiisc.common.config import Settings  # noqa: PLC0415
    from askiisc.common.tokens import TokenVerifier  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--jwks", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings()
    clock = SystemClock()
    directory = Directory()
    directory.load(args.directory)
    if args.directory.is_dir():
        watcher = DirectoryWatcher(directory, args.directory)
        threading.Thread(
            target=watcher.run_forever, args=(RELOAD_S,), daemon=True
        ).start()
    overrides = OverrideLog(args.overrides)
    policies = load_policies(args.corpus, overrides)
    verifier = TokenVerifier.from_file(
        args.jwks, settings.token_issuer, settings.token_audience, clock
    )
    app = create_app(
        directory, policies, verifier, clock, on_change=overrides.record
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    main()
