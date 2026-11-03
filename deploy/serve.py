"""Serves Ask IISc for people: the chat app, admin dashboard and monitoring.

Runs inside one GPU Slurm job (slurm/serve.sh) until the job ends or gets
SIGTERM. Everything binds to 127.0.0.1 (CLAUDE.md safety rule 8); people
reach it through an SSH tunnel. Started, in order:

  1. embedding server (serve venv), subprocess
  2. access service, its own process (askiisc.access.server)
  3. the index: the dense snapshot of state/ (exact, in memory) over Milvus
  4. autoscaler with vLLM workers, 0 at rest, at most 2
  5. gateway + web app (chat at /, admin at /admin), port ASKIISC_WEB_PORT
  6. Prometheus, Grafana and the Locust web UI, subprocesses

Private files (accounts, chats, request log, keys) live in
$ASKIISC_DATA/private/app (mode 700). The ports and links of the running
instance are written to $ASKIISC_DATA/run/current.json.

Usage (inside slurm/serve.sh):
    python -m deploy.serve --run-dir DIR --corpus CORPUS --state-dir STATE
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import secrets
import shutil
import signal
import socket
import sqlite3
import subprocess  # nosec B404 - only the system htpasswd, fixed arguments
import sys
import time
from collections.abc import AsyncIterator, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import uvicorn

from askiisc.access.client import HttpAccessClient
from askiisc.access.directory import Directory, DirectoryWatcher
from askiisc.autoscaler.controller import Controller
from askiisc.autoscaler.policy import ScalingConfig
from askiisc.autoscaler.pool import ProcessPool
from askiisc.common.clock import SystemClock
from askiisc.common.config import Settings
from askiisc.common.errors import ConfigError
from askiisc.common.tokens import TokenIssuer
from askiisc.common.types import DocId
from askiisc.gateway.app import GatewayDeps
from askiisc.gateway.app import create_app as create_gateway
from askiisc.gateway.audit import FileAuditSink
from askiisc.gateway.auth import Authenticator
from askiisc.gateway.cache import InMemoryAnswerCache
from askiisc.gateway.drr import FairDispatcher
from askiisc.gateway.limits import Admission
from askiisc.ingest.corpus import CorpusDoc, public_rows, read_titles
from askiisc.ingest.editions import academic_year_label
from askiisc.ingest.objects import FileObjectStore
from askiisc.ingest.pipeline import Ingestor
from askiisc.ingest.reconciler import SourceDocument
from askiisc.ingest.versions import SQLiteVersionStore
from askiisc.llm.client import LLMClient
from askiisc.llm.models import select_model
from askiisc.llm.rag import RagAnswerer
from askiisc.observability.metrics import GatewayMetrics
from askiisc.observability.vllm_stats import ServingStats, parse_metrics
from askiisc.retrieval.dense import DenseIndex, open_dense
from askiisc.retrieval.embedder import HttpEmbedder, query_prefix
from askiisc.retrieval.rerank import HttpReranker
from askiisc.retrieval.search import Retriever
from askiisc.retrieval.store import MilvusIndex
from askiisc.web.app import InternalTokens, WebDeps, mount_web
from askiisc.web.dbfiles import BACKUP_INTERVAL_S, backup, db_dir, restore
from askiisc.web.documents import find_documents
from askiisc.web.mailer import SmtpMailer, load_smtp
from askiisc.web.people import PeopleTree
from askiisc.web.requestlog import RequestLog
from askiisc.web.store import AppStore, load_key
from deploy import logs
from deploy.stack import (
    embed_server,
    load_world,
    make_controller,
    make_pool,
    wait_http,
)

LOG = logging.getLogger("askiisc.serve")
REVISION_COMMIT_DELAY_S = 5.0
LOOP_S = 5.0

DEFAULT_PORTS = {
    "web": 18000,
    "grafana": 13000,
    "prometheus": 19090,
    "locust": 18089,
}


def port_free(port: int) -> bool:
    """Whether 127.0.0.1:port can be bound, as uvicorn binds it.

    SO_REUSEADDR (which uvicorn sets) lets a port be bound while old
    connections of the previous serving job sit in TIME_WAIT; a listener
    still makes the port busy.
    """
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def choose_ports() -> dict[str, int]:
    """Fixed, overridable ports (stable tunnels); fails if one is taken.

    Raises:
        RuntimeError: If a port is already in use on this server.
    """
    ports = {
        name: int(os.environ.get(f"ASKIISC_{name.upper()}_PORT", default))
        for name, default in DEFAULT_PORTS.items()
    }
    busy = [f"{n}={p}" for n, p in ports.items() if not port_free(p)]
    if busy:
        raise RuntimeError(
            f"ports in use: {', '.join(busy)}; set ASKIISC_<NAME>_PORT"
        )
    return ports


def edition_label(entry: dict[str, Any]) -> str:
    """The edition in the terms questions use.

    'Aug 2025, academic year 2025-26'; 'current' for the undated newest
    edition of a document with several.
    """
    label, start = entry.get("edition") or "", entry.get("edition_start")
    if label and start:
        return f"{label}, academic year {academic_year_label(start)}"
    if label:
        return str(label)
    several = int(entry.get("editions") or 1) > 1
    return "current" if several and entry.get("valid_to") is None else ""


def doc_catalogue(corpus: Path, docs: list[CorpusDoc]) -> dict[str, dict]:
    """Title, address, kind and edition per document, for citations.

    Titles come from the catalogue, the manifest or titles.json (written by
    `python -m askiisc.ingest.corpus`), else from the address. Documents
    with several editions also carry the edition label.
    """
    catalogue: dict[str, dict[str, str]] = {}
    titles = read_titles(corpus)
    for entry in public_rows(corpus):
        url = entry.get("url", "")
        parsed = urlparse(entry.get("original_url") or url)
        path = parsed.path.strip("/").split("/")[-1] or parsed.netloc
        slug = path.rsplit(".", 1)[0].replace("-", " ").replace("_", " ")
        title = entry.get("title") or titles.get(entry["doc_id"]) or slug
        info = {
            "title": title if url else entry["doc_id"],
            "url": url,
            "kind": Path(entry.get("path", "")).suffix.lstrip("."),
            "site": parsed.netloc,
        }
        edition = edition_label(entry)
        if edition:  # any dated document, so the model can tell 2024 from 2026
            info["edition"] = edition
        if entry.get("snapshot") is not None:
            info["archived"] = "yes"
        catalogue[entry["doc_id"]] = info
    for doc in docs:
        catalogue.setdefault(
            doc.policy.doc_id,
            {"title": doc.policy.doc_id.replace("-", " "), "url": ""},
        )
    return catalogue


@dataclass
class Children:
    """Subprocesses started by the supervisor."""

    procs: dict[str, asyncio.subprocess.Process] = field(default_factory=dict)

    async def start(
        self, name: str, cmd: list[str], log: Path, env: dict[str, str]
    ) -> asyncio.subprocess.Process:
        """Starts one subprocess in its own process group."""
        with log.open("ab") as out:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=env,
                stdout=out,
                stderr=out,
                start_new_session=True,
            )
        self.procs[name] = proc
        with (log.parent / "pids").open("a") as pids:
            pids.write(f"{proc.pid}\n")  # process groups for slurm/serve.sh
        LOG.info("started %s (pid %d), log %s", name, proc.pid, log)
        return proc

    async def stop(self, name: str) -> None:
        """Stops one subprocess (SIGTERM, then SIGKILL)."""
        proc = self.procs.pop(name, None)
        if proc is None or proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=20)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()

    async def stop_all(self) -> None:
        """Stops every subprocess, newest first."""
        for name in reversed(list(self.procs)):
            await self.stop(name)


@dataclass
class Paths:
    """Where a serving run reads and writes."""

    run: Path
    corpus: Path
    state: Path
    data: Path

    @property
    def app(self) -> Path:
        """Private app data: accounts, chats, request log, keys."""
        return self.data / "private" / "app"

    @property
    def services(self) -> Path:
        """Prometheus and Grafana binaries (slurm/fetch_services.sh)."""
        return self.data / "services"

    @property
    def targets(self) -> Path:
        """Prometheus file discovery folder (vLLM workers, Locust)."""
        return self.data / "run" / "prom_targets"


def secret_file(path: Path, nbytes: int = 24) -> str:
    """Reads a secret, creating it (mode 600) on first use."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600)
        path.write_text(secrets.token_urlsafe(nbytes))
    return path.read_text().strip()


def prometheus_config(
    paths: Paths, ports: dict[str, int], access_port: int
) -> Path:
    """Writes prometheus.yml: the gateway, access service and file targets."""
    config = {
        "global": {"scrape_interval": "5s", "evaluation_interval": "5s"},
        "scrape_configs": [
            {
                "job_name": "gateway",
                "static_configs": [{"targets": [f"127.0.0.1:{ports['web']}"]}],
            },
            {
                "job_name": "access",
                "static_configs": [{"targets": [f"127.0.0.1:{access_port}"]}],
            },
            {
                "job_name": "vllm",
                "file_sd_configs": [
                    {"files": [str(paths.targets / "vllm.json")]}
                ],
            },
            {
                "job_name": "locust",
                "file_sd_configs": [
                    {"files": [str(paths.targets / "locust*.json")]}
                ],
            },
        ],
    }
    path = paths.run / "prometheus.yml"
    path.write_text(json.dumps(config, indent=1))  # JSON is valid YAML
    return path


PROMETHEUS_USER = "prometheus"
# Threads for index searches, apart from the default pool (logins, chats).
SEARCH_THREADS = 4
LOCUST_UI_USER = "admin"


def _loki_source(loki: tuple[int, str] | None) -> list[dict[str, Any]]:
    if loki is None:
        return []
    port, password = loki
    return [
        {
            "name": "Loki",
            "uid": "askiisc-loki",
            "type": "loki",
            "access": "proxy",
            "url": f"http://127.0.0.1:{port}",
            "basicAuth": True,
            "basicAuthUser": logs.LOKI_USER,
            "secureJsonData": {"basicAuthPassword": password},
        }
    ]


async def start_logs(
    children: Children, paths: Paths
) -> tuple[int, str] | None:
    """Starts Loki, its auth proxy and Alloy.

    Returns:
        The proxy's port and password, or None if the binaries are missing.
    """
    loki_bin = paths.services / "loki" / "loki"
    alloy_bin = paths.services / "alloy" / "alloy"
    if not (loki_bin.exists() and alloy_bin.exists()):
        return None
    env = dict(os.environ)
    password_file = paths.app / "loki_password"
    password = secret_file(password_file)
    certs = logs.make_certs(paths.app / "loki-tls")
    config, socket_path = logs.loki_config(
        paths.run, paths.data / "loki", free_port(), certs
    )
    await children.start(
        "loki", logs.loki_command(loki_bin, config, certs),
        paths.run / "loki.log", env,
    )  # fmt: skip
    port = free_port()
    await children.start(
        "loki-proxy",
        [
            sys.executable, "-m", "askiisc.observability.authproxy",
            "--port", str(port), "--socket", str(socket_path),
            "--user", logs.LOKI_USER, "--password-file", str(password_file),
        ],
        paths.run / "loki-proxy.log",
        env,
    )  # fmt: skip
    job = os.environ.get("SLURM_JOB_ID", "local")
    alloy = logs.alloy_config(paths.run, port, password_file, job)
    await children.start(
        "alloy",
        logs.alloy_command(alloy_bin, alloy, paths.data / "alloy", free_port()),
        paths.run / "alloy.log",
        env,
    )
    return port, password


def prometheus_auth(paths: Paths) -> tuple[str, Path]:
    """Basic auth for Prometheus: the password and its web config file.

    Prometheus has no accounts; its web config takes bcrypt hashes, made
    here with the system `htpasswd` (the password goes in on stdin, never
    on a command line other users could see).

    Raises:
        RuntimeError: If htpasswd is missing or fails.
    """
    password = secret_file(paths.app / "prometheus_password")
    htpasswd = shutil.which("htpasswd")
    if htpasswd is None:
        raise RuntimeError("htpasswd not found: cannot protect Prometheus")
    out = subprocess.run(  # noqa: S603 - fixed binary and arguments
        [htpasswd, "-niBC", "10", PROMETHEUS_USER],
        input=password,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    digest = out.split(":", 1)[1]
    config = paths.run / "prometheus-web.yml"
    config.write_text(
        json.dumps({"basic_auth_users": {PROMETHEUS_USER: digest}})
    )
    return password, config


def grafana_provisioning(
    paths: Paths,
    ports: dict[str, int],
    prometheus_password: str,
    loki: tuple[int, str] | None = None,
) -> Path:
    """Writes Grafana's data sources and dashboard provisioning.

    Args:
        paths: The run's folders.
        ports: The fixed ports.
        prometheus_password: Prometheus' basic-auth password.
        loki: The Loki proxy's port and password, if Loki runs.

    Returns:
        The provisioning folder.
    """
    root = paths.run / "grafana-provisioning"
    for sub in ("datasources", "dashboards", "plugins", "alerting"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "datasources" / "prometheus.yaml").write_text(
        json.dumps(
            {
                "apiVersion": 1,
                "datasources": [
                    {
                        "name": "Prometheus",
                        "uid": "askiisc-prom",
                        "type": "prometheus",
                        "access": "proxy",
                        "url": f"http://127.0.0.1:{ports['prometheus']}",
                        "isDefault": True,
                        "basicAuth": True,
                        "basicAuthUser": PROMETHEUS_USER,
                        "secureJsonData": {
                            "basicAuthPassword": prometheus_password
                        },
                    },
                    *_loki_source(loki),
                ],
            }
        )
    )
    dashboards = Path(__file__).parent / "grafana"
    (root / "dashboards" / "askiisc.yaml").write_text(
        json.dumps(
            {
                "apiVersion": 1,
                "providers": [
                    {
                        "name": "askiisc",
                        "folder": "Ask IISc",
                        "type": "file",
                        "options": {"path": str(dashboards.resolve())},
                    }
                ],
            }
        )
    )
    return root


ALERT_RULES = Path(__file__).parent / "grafana" / "alerts.yaml"
ALERTING_DEFAULT: dict[str, Any] = {
    "enabled": False,
    "email": {
        "to": [],
        "from": "",
        "smtp_host": "",
        "smtp_user": "",
        "smtp_password": "",
    },
    "webhook": {"url": ""},
}


def alerting_config(path: Path) -> dict[str, Any]:
    """The private notification settings; created switched off."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600)
        path.write_text(json.dumps(ALERTING_DEFAULT, indent=1))
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else dict(ALERTING_DEFAULT)


def alerting_provisioning(root: Path, config: dict[str, Any]) -> dict[str, str]:
    """Writes the alert rules and, if enabled, the contact point.

    Args:
        root: Grafana's provisioning folder.
        config: private/app/alerting.json.

    Returns:
        Extra Grafana environment (SMTP settings when e-mail is used).
    """
    folder = root / "alerting"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ALERT_RULES, folder / "rules.yaml")
    contact = folder / "contact-points.yaml"
    contact.unlink(missing_ok=True)
    if not config.get("enabled"):
        return {}
    receivers: list[dict[str, Any]] = []
    env: dict[str, str] = {}
    email = config.get("email") or {}
    if email.get("to") and email.get("smtp_host"):
        receivers.append(
            {
                "uid": "askiisc-email",
                "type": "email",
                "settings": {"addresses": ";".join(email["to"])},
            }
        )
        env = {
            "GF_SMTP_ENABLED": "true",
            "GF_SMTP_HOST": str(email["smtp_host"]),
            "GF_SMTP_USER": str(email.get("smtp_user", "")),
            "GF_SMTP_PASSWORD": str(email.get("smtp_password", "")),
            "GF_SMTP_FROM_ADDRESS": str(email.get("from", "")),
        }
    url = (config.get("webhook") or {}).get("url")
    if url:
        receivers.append(
            {
                "uid": "askiisc-webhook",
                "type": "webhook",
                "settings": {"url": url},
            }
        )
    if receivers:
        contact.write_text(
            json.dumps(
                {
                    "apiVersion": 1,
                    "contactPoints": [
                        {"orgId": 1, "name": "askiisc", "receivers": receivers}
                    ],
                    "policies": [
                        {
                            "orgId": 1,
                            "receiver": "askiisc",
                            "group_by": ["alertname"],
                        }
                    ],
                }
            )
        )
    return env


async def start_monitoring(
    children: Children,
    paths: Paths,
    ports: dict[str, int],
    access_port: int,
) -> dict[str, str]:
    """Starts Prometheus, Grafana and the Locust web UI if available."""
    links: dict[str, str] = {}
    env = dict(os.environ)
    prom = paths.services / "prometheus" / "prometheus"
    prom_password = ""
    if prom.exists():
        prom_password, web_config = prometheus_auth(paths)
        await children.start(
            "prometheus",
            [
                str(prom),
                f"--config.file={prometheus_config(paths, ports, access_port)}",
                f"--storage.tsdb.path={paths.data / 'prometheus'}",
                "--storage.tsdb.retention.time=30d",
                f"--web.listen-address=127.0.0.1:{ports['prometheus']}",
                f"--web.config.file={web_config}",
            ],
            paths.run / "prometheus.log",
            env,
        )
        links["prometheus"] = f"http://localhost:{ports['prometheus']}"
    loki = await start_logs(children, paths)
    grafana = paths.services / "grafana"
    if (grafana / "bin" / "grafana").exists():
        password = secret_file(paths.app / "grafana_admin_password")
        genv = {
            **env,
            "GF_SERVER_HTTP_ADDR": "127.0.0.1",
            "GF_SERVER_HTTP_PORT": str(ports["grafana"]),
            "GF_PATHS_DATA": str(paths.data / "grafana"),
            "GF_PATHS_LOGS": str(paths.run / "grafana-logs"),
            "GF_PATHS_PROVISIONING": str(
                grafana_provisioning(paths, ports, prom_password, loki)
            ),
            "GF_SECURITY_ADMIN_USER": "admin",
            "GF_SECURITY_ADMIN_PASSWORD": password,
            "GF_USERS_ALLOW_SIGN_UP": "false",
            "GF_AUTH_ANONYMOUS_ENABLED": "false",
            # The admin page's Grafana tab shows the dashboard in a frame;
            # Grafana's own sign-in still applies (SameSite=Lax cookie:
            # both are on localhost, so the frame is same-site).
            "GF_SECURITY_ALLOW_EMBEDDING": "true",
            "GF_SECURITY_COOKIE_SAMESITE": "lax",
            "GF_ANALYTICS_REPORTING_ENABLED": "false",
            "GF_ANALYTICS_CHECK_FOR_UPDATES": "false",
            "GF_NEWS_NEWS_FEED_ENABLED": "false",
            **alerting_provisioning(
                paths.run / "grafana-provisioning",
                alerting_config(paths.app / "alerting.json"),
            ),
        }
        await children.start(
            "grafana",
            [
                str(grafana / "bin" / "grafana"),
                "server",
                f"--homepath={grafana}",
            ],
            paths.run / "grafana.log",
            genv,
        )
        links["grafana"] = f"http://localhost:{ports['grafana']}"
    locust = Path(sys.executable).parent / "locust"
    if locust.exists():
        lenv = {
            **env,
            "ASKIISC_LOCUST_ACCOUNTS": str(paths.app / "locust_accounts.json"),
            "ASKIISC_PROM_TARGETS": str(paths.targets),
            "ASKIISC_LOCUST_RESULTS": str(paths.data / "results" / "locust"),
            "ASKIISC_LOCUST_UI_USER": LOCUST_UI_USER,
            "ASKIISC_LOCUST_UI_PASSWORD_FILE": str(
                paths.app / "locust_ui_password"
            ),
        }
        secret_file(paths.app / "locust_ui_password")
        await children.start(
            "locust",
            [
                str(locust),
                "-f",
                "eval/locust/locustfile.py",
                "--host",
                f"http://127.0.0.1:{ports['web']}",
                "--web-host",
                "127.0.0.1",
                "--web-port",
                str(ports["locust"]),
                "--web-login",
            ],
            paths.run / "locust.log",
            lenv,
        )
        links["locust"] = f"http://localhost:{ports['locust']}"
    return links


def write_targets(paths: Paths, pool: ProcessPool) -> None:
    """Lists the vLLM workers for Prometheus file discovery."""
    targets = [
        {
            "targets": [urlparse(w.url).netloc],
            "labels": {"worker": w.worker_id, "gpu": str(w.gpu)},
        }
        for w in pool.list_workers()
        if w.url
    ]
    paths.targets.mkdir(parents=True, exist_ok=True)
    tmp = paths.targets / "vllm.json.tmp"
    tmp.write_text(json.dumps(targets))
    tmp.replace(paths.targets / "vllm.json")


def load_test_runs(results: Path) -> list[dict[str, Any]]:
    """The summary.json files in a results folder, newest first.

    Used for Locust runs (slurm/loadtest.sh, the Locust UI) and for the
    other measurements the scorecard reads (results/measurements).
    """
    runs = []
    for summary in sorted(results.glob("*/summary.json"), reverse=True)[:100]:
        with contextlib.suppress(OSError, ValueError):
            runs.append(
                {
                    **json.loads(summary.read_text()),
                    "folder": str(summary.parent),
                }
            )
    return runs


@contextlib.asynccontextmanager
async def access_process(
    children: Children, paths: Paths, issuer: TokenIssuer, directory: str
) -> AsyncIterator[tuple[str, int]]:
    """Runs the access service as its own process."""
    jwks = paths.run / "jwks.json"
    jwks.write_text(json.dumps(issuer.jwks()))
    port = free_port()
    cmd = [
        sys.executable,
        "-m",
        "askiisc.access.server",
        "--port",
        str(port),
        "--jwks",
        str(jwks),
        "--corpus",
        str(paths.corpus),
        "--overrides",
        str(paths.app / "policy_overrides.jsonl"),
        "--directory",
        directory,
    ]
    await children.start(
        "access", cmd, paths.run / "access.log", dict(os.environ)
    )
    try:
        yield f"http://127.0.0.1:{port}", port
    finally:
        await children.stop("access")


def free_port() -> int:
    """Returns a free TCP port on 127.0.0.1."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@dataclass
class Live:
    """The running system, for the admin page and fault injection."""

    controller: Controller
    pool: ProcessPool
    index: DenseIndex
    model: str
    started: float
    links: dict[str, str]
    generation: list[int] = field(default_factory=lambda: [0])
    gpus: list[dict[str, Any]] = field(default_factory=list)
    stats: ServingStats = field(default_factory=ServingStats)
    vllm_version: str = ""

    def system(self) -> dict[str, Any]:
        """Facts shown on the admin overview."""
        return {
            "model": self.model,
            "vllm_version": self.vllm_version,
            "workers": [
                {"worker_id": w.worker_id, "gpu": w.gpu, "state": str(w.state)}
                for w in self.pool.list_workers()
            ],
            "queue": self.stats.summary(),
            "gpus": self.gpus,
            "chunks": len(self.index),
            "documents": self.index.documents,
            "started": self.started,
            "links": self.links,
        }


async def backups(live: Path, app: Path) -> None:
    """Copies the live databases to /data4 every BACKUP_INTERVAL_S."""
    while True:
        await asyncio.sleep(BACKUP_INTERVAL_S)
        try:
            await asyncio.to_thread(backup, live, app)
        except (OSError, sqlite3.Error) as err:
            LOG.warning("database backup failed: %s", err)


def people_tree(directory_path: str, directory: Directory) -> PeopleTree | None:
    """The editable users tree, if the directory is a folder tree.

    Args:
        directory_path: ASKIISC_DIRECTORY_PATH ("" is the sample tree,
            which stays read-only).
        directory: The web app's directory, reloaded after each edit.
    """
    root = Path(directory_path) if directory_path else None
    if root is None or not root.is_dir():
        return None
    return PeopleTree(root, directory)


WORKER_STATS_TIMEOUT_S = 2.0


async def read_workers(live: Live, http: httpx.AsyncClient, now: float) -> None:
    """Reads the queue, KV cache and token counter of every ready worker.

    A worker that does not answer is left out of this round.
    """
    readings = {}
    for worker in live.pool.list_workers():
        if str(worker.state) != "ready" or not worker.url:
            continue
        try:
            response = await http.get(f"{worker.url}/metrics")
            readings[worker.worker_id] = parse_metrics(response.text)
            if not live.vllm_version:
                version = await http.get(f"{worker.url}/version")
                live.vllm_version = str(version.json().get("version", ""))
        except (httpx.HTTPError, ValueError) as err:
            LOG.debug("worker %s stats: %s", worker.worker_id, err)
    live.stats.update(readings, now)


async def background(
    live: Live, metrics: GatewayMetrics, web: WebDeps, paths: Paths
) -> None:
    """Every few seconds: gauges, GPU stats and Prometheus targets.

    Also reloads the users tree after edits made from the terminal.
    """
    import pynvml  # noqa: PLC0415

    from askiisc.observability.gpu import read_gpus  # noqa: PLC0415

    watcher = (
        DirectoryWatcher(web.directory, web.people.root) if web.people else None
    )
    http = httpx.AsyncClient(timeout=WORKER_STATS_TIMEOUT_S)
    while True:
        with contextlib.suppress(Exception):
            if watcher is not None:
                await asyncio.to_thread(watcher.check)
            states: dict[str, int] = {"starting": 0, "loading": 0, "ready": 0}
            for w in live.pool.list_workers():
                states[str(w.state)] = states.get(str(w.state), 0) + 1
            for state, count in states.items():
                metrics.workers.labels(state).set(count)
            metrics.worker_failures.set(live.controller.failures)
            ours = {w.gpu for w in live.pool.list_workers()}
            gpus = []
            for gpu in read_gpus(pynvml):
                metrics.gpu_utilization.labels(str(gpu.index)).set(
                    gpu.utilization / 100
                )
                used = (gpu.total_mib - gpu.free_mib) * 2**20
                metrics.gpu_memory.labels(str(gpu.index)).set(used)
                gpus.append(
                    {
                        "index": gpu.index,
                        "utilization": gpu.utilization,
                        "memory_used_mib": gpu.total_mib - gpu.free_mib,
                        "memory_total_mib": gpu.total_mib,
                        "ours": gpu.index in ours,
                    }
                )
            live.gpus = gpus
            await read_workers(live, http, web.clock.now())
            write_targets(paths, live.pool)
            now = web.clock.now()
            web.m.sessions.set(
                await asyncio.to_thread(web.store.live_sessions, now)
            )
            accounts = await asyncio.to_thread(web.store.accounts)
            for status in ("pending", "active", "disabled"):
                for kind, synthetic in (("real", False), ("locust", True)):
                    web.m.accounts.labels(status, kind).set(
                        sum(
                            1
                            for a in accounts
                            if a.status == status and a.synthetic == synthetic
                        )
                    )
        await asyncio.sleep(LOOP_S)


def passage_text(
    text_of: Callable[[str], str | None], contexts: Mapping[str, str]
) -> Callable[[str], str | None]:
    """A cited chunk's own text, without the context line ingest put first.

    The line ("title | edition | site") is not on the page, so a link that
    highlights it would highlight nothing.
    """

    def text(chunk_id: str) -> str | None:
        found = text_of(chunk_id)
        context = contexts.get(chunk_id.split("@", 1)[0])
        if found and context and found.startswith(context + "\n"):
            return found[len(context) + 1 :]
        return found

    return text


def check_dimension(index: int, vectors: int, state: Path, model: str) -> None:
    """Stops a job whose embedder cannot search its index.

    Args:
        index: The index's vector dimension (0: empty index).
        vectors: The dimension the embedding model returns.
        state: The index folder, for the message.
        model: The embedding model, for the message.

    Raises:
        ConfigError: If they differ.
    """
    if index and index != vectors:
        raise ConfigError(
            f"{state} holds {index}-dimension vectors but {model} gives "
            f"{vectors}: serve the index built with this model "
            "(ASKIISC_STATE_DIR) or set ASKIISC_EMBED_MODEL to its model"
        )


def mailer(app_dir: Path) -> SmtpMailer | None:
    """The mailer for password links; None (with a warning) if not usable."""
    try:
        found = load_smtp(app_dir / "smtp.json")
    except ConfigError as err:
        LOG.warning("e-mail is off: %s", err)
        return None
    if found is None:
        LOG.warning("e-mail is not set up: admins pass password links on")
    return found


def first_allowlist(store: Any, directory: Any, now: float) -> int:
    """Starts an empty allowlist with the real people of the directory.

    Runs once, on the first start after the allowlist was added (schema 2),
    so people who could sign in before keep their access; afterwards only
    admins allow anyone.

    Returns:
        How many were added (0 once the allowlist has any entry).
    """
    if store.allowed_users():
        return 0
    emails = [p.user_id for p in directory.people() if "@" in p.user_id]
    return int(store.allow_existing(emails, "migration (schema 2)", now))


UPLOADS_FILE = "uploads.json"
UPLOAD_KINDS = {
    "application/pdf": "pdf",
    "text/html": "html",
    "text/plain": "txt",
}


def load_uploads(state: Path) -> dict[str, dict[str, str]]:
    """Titles of documents uploaded from the dashboard, by id.

    Uploaded documents are not in the crawled corpus, so without this
    file they would leave the Documents list (and cite their id) after a
    restart. A missing or unreadable file means none.
    """
    try:
        data = json.loads((state / UPLOADS_FILE).read_text())
    except (OSError, ValueError):
        return {}
    return {str(k): dict(v) for k, v in data.items() if isinstance(v, dict)}


def save_upload(state: Path, doc_id: str, info: dict[str, str]) -> None:
    """Records an uploaded document's title (atomically, mode 600)."""
    uploads = load_uploads(state)
    uploads[doc_id] = info
    tmp = state / f".{UPLOADS_FILE}.tmp"
    tmp.write_text(json.dumps(uploads, indent=1, sort_keys=True))
    tmp.chmod(0o600)
    tmp.replace(state / UPLOADS_FILE)


def make_publisher(
    ingestor: Ingestor,
    live: Live,
    clock: SystemClock,
    catalogue: dict[str, dict[str, str]],
    state: Path,
) -> Any:
    """Publishes a new version of a document from file bytes (FR-5).

    The ingest pipeline extracts, chunks, embeds and writes it to Milvus
    and the in-memory index; the previous version is retired. The title
    is kept in `state/uploads.json` so it survives a restart.
    """

    async def publish(
        doc_id: str, data: bytes, content_type: str, title: str
    ) -> dict[str, Any]:
        doc = SourceDocument(DocId(doc_id), content_type)
        fingerprint = hashlib.sha256(data).hexdigest()
        result = await ingestor.ingest(doc, data, fingerprint, clock.now())
        live.generation[0] += 1  # answer cache entries are now stale
        answer = {
            "doc_id": result.doc_id,
            "version": int(result.version),
            "chunks": result.chunks,
            "visible_from": result.visible_from,
            "retired": result.retired,
        }
        if title:
            info = {
                "title": title,
                "url": "",
                "kind": UPLOAD_KINDS.get(content_type, ""),
                "site": "uploaded",
            }
            catalogue[doc_id] = info
            await asyncio.to_thread(save_upload, state, doc_id, info)
        return answer

    return publish


def version_history(versions: SQLiteVersionStore) -> Any:
    """A document's versions for the admin page (no text)."""

    def history(doc_id: str) -> list[dict[str, Any]]:
        return [
            {
                "version": int(r.version),
                "valid_from": r.valid_from,
                "valid_to": r.valid_to,
                "chunks": len(r.drafts),
            }
            for r in versions.history(DocId(doc_id))
        ]

    return history


def make_chaos(live: Live, restart_access: Any) -> dict[str, Any]:
    """Fault injections for load tests (ASKIISC_CHAOS=1 only)."""

    async def kill_worker() -> str:
        ready = live.controller.ready_workers()
        if not ready:
            return "no ready worker"
        live.pool.kill(ready[0].worker_id)
        return f"killed {ready[0].worker_id}"

    return {"kill_worker": kill_worker, "restart_access": restart_access}


async def run(paths: Paths) -> int:
    """Serves until SIGTERM, then shuts everything down."""
    settings = Settings()
    clock = SystemClock()
    ports = choose_ports()
    for folder in (paths.app, paths.targets):
        folder.mkdir(parents=True, exist_ok=True)
    paths.app.chmod(0o700)
    issuer = TokenIssuer()
    directory, policies, docs = load_world(
        settings.directory_path, paths.corpus
    )
    del policies  # the access process loads its own copy
    children = Children()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        loop.add_signal_handler(sig, stop.set)
    try:
        async with (
            httpx.AsyncClient(limits=httpx.Limits(max_connections=512)) as http,
            embed_server(paths.run, free_port()) as embed_url,
            access_process(
                children,
                paths,
                issuer,
                settings.directory_path or "eval/datasets/sample/users",
            ) as (access_url, access_port),
        ):
            await wait_http(http, embed_url + "/health", 300)
            await wait_http(http, access_url + "/healthz", 300)
            return await _serve(
                settings,
                clock,
                paths,
                ports,
                http,
                issuer,
                directory,
                docs,
                embed_url,
                access_url,
                access_port,
                children,
                stop,
            )
    finally:
        await children.stop_all()


async def _serve(  # noqa: PLR0915 - one wiring function, read top to bottom
    settings: Settings,
    clock: SystemClock,
    paths: Paths,
    ports: dict[str, int],
    http: httpx.AsyncClient,
    issuer: TokenIssuer,
    directory: Any,
    docs: list[CorpusDoc],
    embed_url: str,
    access_url: str,
    access_port: int,
    children: Children,
    stop: asyncio.Event,
) -> int:
    from pymilvus import MilvusClient  # noqa: PLC0415

    embedder = HttpEmbedder(http, embed_url, timeout_s=30.0)
    search_pool = ThreadPoolExecutor(SEARCH_THREADS, "search")
    versions = SQLiteVersionStore(paths.state / "versions.db")
    client = MilvusClient(str(paths.state / "milvus.db"))
    index = await asyncio.to_thread(
        open_dense, paths.state, MilvusIndex(client), client, versions.stamp()
    )
    probe = await embedder.embed(["dimension check"])
    dim = index.dim if len(index) else 0
    check_dimension(dim, len(probe[0]), paths.state, settings.embed_model)
    ingestor = Ingestor(
        versions,
        FileObjectStore(paths.state / "objects"),
        embedder,
        index,
        commit_delay_s=REVISION_COMMIT_DELAY_S,
    )
    access = HttpAccessClient(
        http, access_url, clock, timeout_s=settings.access_timeout_s
    )
    slots = settings.llm_concurrency
    dispatcher = FairDispatcher(capacity=slots, quantum=2048)
    pool = make_pool(settings, paths.run, http)
    controller = make_controller(
        pool,
        ScalingConfig(
            max_workers=settings.max_workers,
            idle_timeout_s=settings.idle_timeout_s,
        ),
        clock,
        dispatcher,
        slots,
    )
    metrics = GatewayMetrics()
    spec = select_model(settings.model)
    llm = LLMClient(
        http,
        controller.endpoint,
        spec.served_name,
        timeout_s=settings.llm_timeout_s,
        extra_body=spec.request_extra(),
    )
    links = await start_monitoring(children, paths, ports, access_port)
    live = Live(controller, pool, index, spec.served_name, clock.now(), links)
    databases = db_dir(paths.app)
    restored = await asyncio.to_thread(restore, paths.app, databases)
    LOG.info("databases in %s (restored: %s)", databases, restored or "none")
    requests = RequestLog(
        databases / "requests.db",
        role_of=lambda user: p.role if (p := directory.get(user)) else "",
        now=clock.now(),
        lines=paths.run / "requests.log",
    )
    requests.start()
    salt = secret_file(paths.app / "audit_salt")
    catalogue = doc_catalogue(paths.corpus, docs)
    catalogue.update(load_uploads(paths.state))
    reranker = (
        HttpReranker(http, embed_url, timeout_s=10.0)
        if settings.rerank_model
        else None
    )
    deps = GatewayDeps(
        clock=clock,
        auth=Authenticator(issuer.verifier(clock), directory),
        access=access,
        rag=RagAnswerer(
            Retriever(
                access,
                embedder,
                index,
                clock,
                observe=lambda stage, s: metrics.stage.labels(stage).observe(s),
                query_prefix=query_prefix(settings.embed_model),
                executor=search_pool,
                reranker=reranker,
                candidates=settings.rerank_candidates,
            ),
            llm,
            clock,
            labels=catalogue,
        ),
        admission=Admission(
            clock,
            settings.tenant_rate_per_s,
            settings.tenant_burst,
            settings.tenant_queue_max,
            settings.max_cold_hold_s + 60,
            concurrency=slots,
            max_in_flight=settings.max_in_flight,
        ),
        dispatcher=dispatcher,
        cache=InMemoryAnswerCache(clock),
        audit=FileAuditSink(paths.run / "audit.jsonl"),
        audit_salt=salt,
        metrics=metrics,
        workers=controller,
        generation=lambda: live.generation[0],
        max_cold_hold_s=settings.max_cold_hold_s,
        on_request=requests.record,
    )
    publish = make_publisher(ingestor, live, clock, catalogue, paths.state)

    async def restart_access() -> str:
        await children.stop("access")
        await asyncio.sleep(10)
        await children.start(
            "access",
            [
                sys.executable,
                "-m",
                "askiisc.access.server",
                "--port",
                str(access_port),
                "--jwks",
                str(paths.run / "jwks.json"),
                "--corpus",
                str(paths.corpus),
                "--overrides",
                str(paths.app / "policy_overrides.jsonl"),
                "--directory",
                settings.directory_path or "eval/datasets/sample/users",
            ],
            paths.run / "access.log",
            dict(os.environ),
        )
        return "access service down for 10 s, restarted"

    web = WebDeps(
        gateway=deps,
        store=AppStore(databases / "app.db", load_key(paths.app / "chat.key")),
        requests=requests,
        directory=directory,
        tokens=InternalTokens(issuer, clock),
        clock=clock,
        docs=lambda d: catalogue.get(d, {"title": d, "url": ""}),
        chunk_text=passage_text(
            index.text, {str(d.policy.doc_id): d.context for d in docs}
        ),
        access_http=httpx.AsyncClient(base_url=access_url),
        system=live.system,
        revise=lambda doc_id, text: publish(
            doc_id, text.encode(), "text/plain", ""
        ),
        publish=publish,
        history=version_history(versions),
        find_docs=lambda q, n, s: find_documents(catalogue, q, n, s),
        chaos=make_chaos(live, restart_access)
        if os.environ.get("ASKIISC_CHAOS") == "1"
        else {},
        loadtests=lambda: load_test_runs(paths.data / "results" / "locust"),
        measurements=lambda: load_test_runs(
            paths.data / "results" / "measurements"
        ),
        people=people_tree(settings.directory_path, directory),
        mailer=mailer(paths.app),
        email_domains=tuple(
            d.strip().lower()
            for d in settings.email_domains.split(",")
            if d.strip()
        ),
        test_logins=settings.test_logins,
        secure_cookies=settings.secure_cookies,
        models_dir=paths.data / "results" / "models",
        active_model=settings.model,
        serve_job=os.environ.get("SLURM_JOB_ID", ""),
    )
    if activated := web.store.activate_pending():
        LOG.info("%d pending accounts activated (no approval step)", activated)
    if added := first_allowlist(web.store, directory, clock.now()):
        LOG.info("allowlist started with %d people of the directory", added)
    app = create_gateway(deps)
    mount_web(app, web)
    tasks = [
        asyncio.create_task(controller.run(interval_s=1.0)),
        asyncio.create_task(background(live, metrics, web, paths)),
        asyncio.create_task(backups(databases, paths.app)),
    ]
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=ports["web"],
        log_level="warning",
        lifespan="off",
        timeout_keep_alive=30,
    )
    server = uvicorn.Server(config)
    serving = asyncio.create_task(server.serve())
    write_current(paths, ports, links)
    LOG.info("serving on http://127.0.0.1:%d", ports["web"])
    try:
        stopping = asyncio.create_task(stop.wait())
        await asyncio.wait(
            {serving, stopping}, return_when=asyncio.FIRST_COMPLETED
        )
        LOG.info("stopping")
    finally:
        # First of all: Slurm kills the job soon after scancel, and the web
        # server's graceful stop waits for open streams. The live copy
        # stays on local disk either way.
        await asyncio.to_thread(requests.flush)
        await asyncio.to_thread(backup, databases, paths.app)
        server.should_exit = True
        await serving
        for task in tasks:
            task.cancel()
        await controller.close()
        write_targets(paths, pool)
        requests.close()
        await web.access_http.aclose()
        search_pool.shutdown(wait=False, cancel_futures=True)
        if index.dirty:
            await asyncio.to_thread(
                index.save, paths.state / "dense", versions.stamp()
            )
        (paths.data / "run" / "current.json").unlink(missing_ok=True)
    return 0


def write_current(
    paths: Paths, ports: dict[str, int], links: dict[str, str]
) -> None:
    """Records the running instance for slurm/loadtest.sh and people."""
    info = {
        "job": os.environ.get("SLURM_JOB_ID", ""),
        "host": socket.gethostname(),
        "ports": ports,
        "links": {"chat": f"http://localhost:{ports['web']}", **links},
        "run_dir": str(paths.run),
        "started": time.time(),
    }
    (paths.data / "run" / "current.json").write_text(json.dumps(info, indent=1))
    tunnel = " ".join(f"-L {p}:localhost:{p}" for p in ports.values())
    LOG.info("tunnel from your laptop: ssh -N %s <you>@<server>", tunnel)


def main() -> None:  # pragma: no cover
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s"
    )
    data = Path(os.environ.get("ASKIISC_DATA", "/data4/muthangiv/askiisc"))
    paths = Paths(args.run_dir, args.corpus, args.state_dir, data)
    sys.exit(asyncio.run(run(paths)))


if __name__ == "__main__":  # pragma: no cover
    main()
