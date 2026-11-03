"""Per-request logs: Loki (log store) and Grafana Alloy (log shipper).

Started by deploy/serve.py inside the serving job:

- **Loki** listens for HTTP on a Unix socket in the run folder (only this
  account can open it). Its internal gRPC port is on 127.0.0.1 and
  requires a client certificate (mutual TLS), so other users on the
  server cannot talk to it either.
- **Auth proxy** (askiisc.observability.authproxy) on 127.0.0.1 checks a
  user name and password, like Prometheus' basic auth, and forwards to the
  socket. Grafana and Alloy use it.
- **Alloy** tails the run folder's logs (requests.log, access, embed, vLLM,
  Locust) and pushes them through the proxy. Labels are only `job` and
  `slurm_job`; user and request ids stay inside the lines.

Storage is on /data4 (`loki/`, `alloy/`), kept 30 days.
"""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404 - only openssl, fixed arguments
from pathlib import Path

LOKI_USER = "loki"
RETENTION = "720h"
LOG_JOBS = {
    "requests": "requests.log",
    "access": "access.log",
    "embed": "embed.log",
    "vllm": "vllm-*.log",
    "locust": "locust.log",
}
# Every internal Loki gRPC client must present the client certificate.
GRPC_CLIENTS = (
    "ingester.client",
    "pattern-ingester.client",
    "querier.frontend-grpc-client",
    "querier.scheduler-grpc-client",
    "query-scheduler.grpc-client-config",
    "frontend.grpc-client-config",
    "compactor.grpc-client",
    "tsdb.shipper.index-gateway-client.grpc",
    "boltdb.shipper.index-gateway-client.grpc",
)


def make_certs(folder: Path) -> dict[str, Path]:
    """A private CA and one certificate for Loki's gRPC (made once).

    Raises:
        RuntimeError: If openssl is missing.
    """
    certs = {
        name: folder / f"{name}" for name in ("ca.crt", "tls.crt", "tls.key")
    }
    if all(p.exists() for p in certs.values()):
        return certs
    openssl = shutil.which("openssl")
    if openssl is None:
        raise RuntimeError("openssl not found: cannot protect Loki's gRPC")
    folder.mkdir(parents=True, exist_ok=True)
    folder.chmod(0o700)
    ext = folder / "ext.cnf"
    ext.write_text(
        "subjectAltName=DNS:localhost,IP:127.0.0.1\n"
        "extendedKeyUsage=serverAuth,clientAuth\n"
    )
    steps = [
        ["req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "3650",
         "-subj", "/CN=askiisc-loki-ca", "-keyout", "ca.key", "-out", "ca.crt"],
        ["req", "-newkey", "rsa:2048", "-nodes", "-subj", "/CN=localhost",
         "-keyout", "tls.key", "-out", "tls.csr"],
        ["x509", "-req", "-in", "tls.csr", "-CA", "ca.crt", "-CAkey", "ca.key",
         "-CAcreateserial", "-days", "3650", "-extfile", "ext.cnf",
         "-out", "tls.crt"],
    ]  # fmt: skip
    for step in steps:
        subprocess.run(  # noqa: S603 - fixed binary and arguments
            [openssl, *step], cwd=folder, check=True, capture_output=True
        )
    for path in folder.iterdir():
        path.chmod(0o600)
    return certs


def loki_config(
    run: Path, data: Path, grpc_port: int, certs: dict[str, Path]
) -> tuple[Path, Path]:
    """Writes loki.yaml; returns it and the HTTP socket Loki will create.

    Loki joins the socket address and port, so the socket file is named
    `loki.sock:0`.
    """
    socket_dir = run / "loki"
    socket_dir.mkdir(parents=True, exist_ok=True)
    socket_dir.chmod(0o700)
    config = {
        "auth_enabled": False,
        "server": {
            "http_listen_network": "unix",
            "http_listen_address": str(socket_dir / "loki.sock"),
            "http_listen_port": 0,
            "grpc_listen_address": "127.0.0.1",
            "grpc_listen_port": grpc_port,
            "grpc_tls_config": {
                "cert_file": str(certs["tls.crt"]),
                "key_file": str(certs["tls.key"]),
                "client_ca_file": str(certs["ca.crt"]),
                "client_auth_type": "RequireAndVerifyClientCert",
            },
            "log_level": "warn",
        },
        "common": {
            "instance_addr": "127.0.0.1",
            "path_prefix": str(data),
            "replication_factor": 1,
            "ring": {"kvstore": {"store": "inmemory"}},
            "storage": {
                "filesystem": {
                    "chunks_directory": str(data / "chunks"),
                    "rules_directory": str(data / "rules"),
                }
            },
        },
        "schema_config": {
            "configs": [
                {
                    "from": "2024-01-01",
                    "store": "tsdb",
                    "object_store": "filesystem",
                    "schema": "v13",
                    "index": {"prefix": "index_", "period": "24h"},
                }
            ]
        },
        "compactor": {
            "working_directory": str(data / "compactor"),
            "retention_enabled": True,
            "delete_request_store": "filesystem",
        },
        "limits_config": {"retention_period": RETENTION},
        "analytics": {"reporting_enabled": False},
    }
    path = run / "loki.yaml"
    path.write_text(json.dumps(config, indent=1))  # JSON is valid YAML
    return path, socket_dir / "loki.sock:0"


def loki_command(
    binary: Path, config: Path, certs: dict[str, Path]
) -> list[str]:
    """Loki's command line, with TLS on every internal gRPC client."""
    cmd = [str(binary), f"-config.file={config}"]
    for client in GRPC_CLIENTS:
        cmd += [
            f"-{client}.tls-enabled=true",
            f"-{client}.tls-cert-path={certs['tls.crt']}",
            f"-{client}.tls-key-path={certs['tls.key']}",
            f"-{client}.tls-ca-path={certs['ca.crt']}",
            f"-{client}.tls-server-name=localhost",
        ]
    return cmd


def alloy_config(
    run: Path, push_port: int, password_file: Path, slurm_job: str
) -> Path:
    """Writes Alloy's config: tail the run folder's logs, push to Loki.

    The password is read from its file by Alloy; it is never in the config.
    """
    targets = ",\n    ".join(
        f'{{"__path__" = "{run / pattern}", "job" = "{job}"}}'
        for job, pattern in LOG_JOBS.items()
    )
    text = f"""// Generated by deploy/logs.py for serving job {slurm_job}.
local.file_match "askiisc" {{
  path_targets = [
    {targets},
  ]
  sync_period = "10s"
}}

loki.source.file "askiisc" {{
  targets    = local.file_match.askiisc.targets
  forward_to = [loki.write.local.receiver]
}}

loki.write "local" {{
  endpoint {{
    url = "http://127.0.0.1:{push_port}/loki/api/v1/push"
    basic_auth {{
      username      = "{LOKI_USER}"
      password_file = "{password_file}"
    }}
  }}
  external_labels = {{"slurm_job" = "{slurm_job}"}}
}}
"""
    path = run / "alloy.river"
    path.write_text(text)
    return path


def alloy_command(
    binary: Path, config: Path, storage: Path, http_port: int
) -> list[str]:
    """Alloy's command line (its own UI bound to 127.0.0.1)."""
    return [
        str(binary),
        "run",
        f"--server.http.listen-addr=127.0.0.1:{http_port}",
        f"--storage.path={storage}",
        "--disable-reporting",
        str(config),
    ]
