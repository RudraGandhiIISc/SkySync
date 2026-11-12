"""The one typed settings object, read from ASKIISC_* environment variables."""

from __future__ import annotations

import getpass
import os
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from askiisc.common import targets


def data_dir() -> Path:
    """Run state, results and private files: $ASKIISC_DATA.

    Without it, /data4/<you>/askiisc, the default slurm/lib.sh sets, so a
    command run outside a job uses the caller's own folder, never another
    user's.
    """
    configured = os.environ.get("ASKIISC_DATA", "")
    if configured:
        return Path(configured)
    return Path("/data4") / getpass.getuser() / "askiisc"


class Settings(BaseSettings):
    """Runtime configuration. Every field maps to ASKIISC_<NAME>."""

    model_config = SettingsConfigDict(env_prefix="ASKIISC_", extra="ignore")

    # Model and serving (FR-8).
    model: str = "qwen3-4b-instruct-2507"
    llm_timeout_s: float = 60.0

    # Services (their addresses are chosen at start-up: deploy/serve.py).
    access_timeout_s: float = Field(
        default=targets.ACCESS_CLIENT_TIMEOUT_S, gt=0
    )

    # Identity (A4).
    jwks_path: str = ""
    token_issuer: str = "iisc-sso"  # noqa: S105 - not a secret
    token_audience: str = "ask-iisc"  # noqa: S105 - not a secret
    directory_path: str = ""

    # Password sign-in for Locust's test identities (ids without "@").
    test_logins: bool = True
    # Send the session cookie only over HTTPS (off for the SSH tunnel's
    # http://localhost; on when the app is served over HTTPS).
    secure_cookies: bool = False
    # E-mail domains people may be added with on the dashboard (comma list).
    email_domains: str = "iisc.ac.in"

    # Privacy (NFR-9). Required in deployment; no default secret.
    audit_salt: SecretStr = SecretStr("")

    # Load limits (FR-6).
    tenant_rate_per_s: float = Field(default=2.0, gt=0)
    tenant_burst: float = Field(default=10.0, gt=0)
    tenant_queue_max: int = Field(default=32, gt=0)
    tenant_queue_max_wait_s: float = Field(default=3.0, gt=0)
    llm_concurrency: int = Field(default=64, gt=0)
    # Requests the gateway process handles at once before answering 429
    # (FR-6). One Python process serves about 11 requests/s (2026-10-26);
    # 24 in flight keeps p95 near 3 s. Shared fairly between departments.
    max_in_flight: int = Field(default=24, gt=0)
    # Admission control on or off (ASKIISC_ADMISSION=on|off). Off admits
    # every request, which then queues without bound; it exists only for
    # the E4 arm that shows the collapse.
    admission: bool = True
    # Fair queueing between departments for LLM slots (FR-11;
    # ASKIISC_FAIR_QUEUE=off gives one first-come queue, the E4 arm).
    fair_queue: bool = True
    # The answer cache (ASKIISC_ANSWER_CACHE=off sends every request to
    # vLLM, for the FR-7 prefix-cache experiment).
    answer_cache: bool = True
    # vLLM prefix caching (ASKIISC_PREFIX_CACHE=off for the FR-7 ablation).
    prefix_cache: bool = True
    # Sample the event loop's Python stacks to profile.folded (FR-6
    # diagnosis; ASKIISC_PROFILE=1). Off in normal serving.
    profile: bool = False

    # Retrieval.
    top_k: int = targets.PROMPT_TOP_K
    embed_model: str = targets.EMBED_MODEL
    # Empty: no reranking (hybrid search order is kept).
    rerank_model: str = targets.RERANK_MODEL
    rerank_candidates: int = Field(default=targets.RERANK_CANDIDATES, gt=0)

    # Autoscaler (FR-9, FR-10).
    max_workers: int = Field(
        default=targets.MAX_GPU_WORKERS, ge=0, le=targets.MAX_GPU_WORKERS
    )
    idle_timeout_s: float = Field(default=300.0, ge=0)
    max_cold_hold_s: float = targets.NFR2_COLD_TTFT_8B_S
