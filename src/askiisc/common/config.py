"""The one typed settings object, read from ASKIISC_* environment variables."""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from askiisc.common import targets


class Settings(BaseSettings):
    """Runtime configuration. Every field maps to ASKIISC_<NAME>."""

    model_config = SettingsConfigDict(env_prefix="ASKIISC_", extra="ignore")

    # Model and serving (FR-8).
    model: str = "qwen3-4b-instruct-2507"
    llm_base_url: str = "http://127.0.0.1:8001"
    llm_timeout_s: float = 60.0

    # Services.
    access_url: str = "http://127.0.0.1:8002"
    embed_url: str = "http://127.0.0.1:8003"
    milvus_uri: str = "/data4/muthangiv/askiisc/state/milvus.db"
    redis_url: str = "redis://127.0.0.1:6379/0"
    audit_path: str = "/data4/muthangiv/askiisc/state/audit.jsonl"

    # Identity (A4).
    jwks_path: str = ""
    token_issuer: str = "iisc-sso"  # noqa: S105 - not a secret
    token_audience: str = "ask-iisc"  # noqa: S105 - not a secret
    directory_path: str = ""

    # Privacy (NFR-9). Required in deployment; no default secret.
    audit_salt: SecretStr = SecretStr("")

    # Load limits (FR-6).
    tenant_rate_per_s: float = Field(default=2.0, gt=0)
    tenant_burst: float = Field(default=10.0, gt=0)
    tenant_queue_max: int = Field(default=32, gt=0)
    tenant_queue_max_wait_s: float = Field(default=3.0, gt=0)
    llm_concurrency: int = Field(default=64, gt=0)

    # Retrieval.
    top_k: int = targets.RETRIEVAL_TOP_K

    # Autoscaler (FR-9, FR-10).
    max_workers: int = Field(
        default=targets.MAX_GPU_WORKERS, ge=0, le=targets.MAX_GPU_WORKERS
    )
    idle_timeout_s: float = Field(default=300.0, ge=0)
    max_cold_hold_s: float = targets.NFR2_COLD_TTFT_8B_S
