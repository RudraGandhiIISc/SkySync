"""The model registry (FR-8). Switching models is a config change only."""

from __future__ import annotations

from importlib import resources
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from askiisc.common.errors import ConfigError

SizeClass = Literal["4B", "8B", "8B-AWQ"]


class ModelSpec(BaseModel):
    """One servable model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    hf_repo: str
    served_name: str
    family: str
    size_class: SizeClass
    dtype: str
    quantization: str | None = None
    gated: bool = False
    # Sent with every chat request; Qwen3 (hybrid reasoning) needs
    # {"enable_thinking": false} to answer directly.
    chat_template_kwargs: dict[str, bool] | None = None

    def request_extra(self) -> dict[str, object]:
        """Fields added to every chat request for this model."""
        if not self.chat_template_kwargs:
            return {}
        return {"chat_template_kwargs": dict(self.chat_template_kwargs)}

    def vllm_args(self, local_dir: str | None = None) -> list[str]:
        """Builds the `vllm serve` arguments for this model.

        Args:
            local_dir: Checkpoint directory for local (AWQ) models.

        Returns:
            The model path followed by flags.

        Raises:
            ConfigError: If a local model has no directory.
        """
        path = self.hf_repo
        if path.startswith("local:"):
            if local_dir is None:
                raise ConfigError(f"{self.key} needs a local checkpoint")
            path = local_dir
        args = [
            path,
            "--served-model-name",
            self.served_name,
            "--dtype",
            self.dtype,
        ]
        if self.quantization:
            args += ["--quantization", self.quantization]
        return args


def load_models(text: str | None = None) -> dict[str, ModelSpec]:
    """Loads the registry.

    Args:
        text: YAML text; the packaged models.yaml by default.

    Returns:
        Models by key.

    Raises:
        ConfigError: If the YAML is malformed.
    """
    if text is None:
        text = (resources.files("askiisc.llm") / "models.yaml").read_text()
    try:
        raw = yaml.safe_load(text)
        return {k: ModelSpec(key=k, **v) for k, v in raw.items()}
    except (yaml.YAMLError, AttributeError, TypeError, ValueError) as err:
        raise ConfigError(f"bad model registry: {err}") from err


def select_model(
    key: str, models: dict[str, ModelSpec] | None = None
) -> ModelSpec:
    """Returns the model named by ASKIISC_MODEL.

    Raises:
        ConfigError: If the key is unknown.
    """
    registry = load_models() if models is None else models
    if key not in registry:
        raise ConfigError(f"unknown model {key}; known: {sorted(registry)}")
    return registry[key]
