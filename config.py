"""Environment-driven configuration for the portable GPU worker.

The same source code can run on a local GPU machine or on AWS. Only the
environment variables and available model files need to differ.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"{name} must be one of true/false, yes/no, on/off, 1/0."
    )


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return value


def _env_float(
    name: str,
    default: float,
    *,
    minimum: float = 0.0,
) -> float:
    raw = os.getenv(name)
    value = default if raw is None else float(raw)
    if value <= minimum:
        raise ValueError(f"{name} must be > {minimum}.")
    return value


def _optional_path(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None

    cleaned = raw.strip()
    return cleaned or None


@dataclass(frozen=True, slots=True)
class GPUWorkerSettings:
    host: str
    port: int
    api_key: str | None

    queue_maxsize: int
    embed_timeout_seconds: float
    decompose_timeout_seconds: float
    generate_timeout_seconds: float

    load_embedding: bool
    load_small_llm: bool
    load_response_llm: bool

    bge_model_path: str | None
    small_llm_model_path: str | None
    response_llm_model_path: str | None

    bge_batch_size: int
    bge_max_seq_length: int

    decompose_max_new_tokens: int
    decision_max_output_tokens: int
    response_max_output_tokens: int

    small_llm_enable_thinking: bool
    response_llm_enable_thinking: bool

    trust_remote_code: bool
    device_map: str

    @property
    def profile_configuration(self) -> dict[str, bool]:
        """Profiles configured to be loaded on this machine."""
        return {
            "embedding": self.load_embedding,
            "decomposer": self.load_small_llm,
            "decision": self.load_small_llm,
            "response": self.load_response_llm,
        }

    def validate(self) -> None:
        if not self.host.strip():
            raise ValueError("GPU_SERVICE_HOST cannot be empty.")

        if self.load_embedding and not self.bge_model_path:
            raise ValueError(
                "BGE_MODEL_PATH is required when GPU_LOAD_EMBEDDING=true."
            )

        if self.load_small_llm and not self.small_llm_model_path:
            raise ValueError(
                "SMALL_LLM_MODEL_PATH is required when "
                "GPU_LOAD_SMALL_LLM=true."
            )

        if self.load_response_llm and not self.response_llm_model_path:
            raise ValueError(
                "RESPONSE_LLM_MODEL_PATH is required when "
                "GPU_LOAD_RESPONSE_LLM=true."
            )


def load_settings(
    env_file: str | Path | None = None,
) -> GPUWorkerSettings:
    """Load worker settings.

    Precedence:
        1. Explicit ``env_file`` argument
        2. ``GPU_WORKER_ENV_FILE``
        3. ``gpu_worker/.env`` if present
        4. Existing process environment
    """
    resolved_env_file: Path | None = None

    if env_file is not None:
        resolved_env_file = Path(env_file)
    else:
        configured = os.getenv("GPU_WORKER_ENV_FILE")
        if configured:
            resolved_env_file = Path(configured)
        else:
            candidate = Path(__file__).resolve().parent / ".env"
            if candidate.exists():
                resolved_env_file = candidate

    if resolved_env_file is not None and resolved_env_file.exists():
        load_dotenv(resolved_env_file, override=False)

    api_key_raw = os.getenv("GPU_SERVICE_API_KEY")
    api_key = api_key_raw.strip() if api_key_raw else None

    settings = GPUWorkerSettings(
        host=os.getenv("GPU_SERVICE_HOST", "127.0.0.1").strip(),
        port=_env_int("GPU_SERVICE_PORT", 8001),
        api_key=api_key,
        queue_maxsize=_env_int("GPU_QUEUE_MAXSIZE", 32),
        embed_timeout_seconds=_env_float(
            "GPU_EMBED_TIMEOUT_SECONDS",
            90.0,
        ),
        decompose_timeout_seconds=_env_float(
            "GPU_DECOMPOSE_TIMEOUT_SECONDS",
            120.0,
        ),
        generate_timeout_seconds=_env_float(
            "GPU_GENERATE_TIMEOUT_SECONDS",
            600.0,
        ),
        load_embedding=_env_bool(
            "GPU_LOAD_EMBEDDING",
            True,
        ),
        load_small_llm=_env_bool(
            "GPU_LOAD_SMALL_LLM",
            True,
        ),
        load_response_llm=_env_bool(
            "GPU_LOAD_RESPONSE_LLM",
            True,
        ),
        bge_model_path=_optional_path("BGE_MODEL_PATH"),
        small_llm_model_path=_optional_path("SMALL_LLM_MODEL_PATH"),
        response_llm_model_path=_optional_path(
            "RESPONSE_LLM_MODEL_PATH"
        ),
        bge_batch_size=_env_int("BGE_BATCH_SIZE", 16),
        bge_max_seq_length=_env_int(
            "BGE_MAX_SEQ_LENGTH",
            8192,
        ),
        decompose_max_new_tokens=_env_int(
            "DECOMPOSE_MAX_NEW_TOKENS",
            128,
        ),
        decision_max_output_tokens=_env_int(
            "DECISION_MAX_OUTPUT_TOKENS",
            512,
        ),
        response_max_output_tokens=_env_int(
            "RESPONSE_MAX_OUTPUT_TOKENS",
            2048,
        ),
        small_llm_enable_thinking=_env_bool(
            "SMALL_LLM_ENABLE_THINKING",
            False,
        ),
        response_llm_enable_thinking=_env_bool(
            "RESPONSE_LLM_ENABLE_THINKING",
            False,
        ),
        trust_remote_code=_env_bool(
            "GPU_TRUST_REMOTE_CODE",
            True,
        ),
        device_map=os.getenv("GPU_DEVICE_MAP", "auto").strip() or "auto",
    )

    settings.validate()
    return settings
