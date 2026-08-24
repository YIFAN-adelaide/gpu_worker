"""Thin synchronous client for the local vLLM response server.

This adapter intentionally exposes the same generation contract as ChatModel so
the GPU worker can switch the response backend without changing callers.

Expected vLLM endpoint:
    POST /v1/chat/completions
"""

from __future__ import annotations

import json
import socket
import time
from json import JSONDecodeError
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .chat_model import ChatGeneration


class VLLMClientError(RuntimeError):
    """Raised when the local vLLM service cannot complete a request."""


class VLLMChatModel:
    """OpenAI-compatible client for a locally hosted vLLM chat model."""

    backend = "vllm"

    def __init__(
        self,
        *,
        base_url: str,
        model_name: str,
        timeout_seconds: float = 180.0,
        enable_thinking: bool = False,
        api_key: str | None = None,
    ) -> None:
        base_url = base_url.strip().rstrip("/")
        model_name = model_name.strip()

        if not base_url:
            raise ValueError("base_url cannot be empty.")
        if not model_name:
            raise ValueError("model_name cannot be empty.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")

        self.base_url = base_url
        self.model_name = model_name
        self.timeout_seconds = float(timeout_seconds)
        self.enable_thinking = bool(enable_thinking)
        self.api_key = api_key.strip() if api_key else None

    @property
    def input_device(self) -> None:
        """Compatibility placeholder.

        The vLLM model lives in a separate process, so this client does not own
        a torch device.
        """
        return None

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> ChatGeneration:
        """Generate one non-streaming chat completion through local vLLM."""

        if not messages:
            raise ValueError("messages cannot be empty.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(
                "temperature must be between 0.0 and 2.0."
            )

        preparation_started = time.perf_counter()

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_new_tokens),
            "stream": False,
            # Keep Qwen3 response behavior aligned with the existing worker.
            "chat_template_kwargs": {
                "enable_thinking": self.enable_thinking,
            },
        }

        body = json.dumps(payload).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = Request(
            url=f"{self.base_url}/v1/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )

        input_preparation_seconds = (
            time.perf_counter() - preparation_started
        )

        started = time.perf_counter()

        try:
            with urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                raw = response.read()

        except HTTPError as exc:
            detail = _read_error_body(exc)
            raise VLLMClientError(
                f"vLLM returned HTTP {exc.code}: {detail}"
            ) from exc

        except (URLError, TimeoutError, socket.timeout) as exc:
            raise VLLMClientError(
                f"Unable to reach vLLM at {self.base_url}: {exc}"
            ) from exc

        generation_seconds = time.perf_counter() - started

        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, JSONDecodeError) as exc:
            raise VLLMClientError(
                "vLLM returned a malformed JSON response."
            ) from exc

        return self._to_chat_generation(
            data=data,
            generation_seconds=generation_seconds,
            input_preparation_seconds=input_preparation_seconds,
        )

    def health(self) -> dict[str, Any]:
        """Return vLLM's model registry response.

        Useful during GPU-worker startup to verify that the expected response
        model is available before accepting Agent requests.
        """

        request = Request(
            url=f"{self.base_url}/v1/models",
            headers=self._headers(),
            method="GET",
        )

        try:
            with urlopen(
                request,
                timeout=min(self.timeout_seconds, 10.0),
            ) as response:
                raw = response.read()

        except HTTPError as exc:
            detail = _read_error_body(exc)
            raise VLLMClientError(
                f"vLLM health check returned HTTP {exc.code}: {detail}"
            ) from exc

        except (URLError, TimeoutError, socket.timeout) as exc:
            raise VLLMClientError(
                f"Unable to reach vLLM at {self.base_url}: {exc}"
            ) from exc

        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, JSONDecodeError) as exc:
            raise VLLMClientError(
                "vLLM /v1/models returned malformed JSON."
            ) from exc

        if not isinstance(data, dict):
            raise VLLMClientError(
                "vLLM /v1/models returned an unexpected response."
            )

        return data

    def ensure_ready(self) -> None:
        """Raise if the configured served model is not exposed by vLLM."""

        data = self.health()
        models = data.get("data")

        if not isinstance(models, list):
            raise VLLMClientError(
                "vLLM /v1/models response does not contain a model list."
            )

        served_ids = {
            str(item.get("id"))
            for item in models
            if isinstance(item, dict) and item.get("id") is not None
        }

        if self.model_name not in served_ids:
            raise VLLMClientError(
                "Configured vLLM model "
                f"{self.model_name!r} is not available. "
                f"Served models: {sorted(served_ids)!r}"
            )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _to_chat_generation(
        self,
        *,
        data: dict[str, Any],
        generation_seconds: float,
        input_preparation_seconds: float,
    ) -> ChatGeneration:
        choices = data.get("choices")
        usage = data.get("usage")

        if not isinstance(choices, list) or not choices:
            raise VLLMClientError(
                "vLLM response does not contain choices."
            )

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise VLLMClientError(
                "vLLM returned an invalid first choice."
            )

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise VLLMClientError(
                "vLLM response does not contain an assistant message."
            )

        text = message.get("content")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise VLLMClientError(
                "vLLM assistant content is not text."
            )

        if not isinstance(usage, dict):
            usage = {}

        input_tokens = _safe_non_negative_int(
            usage.get("prompt_tokens")
        )
        output_tokens = _safe_non_negative_int(
            usage.get("completion_tokens")
        )

        finish_reason = first_choice.get("finish_reason")
        if finish_reason is not None:
            finish_reason = str(finish_reason)

        returned_model = data.get("model")
        model_name = (
            str(returned_model)
            if returned_model
            else self.model_name
        )

        # The current non-streaming OpenAI-compatible endpoint does not expose
        # a trustworthy TTFT/decode split. Keep these unset rather than
        # fabricating values. Overall request timing and usage remain available.
        return ChatGeneration(
            text=text.strip(),
            model_name=model_name,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_preparation_seconds=input_preparation_seconds,
            generation_seconds=generation_seconds,
            time_to_first_token_seconds=None,
            decode_seconds=None,
            decode_tokens_per_second=None,
        )


def _safe_non_negative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _read_error_body(exc: HTTPError) -> str:
    try:
        raw = exc.read()
        detail = raw.decode("utf-8", errors="replace").strip()
    except Exception:
        detail = ""

    if not detail:
        detail = str(exc.reason or "unknown error")

    # Avoid copying an unexpectedly huge server error into application logs.
    return detail[:2000]
