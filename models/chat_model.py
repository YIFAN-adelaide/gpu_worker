"""Generic Hugging Face causal-chat runtime used by GPU worker profiles."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gptqmodel import GPTQModel


@dataclass(frozen=True, slots=True)
class ChatGeneration:
    text: str
    model_name: str
    finish_reason: str | None
    input_tokens: int
    output_tokens: int
    input_preparation_seconds: float
    generation_seconds: float
    time_to_first_token_seconds: float | None
    decode_seconds: float | None
    decode_tokens_per_second: float | None

    @property
    def tokens_per_second(self) -> float | None:
        if self.generation_seconds <= 0:
            return None
        return self.output_tokens / self.generation_seconds


class _GenerationTimingStreamer:
    """Minimal HF-compatible streamer used only for token timing.

    Hugging Face generation normally sends the prompt once and then generated
    tokens one by one. The prompt callback is ignored. The first generated
    token callback marks practical TTFT. No text decoding happens here.
    """

    def __init__(self, *, prompt_tokens: int) -> None:
        self._prompt_tokens = max(1, int(prompt_tokens))
        self._prompt_seen = False
        self.first_token_at: float | None = None
        self.ended_at: float | None = None

    def put(self, value: Any) -> None:
        now = time.perf_counter()
        token_count: int | None = None
        numel = getattr(value, "numel", None)
        if callable(numel):
            try:
                token_count = int(numel())
            except (TypeError, ValueError):
                token_count = None

        if (
            not self._prompt_seen
            and token_count is not None
            and token_count >= self._prompt_tokens
        ):
            self._prompt_seen = True
            return

        if self.first_token_at is None:
            self.first_token_at = now

    def end(self) -> None:
        self.ended_at = time.perf_counter()


class ChatModel:
    """One loaded physical chat model shared by one or more logical roles."""

    def __init__(
        self,
        *,
        model_path: str,
        enable_thinking: bool = False,
        trust_remote_code: bool = True,
        device_map: str = "auto",
        backend: str = "transformers",
    ) -> None:
        if not model_path or not model_path.strip():
            raise ValueError("model_path cannot be empty.")

        self.backend = backend.strip().lower()
        self.model_path = model_path
        self.model_name = Path(
            model_path.rstrip("/\\")
        ).name
        self.enable_thinking = bool(enable_thinking)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

        if self.backend == "gptqmodel":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "GPTQModel backend requires CUDA."
                )

            self.model = GPTQModel.load(
                model_path,
                device="cuda:0",
                trust_remote_code=trust_remote_code,
            )

        elif self.backend == "transformers":
            model_kwargs: dict[str, Any] = {
                "trust_remote_code": trust_remote_code,
            }

            if torch.cuda.is_available():
                model_kwargs["device_map"] = device_map
                model_kwargs["torch_dtype"] = "auto"
            else:
                model_kwargs["torch_dtype"] = torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs,
            )

        else:
            raise ValueError(
                f"Unsupported chat model backend: {self.backend!r}"
            )
        if hasattr(self.model, "eval"):
            self.model.eval()

        if (
            not torch.cuda.is_available()
            and hasattr(self.model, "to")
        ):
            self.model.to("cpu")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def input_device(self) -> torch.device:
        device = getattr(self.model, "device", None)
        if device is not None:
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda:0")

        return torch.device("cpu")

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> ChatGeneration:
        if not messages:
            raise ValueError("messages cannot be empty.")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(
                "temperature must be between 0.0 and 2.0."
            )

        preparation_started = time.perf_counter()

        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
            "enable_thinking": self.enable_thinking,
        }

        inputs = self.tokenizer.apply_chat_template(
            messages,
            **template_kwargs,
        )
        inputs = {
            key: value.to(self.input_device)
            for key, value in inputs.items()
        }

        prompt_tokens = int(
            inputs["input_ids"].shape[-1]
        )

        input_preparation_seconds = (
            time.perf_counter() - preparation_started
        )

        '''
        timing_streamer = _GenerationTimingStreamer(
            prompt_tokens=prompt_tokens
        )
        '''
        generation_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": temperature > 0.0,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
#            "streamer": timing_streamer,
        }

        if temperature > 0.0:
            generation_kwargs["temperature"] = temperature

        started = time.perf_counter()

        with torch.inference_mode():
            outputs = self.model.generate(
                **generation_kwargs
            )

        generation_seconds = (
            time.perf_counter() - started
        )

        new_tokens = outputs[0][prompt_tokens:]
        output_tokens = int(new_tokens.shape[-1])


        #time_to_first_token_seconds: float | None = None
        #decode_seconds: float | None = None
        #decode_tokens_per_second: float | None = None

        time_to_first_token_seconds: float | None = None
        decode_seconds: float | None = None
        decode_tokens_per_second: float | None = None
        '''
        if timing_streamer.first_token_at is not None:
            time_to_first_token_seconds = max(
                0.0,
                timing_streamer.first_token_at - started,
            )
            decode_seconds = max(
                0.0,
                generation_seconds - time_to_first_token_seconds,
            )
            remaining_decode_tokens = max(0, output_tokens - 1)
            if decode_seconds > 0 and remaining_decode_tokens > 0:
                decode_tokens_per_second = (
                    remaining_decode_tokens / decode_seconds
                )
        '''
        text = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        ).strip()

        finish_reason = (
            "length"
            if output_tokens >= max_new_tokens
            else "stop"
        )

        return ChatGeneration(
            text=text,
            model_name=self.model_name,
            finish_reason=finish_reason,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            input_preparation_seconds=input_preparation_seconds,
            generation_seconds=generation_seconds,
            time_to_first_token_seconds=time_to_first_token_seconds,
            decode_seconds=decode_seconds,
            decode_tokens_per_second=decode_tokens_per_second,
        )
