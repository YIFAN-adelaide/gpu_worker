"""Self-contained BGE-M3 dense embedding runtime for the GPU worker."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class BGEM3EmbeddingModel:
    """BGE-M3 dense embedding runtime returning normalized float32 vectors."""

    model_name = "BAAI/bge-m3"

    def __init__(
        self,
        *,
        model_path: str,
        max_seq_length: int = 8192,
        trust_remote_code: bool = True,
    ) -> None:
        if not model_path or not model_path.strip():
            raise ValueError("model_path cannot be empty.")

        self.model_path = model_path
        self.max_seq_length = int(max_seq_length)

        if self.max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive.")

        self.device = (
            torch.device("cuda:0")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

        dtype = (
            torch.float16
            if self.device.type == "cuda"
            else torch.float32
        )

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

        hidden_size = getattr(
            self.model.config,
            "hidden_size",
            None,
        )
        if hidden_size is None:
            raise RuntimeError(
                "Could not determine BGE-M3 embedding dimension."
            )

        self.dimension = int(hidden_size)

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 16,
    ) -> list[list[float]]:
        text_list = list(texts)

        if not text_list:
            return []

        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        all_embeddings: list[torch.Tensor] = []

        for start in range(0, len(text_list), batch_size):
            batch = text_list[start : start + batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0]
                embeddings = F.normalize(
                    embeddings,
                    p=2,
                    dim=1,
                )

            all_embeddings.append(
                embeddings.float().cpu()
            )

        dense = torch.cat(
            all_embeddings,
            dim=0,
        ).numpy().astype(np.float32, copy=False)

        if dense.ndim != 2 or dense.shape[1] != self.dimension:
            raise RuntimeError(
                "Unexpected BGE-M3 output shape: "
                f"{dense.shape}."
            )

        return dense.tolist()

    def warm_up(self) -> None:
        self.encode(["warm up"], batch_size=1)
