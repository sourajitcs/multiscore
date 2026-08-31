"""Qwen3 Embedding backend with Matryoshka truncation (Stage-1).

The paper uses ``Qwen3-Embedding-0.6B`` -- to our knowledge the only public
foundation embedding model that exposes an MRL hierarchy -- with ``L = 6``
nested levels from ``d = 32`` up to ``D = 1024``.

The checkpoint is text-only, which is why every image / video / audio item is
captioned first (see :mod:`multiscore.data.captioning`).  Nothing in Pyramid
Rank depends on that choice: swap in any encoder whose prefixes are nested and
the algorithm is unchanged.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from multiscore.models.base import EmbeddingBackend

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# Qwen3-Embedding is instruction-tuned; queries carry a task instruction while
# documents are encoded bare.
DEFAULT_QUERY_INSTRUCTION = (
    "Given a multimodal retrieval query, retrieve the database item that best "
    "matches it"
)


def last_token_pool(hidden_states, attention_mask):
    """Pool the final non-padding token (the pooling Qwen3-Embedding was trained with)."""

    import torch

    left_padded = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padded:
        return hidden_states[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), lengths]


class Qwen3MRLEmbedder(EmbeddingBackend):
    """Hugging Face wrapper around Qwen3-Embedding with MRL truncation."""

    name = "qwen3-mrl"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        full_dim: int = 1024,
        device: str = "auto",
        dtype: str = "bfloat16",
        max_length: int = 8192,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Qwen3MRLEmbedder needs `torch` and `transformers`: "
                "pip install -e '.[models]'"
            ) from exc

        self._torch = torch
        self.model_name = model_name
        self._full_dim = int(full_dim)
        self.max_length = int(max_length)
        self.query_instruction = query_instruction

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        torch_dtype = getattr(torch, dtype) if device != "cpu" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", trust_remote_code=trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code
        ).to(device)
        self.model.eval()

    @property
    def full_dim(self) -> int:
        return self._full_dim

    # ------------------------------------------------------------------ #
    def _format(self, texts: Sequence[str], is_query: bool) -> List[str]:
        if not is_query or not self.query_instruction:
            return list(texts)
        return [f"Instruct: {self.query_instruction}\nQuery: {t}" for t in texts]

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        is_query: bool = False,
    ) -> np.ndarray:
        torch = self._torch
        formatted = self._format(texts, is_query)
        chunks: List[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(formatted), batch_size):
                batch = formatted[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                outputs = self.model(**encoded)
                pooled = last_token_pool(
                    outputs.last_hidden_state, encoded["attention_mask"]
                )
                # MRL truncation to level L, then re-normalise (Eq. 3).
                pooled = pooled[:, : self._full_dim].float()
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
                chunks.append(pooled.cpu().numpy().astype(np.float32))

        if not chunks:
            return np.zeros((0, self._full_dim), dtype=np.float32)
        return np.concatenate(chunks, axis=0)


class ResNetMRLEmbedder(EmbeddingBackend):
    """Placeholder for the image-native MRL encoder of Kusupati et al. (2022).

    Reported in the appendix as an ablation (image-MRL vs. captioned text-MRL on
    NIGHTS).  Wire your own ResNet-MRL checkpoint here; the rest of Stage-1 is
    modality-agnostic and needs no changes.
    """

    name = "resnet-mrl"

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - stub
        raise NotImplementedError(
            "Provide a ResNet-MRL checkpoint and implement `encode`; see "
            "docs/METHOD.md#choice-of-mrl-embedding-model"
        )

    @property
    def full_dim(self) -> int:  # pragma: no cover - stub
        raise NotImplementedError

    def encode(self, texts, batch_size: int = 64, is_query: bool = False):  # pragma: no cover
        raise NotImplementedError
