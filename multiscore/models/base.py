"""Backend interfaces.

MULTI-SCORE is deliberately backbone-agnostic: the paper reports the same gains
with Qwen2.5-Omni-3B/7B and Qwen2-Audio-7B in Stage-2.  Anything satisfying the
two protocols below can be dropped in.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class EmbeddingBackend(abc.ABC):
    """A text encoder producing nested (Matryoshka) embeddings.

    Implementations must return **unit-norm** level-``L`` vectors of width
    ``full_dim``; every shorter level is the corresponding prefix (Eq. 2).
    """

    name: str = "embedding-backend"

    @property
    @abc.abstractmethod
    def full_dim(self) -> int:
        """``D`` -- dimensionality of the finest MRL level."""

    @abc.abstractmethod
    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        is_query: bool = False,
    ) -> np.ndarray:
        """Encode ``texts`` into an ``(n, full_dim)`` array of unit-norm rows."""

    def encode_one(self, text: str, is_query: bool = False) -> np.ndarray:
        return self.encode([text], is_query=is_query)[0]


class MLLMBackend(abc.ABC):
    """A multimodal LLM used for captioning, QA scoring and CoT embedding.

    ``media`` entries are ``{"image": path}``, ``{"video": path}`` or
    ``{"audio": path}`` dictionaries; a backend may ignore modalities it does
    not support (the text-only stub does).
    """

    name: str = "mllm-backend"

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        media: Optional[Sequence[Dict[str, Any]]] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        """Free-form generation."""

    @abc.abstractmethod
    def embed_with_marker(
        self,
        prompt: str,
        media: Optional[Sequence[Dict[str, Any]]] = None,
        marker: str = "<emb>",
    ) -> np.ndarray:
        """Hidden state of the token *preceding* ``marker``.

        This is the "one-word limitation" trick used for the Bidirectional-CoT
        Embedding Score: the prompt is terminated with a marker token, and the
        last hidden state before it aggregates the whole prompt into a single
        vector without ever decoding.
        """

    def batch_generate(
        self,
        prompts: Sequence[str],
        media: Optional[Sequence[Optional[Sequence[Dict[str, Any]]]]] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> List[str]:
        """Default sequential implementation; override for real batching."""

        media = list(media) if media is not None else [None] * len(prompts)
        return [
            self.generate(
                prompt, media=m, max_new_tokens=max_new_tokens, temperature=temperature
            )
            for prompt, m in zip(prompts, media)
        ]
