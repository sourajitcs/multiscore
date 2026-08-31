"""Model backends: MRL text embedders (Stage-1) and MLLMs (Stage-2)."""

from multiscore.models.base import EmbeddingBackend, MLLMBackend
from multiscore.models.registry import (
    available_backends,
    load_embedder,
    load_mllm,
    register_embedder,
    register_mllm,
)

__all__ = [
    "EmbeddingBackend",
    "MLLMBackend",
    "available_backends",
    "load_embedder",
    "load_mllm",
    "register_embedder",
    "register_mllm",
]
