"""Offline embedding index: build once, reuse across queries and runs."""

from multiscore.index.store import EmbeddingIndex, build_index

__all__ = ["EmbeddingIndex", "build_index"]
