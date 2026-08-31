"""Naive full-resolution MRL ranking -- the Stage-1 baseline of the paper.

This is what Pyramid Rank replaces: score every one of the ``N`` candidates with
the full ``D``-dimensional level-``L`` embedding, then sort.  Retrieval quality
is (by construction) the ceiling Pyramid Rank approaches to within ``eps``,
while the cost is a flat ``N * D``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from multiscore.stage1.mrl import MRLEmbeddingStore
from multiscore.stage1.pyramid_rank import Stage1Output


def naive_full_scale_rank(
    query_embedding: np.ndarray,
    store: MRLEmbeddingStore,
    top_k: int = 100,
    level: Optional[int] = None,
) -> Stage1Output:
    """Exhaustive cosine ranking at ``level`` (default: the finest level ``L``)."""

    mrl = store.mrl
    level = mrl.num_levels if level is None else level
    dim = mrl.level_dim(level)

    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)[:dim]
    scores = store.embeddings[:, :dim] @ query

    k = min(int(top_k), len(store))
    order = np.argpartition(-scores, k - 1)[:k] if k < len(store) else np.arange(len(store))
    order = order[np.argsort(-scores[order], kind="stable")]

    return Stage1Output(
        indices=order.astype(np.int64),
        scores=scores[order].astype(np.float32),
        ids=[store.ids[i] for i in order],
        iterations=1,
        final_level=level,
        level_history=[level],
        survivors_history=[len(store)],
        cost=len(store) * dim,
        naive_cost=len(store) * mrl.full_dim,
    )


def exact_top_k(
    query_embedding: np.ndarray, store: MRLEmbeddingStore, top_k: int = 100
) -> np.ndarray:
    """Ground-truth top-``K`` indices at level ``L`` (used by the guarantee tests)."""

    return naive_full_scale_rank(query_embedding, store, top_k=top_k).indices
