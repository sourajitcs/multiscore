"""Bidirectional-CoT Embedding Score, ``S_CoT`` (Eq. 7).

For a query ``q`` and a Stage-1 candidate ``c_k`` we build two few-shot
chain-of-thought prompts -- query-to-candidate and candidate-to-query -- each
terminated by an ``<emb>`` marker.  The hidden state preceding the marker is
taken as an embedding (``z_q2c`` and ``z_c2q``), and

    S_CoT(q, c_k) = cosine(z_q2c(q, c_k), z_c2q(c_k, q)).

Both directions matter: alignment is asymmetric (a candidate may cover every
query concept while introducing many of its own), and scoring both ways is what
makes the score discriminative without any supervision.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from multiscore.config import Stage2Config
from multiscore.models.base import MLLMBackend
from multiscore.stage2.prompts import (
    candidate_to_query_prompt,
    query_to_candidate_prompt,
)


def cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < eps:
        return 0.0
    return float(np.dot(a, b) / denominator)


class BidirectionalCoTScorer:
    """Computes ``S_CoT`` for a query against a list of candidates."""

    def __init__(self, backend: MLLMBackend, config: Optional[Stage2Config] = None) -> None:
        self.backend = backend
        self.config = config or Stage2Config()

    # ------------------------------------------------------------------ #
    def score_pair(
        self,
        query: str,
        candidate: str,
        query_media: Optional[Sequence[Dict[str, Any]]] = None,
        candidate_media: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> float:
        """``S_CoT`` for a single (query, candidate) pair.

        ``*_media`` are passed straight through to the backend so that Stage-2
        can consume the *native* modality (image / video / audio) rather than
        the Stage-1 caption -- this is what avoids the modality-conversion
        information loss discussed in the paper.
        """

        emb_token = self.config.emb_token
        num_examples = self.config.num_cot_examples

        q2c_prompt = query_to_candidate_prompt(query, candidate, num_examples, emb_token)
        c2q_prompt = candidate_to_query_prompt(candidate, query, num_examples, emb_token)

        media = list(query_media or []) + list(candidate_media or [])
        z_q2c = self.backend.embed_with_marker(q2c_prompt, media=media, marker=emb_token)
        z_c2q = self.backend.embed_with_marker(c2q_prompt, media=media, marker=emb_token)
        return cosine(z_q2c, z_c2q)

    def score_candidates(
        self,
        query: str,
        candidates: Sequence[str],
        query_media: Optional[Sequence[Dict[str, Any]]] = None,
        candidate_media: Optional[Sequence[Optional[Sequence[Dict[str, Any]]]]] = None,
    ) -> np.ndarray:
        """``S_CoT`` for every candidate; returns a ``(len(candidates),)`` array."""

        media_list: List[Optional[Sequence[Dict[str, Any]]]] = (
            list(candidate_media) if candidate_media is not None else [None] * len(candidates)
        )
        scores = [
            self.score_pair(query, candidate, query_media, media_list[i])
            for i, candidate in enumerate(candidates)
        ]
        return np.asarray(scores, dtype=np.float32)


def cot_score(
    query: str,
    candidates: Sequence[str],
    backend: MLLMBackend,
    config: Optional[Stage2Config] = None,
    **media: Any,
) -> np.ndarray:
    """Functional shorthand for :meth:`BidirectionalCoTScorer.score_candidates`."""

    return BidirectionalCoTScorer(backend, config).score_candidates(
        query,
        candidates,
        query_media=media.get("query_media"),
        candidate_media=media.get("candidate_media"),
    )
