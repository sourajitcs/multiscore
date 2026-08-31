"""Combine the two Stage-2 scores and re-rank the Stage-1 shortlist (Eq. 9).

    S_rerank(q, c_k) = alpha * S_CoT(q, c_k) + (1 - alpha) * S_QA(q, c_k)

``alpha`` is the only Stage-2 hyper-parameter; see ``configs/`` for the values
used per modality and ``scripts/reproduce/sweep_alpha.py`` for the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from multiscore.config import Stage2Config
from multiscore.models.base import MLLMBackend
from multiscore.stage2.cot_score import BidirectionalCoTScorer
from multiscore.stage2.qa_score import QALog, QAPair, QARelevanceScorer


@dataclass
class Stage2Output:
    """Re-ranked shortlist plus the individual score components."""

    indices: np.ndarray  # positions into the Stage-1 shortlist, best first
    scores: np.ndarray  # S_rerank, aligned with `indices`
    cot_scores: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    qa_scores: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    ids: List[str] = field(default_factory=list)
    qa_logs: List[QALog] = field(default_factory=list)
    questions: List[QAPair] = field(default_factory=list)


def combine_scores(
    cot_scores: np.ndarray, qa_scores: np.ndarray, alpha: float
) -> np.ndarray:
    """Convex combination of the two Stage-2 scores (Eq. 9)."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
    cot_scores = np.asarray(cot_scores, dtype=np.float32)
    qa_scores = np.asarray(qa_scores, dtype=np.float32)
    if cot_scores.shape != qa_scores.shape:
        raise ValueError(
            f"score shapes disagree: {cot_scores.shape} vs {qa_scores.shape}"
        )
    return alpha * cot_scores + (1.0 - alpha) * qa_scores


def rerank(
    query: str,
    candidates: Sequence[str],
    backend: MLLMBackend,
    config: Optional[Stage2Config] = None,
    candidate_ids: Optional[Sequence[str]] = None,
    query_media: Optional[Sequence[Dict[str, Any]]] = None,
    candidate_media: Optional[Sequence[Optional[Sequence[Dict[str, Any]]]]] = None,
    top_k: Optional[int] = None,
) -> Stage2Output:
    """Score the Stage-1 shortlist with ``S_CoT`` and ``S_QA`` and re-order it.

    ``S_CoT`` and ``S_QA`` are independent; in the paper they are executed in
    parallel on separate workers, so the wall-clock cost is ``max`` of the two
    rather than their sum.
    """

    config = config or Stage2Config()
    ids = list(candidate_ids) if candidate_ids is not None else [str(i) for i in range(len(candidates))]
    n = len(candidates)

    cot_scores = np.zeros(n, dtype=np.float32)
    qa_scores = np.zeros(n, dtype=np.float32)
    qa_logs: List[QALog] = []
    questions: List[QAPair] = []

    if config.enable_cot:
        cot_scores = BidirectionalCoTScorer(backend, config).score_candidates(
            query, candidates, query_media=query_media, candidate_media=candidate_media
        )

    if config.enable_qa:
        qa_scorer = QARelevanceScorer(backend, config)
        questions = qa_scorer.generate_questions(query, query_media)
        qa_scores = qa_scorer.score_candidates(
            query,
            candidates,
            candidate_ids=ids,
            query_media=query_media,
            candidate_media=candidate_media,
            qa_pairs=questions,
        )
        qa_logs = qa_scorer.last_logs

    alpha = config.alpha
    if not config.enable_qa:
        alpha = 1.0
    elif not config.enable_cot:
        alpha = 0.0

    combined = combine_scores(cot_scores, qa_scores, alpha)
    order = np.argsort(-combined, kind="stable")
    if top_k is not None:
        order = order[:top_k]

    return Stage2Output(
        indices=order.astype(np.int64),
        scores=combined[order],
        cot_scores=cot_scores[order],
        qa_scores=qa_scores[order],
        ids=[ids[i] for i in order],
        qa_logs=[qa_logs[i] for i in order] if qa_logs else [],
        questions=questions,
    )
