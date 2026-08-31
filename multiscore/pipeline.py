"""End-to-end MULTI-SCORE retrieval: Pyramid Rank then multimodal re-ranking.

    query --[Stage-1: Pyramid Rank over MRL levels]--> top-K shortlist
          --[Stage-2: alpha * S_CoT + (1 - alpha) * S_QA]--> re-ranked top-K

Stage-1 is cheap and text-only (it runs on offline captions); Stage-2 is
expensive but only ever sees ``K << N`` candidates, and consumes their native
modality.  Neither stage is trained or fine-tuned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from multiscore.config import MultiScoreConfig
from multiscore.data.schema import Candidate, Query
from multiscore.models.base import EmbeddingBackend, MLLMBackend
from multiscore.models.registry import load_embedder, load_mllm
from multiscore.stage1.mrl import MRLEmbeddingStore
from multiscore.stage1.naive import naive_full_scale_rank
from multiscore.stage1.pyramid_rank import Stage1Output, pyramid_rank
from multiscore.stage2.rerank import Stage2Output, rerank
from multiscore.utils.logging import get_logger
from multiscore.utils.timer import Timer

LOGGER = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Final ranking for one query, with both stages' diagnostics attached."""

    query_id: str
    ranked_ids: List[str]
    scores: List[float]
    stage1: Optional[Stage1Output] = None
    stage2: Optional[Stage2Output] = None
    timings: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "query_id": self.query_id,
            "ranked_ids": list(self.ranked_ids),
            "scores": [float(s) for s in self.scores],
            "timings_ms": {k: float(v) for k, v in self.timings.items()},
        }
        if self.stage1 is not None:
            payload["stage1"] = {
                "iterations": self.stage1.iterations,
                "final_level": self.stage1.final_level,
                "level_history": list(self.stage1.level_history),
                "cost": int(self.stage1.cost),
                "naive_cost": int(self.stage1.naive_cost),
                "speedup": round(self.stage1.speedup, 3),
            }
        if self.stage2 is not None:
            payload["stage2"] = {
                "cot_scores": [float(s) for s in self.stage2.cot_scores],
                "qa_scores": [float(s) for s in self.stage2.qa_scores],
                "questions": [
                    {"question": p.question, "answer": p.answer} for p in self.stage2.questions
                ],
            }
        return payload

    def explain(self, top: int = 3) -> str:
        """Human-readable trace: why the top candidates were ranked where they are."""

        lines = [f"query {self.query_id}"]
        if self.stage2 is not None and self.stage2.qa_logs:
            for log in self.stage2.qa_logs[:top]:
                lines.append(log.explain())
        else:
            for rank, (cid, score) in enumerate(zip(self.ranked_ids[:top], self.scores[:top]), 1):
                lines.append(f"  {rank}. {cid}  score={score:.4f}")
        return "\n".join(lines)


class MultiScore:
    """The two-stage retriever.

    Parameters
    ----------
    config:
        Full pipeline config; see :mod:`multiscore.config` and ``configs/``.
    embedder, mllm:
        Optional pre-constructed backends.  When omitted they are lazily loaded
        from the registry using the names in ``config`` -- lazily, so that a
        Stage-1-only run never has to instantiate a 7B MLLM.
    """

    def __init__(
        self,
        config: Optional[MultiScoreConfig] = None,
        embedder: Optional[EmbeddingBackend] = None,
        mllm: Optional[MLLMBackend] = None,
    ) -> None:
        self.config = config or MultiScoreConfig()
        self._embedder = embedder
        self._mllm = mllm
        self.timer = Timer()

    # ------------------------------------------------------------------ #
    # lazily-loaded backends
    # ------------------------------------------------------------------ #
    @property
    def embedder(self) -> EmbeddingBackend:
        if self._embedder is None:
            LOGGER.info("loading Stage-1 embedder: %s", self.config.stage1.embedder)
            self._embedder = load_embedder(
                self.config.stage1.embedder,
                device=self.config.device,
                dtype=self.config.dtype,
            )
        return self._embedder

    @property
    def mllm(self) -> MLLMBackend:
        if self._mllm is None:
            LOGGER.info("loading Stage-2 MLLM: %s", self.config.stage2.backend)
            self._mllm = load_mllm(
                self.config.stage2.backend,
                device=self.config.device,
                dtype=self.config.dtype,
            )
        return self._mllm

    # ------------------------------------------------------------------ #
    # indexing
    # ------------------------------------------------------------------ #
    def index_candidates(
        self, candidates: Sequence[Candidate], batch_size: int = 128
    ) -> MRLEmbeddingStore:
        """Embed a captioned candidate database into a searchable store."""

        texts = [c.stage1_text() for c in candidates]
        embeddings = self.embedder.encode(texts, batch_size=batch_size)
        return MRLEmbeddingStore(
            embeddings, mrl=self.config.stage1.mrl, ids=[c.id for c in candidates]
        )

    def encode_query(self, query: Query) -> np.ndarray:
        return self.embedder.encode([query.stage1_text()], is_query=True)[0]

    # ------------------------------------------------------------------ #
    # retrieval
    # ------------------------------------------------------------------ #
    def retrieve(
        self,
        query: Query,
        store: MRLEmbeddingStore,
        candidates: Optional[Sequence[Candidate]] = None,
        query_embedding: Optional[np.ndarray] = None,
        stage2: bool = True,
        naive_stage1: bool = False,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """Retrieve for one query.

        Parameters
        ----------
        store:
            The Stage-1 MRL index over the database.
        candidates:
            Candidate objects, needed only when ``stage2`` is on (Stage-2 reads
            their text and media).  Must be aligned with ``store.ids``.
        naive_stage1:
            Replace Pyramid Rank with exhaustive full-resolution scoring -- the
            ablation baseline of the paper.
        """

        cfg = self.config
        k = int(top_k if top_k is not None else cfg.stage1.top_k)

        if query_embedding is None:
            with self.timer.section("encode_query"):
                query_embedding = self.encode_query(query)

        with self.timer.section("stage1"):
            if naive_stage1:
                stage1 = naive_full_scale_rank(query_embedding, store, top_k=k)
            else:
                stage1 = pyramid_rank(query_embedding, store, config=cfg.stage1, top_k=k)

        ranked_ids = list(stage1.ids)
        scores = [float(s) for s in stage1.scores]
        stage2_out: Optional[Stage2Output] = None

        if stage2 and (cfg.stage2.enable_cot or cfg.stage2.enable_qa):
            if candidates is None:
                raise ValueError("Stage-2 needs the candidate objects; pass `candidates=`")
            by_id = {c.id: c for c in candidates}
            shortlist = [by_id[cid] for cid in ranked_ids if cid in by_id]

            with self.timer.section("stage2"):
                stage2_out = rerank(
                    query.stage2_text(),
                    [c.stage2_text() for c in shortlist],
                    backend=self.mllm,
                    config=cfg.stage2,
                    candidate_ids=[c.id for c in shortlist],
                    query_media=query.media(),
                    candidate_media=[c.media() for c in shortlist],
                )
            ranked_ids = list(stage2_out.ids)
            scores = [float(s) for s in stage2_out.scores]

        return RetrievalResult(
            query_id=query.id,
            ranked_ids=ranked_ids,
            scores=scores,
            stage1=stage1,
            stage2=stage2_out,
            timings=self.timer.report(),
        )

    def retrieve_all(
        self,
        queries: Sequence[Query],
        store: MRLEmbeddingStore,
        candidates: Optional[Sequence[Candidate]] = None,
        stage2: bool = True,
        naive_stage1: bool = False,
        progress: bool = True,
    ) -> List[RetrievalResult]:
        """Run :meth:`retrieve` over a query set, batching the query encoder."""

        with self.timer.section("encode_queries"):
            query_embeddings = self.embedder.encode(
                [q.stage1_text() for q in queries], is_query=True
            )

        iterator = range(len(queries))
        if progress:
            try:
                from tqdm.auto import tqdm  # type: ignore

                iterator = tqdm(iterator, desc="retrieval", unit="query")
            except ImportError:
                pass

        return [
            self.retrieve(
                queries[i],
                store,
                candidates=candidates,
                query_embedding=query_embeddings[i],
                stage2=stage2,
                naive_stage1=naive_stage1,
            )
            for i in iterator
        ]
