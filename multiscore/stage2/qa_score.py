"""Question Answering Relevance Score, ``S_QA`` (Eq. 8).

An MLLM turns the query into ``M`` discriminative Yes/No question-answer pairs.
Each Stage-1 candidate is then used as the *only* context for answering those
questions, and the score is the answer accuracy

    S_QA(q, c_k) = (1 / M) * sum_m 1[ ans^m == MLLM(que^m, c_k) ].

The per-question logs are kept: they are the natural-language explanation of
*why* a candidate was ranked where it was.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from multiscore.config import Stage2Config
from multiscore.models.base import MLLMBackend
from multiscore.stage2.prompts import qa_answer_prompt, qa_generation_prompt


@dataclass
class QAPair:
    question: str
    answer: str  # "Yes" or "No"


@dataclass
class QALog:
    """Interpretable trace for one (query, candidate) pair."""

    candidate_id: str
    questions: List[str] = field(default_factory=list)
    gold: List[str] = field(default_factory=list)
    predicted: List[str] = field(default_factory=list)
    score: float = 0.0

    def explain(self) -> str:
        lines = [f"candidate={self.candidate_id}  S_QA={self.score:.3f}"]
        for question, gold, pred in zip(self.questions, self.gold, self.predicted):
            mark = "OK  " if gold == pred else "MISS"
            lines.append(f"  [{mark}] {question}  gold={gold} pred={pred}")
        return "\n".join(lines)


_QA_LINE = re.compile(
    r"Q\s*\d*\s*[:.)]?\s*(?P<question>.+?)\s*[;\n]?\s*A\s*[:.)]\s*(?P<answer>Yes|No)\b",
    re.IGNORECASE | re.DOTALL,
)


def normalize_yes_no(text: str) -> str:
    """Map a free-form model answer onto ``Yes`` / ``No`` (unparseable -> ``No``)."""

    lowered = (text or "").strip().lower()
    if lowered.startswith("yes") or re.search(r"\byes\b", lowered):
        return "Yes"
    if lowered.startswith("no") or re.search(r"\bno\b", lowered):
        return "No"
    return "No"


def parse_qa_pairs(raw: str, max_pairs: Optional[int] = None) -> List[QAPair]:
    """Parse the ``Q..? A: Yes/No;`` format emitted by the generation prompt."""

    pairs: List[QAPair] = []
    for match in _QA_LINE.finditer(raw or ""):
        question = " ".join(match.group("question").split()).strip(" ;")
        if not question:
            continue
        pairs.append(QAPair(question=question, answer=match.group("answer").capitalize()))
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


class QARelevanceScorer:
    """Generates the query's questions once, then scores every candidate."""

    def __init__(self, backend: MLLMBackend, config: Optional[Stage2Config] = None) -> None:
        self.backend = backend
        self.config = config or Stage2Config()
        self.last_logs: List[QALog] = []

    # ------------------------------------------------------------------ #
    def generate_questions(
        self,
        query: str,
        query_media: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[QAPair]:
        prompt = qa_generation_prompt(query, self.config.num_questions)
        raw = self.backend.generate(
            prompt,
            media=query_media,
            max_new_tokens=64 * max(1, self.config.num_questions),
            temperature=self.config.temperature,
        )
        return parse_qa_pairs(raw, max_pairs=self.config.num_questions)

    def score_candidates(
        self,
        query: str,
        candidates: Sequence[str],
        candidate_ids: Optional[Sequence[str]] = None,
        query_media: Optional[Sequence[Dict[str, Any]]] = None,
        candidate_media: Optional[Sequence[Optional[Sequence[Dict[str, Any]]]]] = None,
        qa_pairs: Optional[Sequence[QAPair]] = None,
    ) -> np.ndarray:
        """Accuracy of the query's questions when answered from each candidate."""

        pairs = list(qa_pairs) if qa_pairs is not None else self.generate_questions(query, query_media)
        ids = list(candidate_ids) if candidate_ids is not None else [str(i) for i in range(len(candidates))]
        media_list: List[Optional[Sequence[Dict[str, Any]]]] = (
            list(candidate_media) if candidate_media is not None else [None] * len(candidates)
        )

        self.last_logs = []
        if not pairs:  # degenerate query -> uninformative, uniform score
            return np.zeros(len(candidates), dtype=np.float32)

        scores = np.zeros(len(candidates), dtype=np.float32)
        for i, candidate in enumerate(candidates):
            log = QALog(candidate_id=ids[i])
            correct = 0
            for pair in pairs:
                prompt = qa_answer_prompt(candidate, pair.question)
                raw = self.backend.generate(
                    prompt,
                    media=media_list[i],
                    max_new_tokens=self.config.max_answer_tokens,
                    temperature=self.config.temperature,
                )
                predicted = normalize_yes_no(raw)
                correct += int(predicted == pair.answer)
                log.questions.append(pair.question)
                log.gold.append(pair.answer)
                log.predicted.append(predicted)
            log.score = correct / len(pairs)
            scores[i] = log.score
            self.last_logs.append(log)
        return scores


def qa_score(
    query: str,
    candidates: Sequence[str],
    backend: MLLMBackend,
    config: Optional[Stage2Config] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Functional shorthand for :meth:`QARelevanceScorer.score_candidates`."""

    return QARelevanceScorer(backend, config).score_candidates(query, candidates, **kwargs)
