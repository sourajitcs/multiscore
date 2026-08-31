"""Deterministic, dependency-free stand-ins for the real backbones.

They exist so the *algorithms* -- Pyramid Rank, the CoT/QA plumbing, the
re-ranking arithmetic and the metrics -- can be exercised in unit tests, in CI
and in ``examples/toy_demo.py`` without downloading a 7B checkpoint or a
multi-terabyte benchmark.  Numbers produced with these stubs are meaningless as
retrieval results; they are only meant to be reproducible.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from multiscore.models.base import EmbeddingBackend, MLLMBackend


def _seed_from_text(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HashingMRLEmbedder(EmbeddingBackend):
    """Bag-of-words hashing encoder with an MRL-like energy decay.

    Coordinates are drawn from a text-seeded hash and scaled by ``1 / (1 + i)``
    so that, exactly like a trained Matryoshka model, the leading dimensions
    carry most of the norm and short prefixes are already informative.
    """

    name = "stub-mrl"

    def __init__(self, dim: int = 1024, decay: float = 1.0) -> None:
        self._dim = int(dim)
        self._decay = float(decay)
        self._scale = 1.0 / (1.0 + np.arange(self._dim, dtype=np.float32)) ** self._decay

    @property
    def full_dim(self) -> int:
        return self._dim

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 64,
        is_query: bool = False,
    ) -> np.ndarray:
        rows = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _tokenize(text) or ["<empty>"]
            for token in tokens:
                rng = np.random.default_rng(_seed_from_text(token))
                rows[row] += rng.standard_normal(self._dim, dtype=np.float32)
            rows[row] /= len(tokens)
        rows *= self._scale
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        return rows / np.maximum(norms, 1e-12)


class LexicalMLLM(MLLMBackend):
    """A rule-based 'MLLM' driven by lexical overlap.

    * :meth:`generate` answers yes/no questions by checking whether the content
      words of the question occur in the prompt context, and otherwise echoes a
      short deterministic caption.
    * :meth:`embed_with_marker` hashes the prompt text up to ``marker``.
    """

    name = "stub-mllm"

    def __init__(self, dim: int = 256) -> None:
        self._embedder = HashingMRLEmbedder(dim=dim, decay=0.0)

    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        media: Optional[Sequence[Dict[str, Any]]] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        if "Answer with only Yes or No" in prompt or prompt.rstrip().endswith("?"):
            return self._answer_yes_no(prompt)
        if "Yes/No questions" in prompt:
            return self._fabricate_questions(prompt)
        return "a deterministic stub caption of the provided content"

    def embed_with_marker(
        self,
        prompt: str,
        media: Optional[Sequence[Dict[str, Any]]] = None,
        marker: str = "<emb>",
    ) -> np.ndarray:
        text = prompt.split(marker)[0]
        return self._embedder.encode([text])[0]

    # ------------------------------------------------------------------ #
    def _answer_yes_no(self, prompt: str) -> str:
        question = prompt.strip().splitlines()[-1]
        context = prompt[: prompt.rfind(question)]
        q_tokens = set(_tokenize(question)) - _STOPWORDS
        c_tokens = set(_tokenize(context))
        if not q_tokens:
            return "No"
        overlap = len(q_tokens & c_tokens) / len(q_tokens)
        return "Yes" if overlap >= 0.5 else "No"

    def _fabricate_questions(self, prompt: str) -> str:
        """One question per content word of the query, all with answer 'Yes'."""

        head, _, tail = prompt.partition("multimodal query:")
        query = (tail or head).splitlines()[0]
        requested = re.search(r"Generate (\d+) Yes/No questions", prompt)
        budget = int(requested.group(1)) if requested else 5

        tokens = [t for t in _tokenize(query) if t not in _STOPWORDS and len(t) > 2]
        seen: List[str] = []
        for token in tokens:
            if token not in seen:
                seen.append(token)
        lines = [
            "Q{}: Does the candidate involve {}? A: Yes".format(i + 1, token)
            for i, token in enumerate(seen[:budget])
        ]
        return "\n".join(lines)


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "do", "for", "from",
    "given", "in", "is", "it", "of", "on", "or", "question", "questions", "the",
    "there", "this", "to", "with", "you", "your", "answer", "only", "yes", "no",
    "generate", "output", "caption", "candidate", "query", "above", "each",
    "some", "any", "that", "these", "those", "was", "were", "will", "can",
}
