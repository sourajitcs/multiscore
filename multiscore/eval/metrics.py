"""Recall@k, nDCG@k and friends.

Conventions follow the retrieval literature the paper compares against:

* ``R@k`` is the fraction of queries with *at least one* relevant item in the
  top-``k`` (the "% of queries with the relevant item in top-K" of Section 4).
* ``nDCG@k`` uses binary relevance with the standard ``1 / log2(rank + 1)``
  discount and an ideal ranking capped at ``min(k, #relevant)``.

Rankings are lists of candidate ids ordered best-first; ground truth is a set of
relevant ids per query.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence, Set

Ranking = Sequence[str]
Relevant = Iterable[str]


def _as_set(relevant: Relevant) -> Set[str]:
    return {str(r) for r in relevant}


def recall_at_k(rankings: Sequence[Ranking], relevants: Sequence[Relevant], k: int) -> float:
    """Fraction of queries whose top-``k`` contains at least one relevant item."""

    if not rankings:
        return 0.0
    hits = 0
    for ranking, relevant in zip(rankings, relevants):
        gold = _as_set(relevant)
        if gold and any(str(item) in gold for item in list(ranking)[:k]):
            hits += 1
    return 100.0 * hits / len(rankings)


def hit_at_k(rankings: Sequence[Ranking], relevants: Sequence[Relevant], k: int) -> float:
    """Alias of :func:`recall_at_k` used for candidate-survival analyses."""

    return recall_at_k(rankings, relevants, k)


def full_recall_at_k(
    rankings: Sequence[Ranking], relevants: Sequence[Relevant], k: int
) -> float:
    """Mean fraction of *all* relevant items recovered in the top-``k``.

    Differs from :func:`recall_at_k` only for queries with several gold items
    (e.g. i->t on MSCOCO, where each image has five captions).
    """

    if not rankings:
        return 0.0
    total = 0.0
    for ranking, relevant in zip(rankings, relevants):
        gold = _as_set(relevant)
        if not gold:
            continue
        found = sum(1 for item in list(ranking)[:k] if str(item) in gold)
        total += found / len(gold)
    return 100.0 * total / len(rankings)


def mean_reciprocal_rank(rankings: Sequence[Ranking], relevants: Sequence[Relevant]) -> float:
    if not rankings:
        return 0.0
    total = 0.0
    for ranking, relevant in zip(rankings, relevants):
        gold = _as_set(relevant)
        for rank, item in enumerate(ranking, start=1):
            if str(item) in gold:
                total += 1.0 / rank
                break
    return 100.0 * total / len(rankings)


def ndcg_at_k(rankings: Sequence[Ranking], relevants: Sequence[Relevant], k: int) -> float:
    """Binary-relevance nDCG@k, averaged over queries."""

    if not rankings:
        return 0.0
    total = 0.0
    for ranking, relevant in zip(rankings, relevants):
        gold = _as_set(relevant)
        if not gold:
            continue
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, item in enumerate(list(ranking)[:k], start=1)
            if str(item) in gold
        )
        ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(gold)) + 1))
        if ideal > 0:
            total += dcg / ideal
    return 100.0 * total / len(rankings)


def summarize(
    rankings: Sequence[Ranking],
    relevants: Sequence[Relevant],
    ks: Sequence[int] = (1, 5, 10),
    ndcg_ks: Sequence[int] = (10,),
) -> Dict[str, float]:
    """The metric block reported in the paper: R@1/5/10 and nDCG@10."""

    metrics: Dict[str, float] = {}
    for k in ks:
        metrics[f"R@{k}"] = recall_at_k(rankings, relevants, k)
    for k in ndcg_ks:
        metrics[f"nDCG@{k}"] = ndcg_at_k(rankings, relevants, k)
    metrics["MRR"] = mean_reciprocal_rank(rankings, relevants)
    return metrics


def format_metrics(metrics: Dict[str, float], precision: int = 1) -> str:
    return "  ".join(f"{name}={value:.{precision}f}" for name, value in metrics.items())


__all__ = [
    "format_metrics",
    "full_recall_at_k",
    "hit_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "recall_at_k",
    "summarize",
]
