"""Turn :class:`RetrievalResult` objects into the paper's metric tables."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np

from multiscore.data.schema import Query
from multiscore.eval.metrics import summarize
from multiscore.pipeline import RetrievalResult


def evaluate_results(
    results: Sequence[RetrievalResult],
    queries: Sequence[Query],
    ks: Sequence[int] = (1, 5, 10),
    ndcg_ks: Sequence[int] = (10,),
) -> Dict[str, float]:
    """R@1/5/10 and nDCG@10 over a set of finished retrievals."""

    positives = {q.id: q.positives for q in queries}
    rankings = [r.ranked_ids for r in results]
    relevants = [positives.get(r.query_id, []) for r in results]
    return summarize(rankings, relevants, ks=ks, ndcg_ks=ndcg_ks)


def efficiency_report(results: Sequence[RetrievalResult]) -> Dict[str, float]:
    """Stage-1 cost statistics: speed-up, iterations, level usage.

    Mirrors the online Stage-1 cost table (naive MRL vs. Pyramid Rank).
    """

    stage1 = [r.stage1 for r in results if r.stage1 is not None]
    if not stage1:
        return {}

    costs = np.array([s.cost for s in stage1], dtype=np.float64)
    naive = np.array([s.naive_cost for s in stage1], dtype=np.float64)
    levels = [level for s in stage1 for level in s.level_history]

    return {
        "mean_cost_macs": float(costs.mean()),
        "mean_naive_cost_macs": float(naive.mean()),
        "mean_speedup": float((naive / np.maximum(costs, 1)).mean()),
        "mean_iterations": float(np.mean([s.iterations for s in stage1])),
        "mean_final_level": float(np.mean([s.final_level for s in stage1])),
        "mean_level_used": float(np.mean(levels)) if levels else 0.0,
    }


def level_histogram(results: Sequence[RetrievalResult], num_levels: int = 6) -> Dict[int, int]:
    """Distribution of MRL levels actually used (the dataset-difficulty plot)."""

    histogram = {level: 0 for level in range(1, num_levels + 1)}
    for result in results:
        if result.stage1 is None:
            continue
        for level in result.stage1.level_history:
            histogram[level] = histogram.get(level, 0) + 1
    return histogram


def compare_rankings(
    results_a: Sequence[RetrievalResult],
    results_b: Sequence[RetrievalResult],
    queries: Sequence[Query],
    k: int = 1,
) -> Dict[str, Any]:
    """Head-to-head R@k for two runs (e.g. Pyramid Rank vs. naive MRL)."""

    metrics_a = evaluate_results(results_a, queries, ks=(k,), ndcg_ks=())
    metrics_b = evaluate_results(results_b, queries, ks=(k,), ndcg_ks=())
    return {
        "a": metrics_a,
        "b": metrics_b,
        "delta": {name: metrics_a[name] - metrics_b.get(name, 0.0) for name in metrics_a},
    }
