from __future__ import annotations

import math

from multiscore.eval.metrics import (
    full_recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    summarize,
)


def test_recall_counts_queries_not_items():
    rankings = [["a", "b", "c"], ["x", "y", "z"]]
    relevants = [{"c"}, {"q"}]
    assert recall_at_k(rankings, relevants, 3) == 50.0
    assert recall_at_k(rankings, relevants, 1) == 0.0


def test_recall_is_monotone_in_k():
    rankings = [["a", "b", "c", "d", "e"]]
    relevants = [{"d"}]
    values = [recall_at_k(rankings, relevants, k) for k in (1, 5, 10)]
    assert values == sorted(values)
    assert values[-1] == 100.0


def test_full_recall_handles_multiple_positives():
    rankings = [["a", "b", "c", "d"]]
    relevants = [{"a", "b", "z", "w"}]
    assert full_recall_at_k(rankings, relevants, 4) == 50.0


def test_ndcg_rewards_earlier_hits():
    early = ndcg_at_k([["a", "b", "c"]], [{"a"}], 3)
    late = ndcg_at_k([["b", "c", "a"]], [{"a"}], 3)
    assert early == 100.0
    assert late < early


def test_ndcg_matches_closed_form():
    # single relevant item at rank 3 -> DCG = 1/log2(4), IDCG = 1/log2(2)
    value = ndcg_at_k([["x", "y", "gold"]], [{"gold"}], 10)
    assert math.isclose(value, 100.0 / math.log2(4), rel_tol=1e-6)


def test_mrr():
    assert math.isclose(mean_reciprocal_rank([["a", "gold"]], [{"gold"}]), 50.0)


def test_summarize_reports_the_paper_block():
    metrics = summarize([["gold", "b"]], [{"gold"}])
    assert set(metrics) == {"R@1", "R@5", "R@10", "nDCG@10", "MRR"}
    assert metrics["R@1"] == 100.0


def test_empty_input_is_zero():
    assert recall_at_k([], [], 5) == 0.0
    assert ndcg_at_k([], [], 5) == 0.0
