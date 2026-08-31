"""End-to-end smoke test on the toy corpus with the deterministic stub backends.

This exercises the wiring, not retrieval quality: indexing -> Pyramid Rank ->
CoT/QA re-ranking -> metrics.  Real numbers need the Qwen backbones and the
public benchmarks (see docs/REPRODUCING.md).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from multiscore.config import MRLConfig, MultiScoreConfig, Stage1Config, Stage2Config
from multiscore.data.loaders import load_candidates, load_queries
from multiscore.eval.evaluate import efficiency_report, evaluate_results, level_histogram
from multiscore.models.stub import HashingMRLEmbedder, LexicalMLLM
from multiscore.pipeline import MultiScore

TOY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "toy_data")


@pytest.fixture
def toy():
    queries = load_queries(os.path.join(TOY, "queries.jsonl"))
    candidates = load_candidates(os.path.join(TOY, "candidates.jsonl"))
    return queries, candidates


@pytest.fixture
def engine():
    config = MultiScoreConfig(
        stage1=Stage1Config(top_k=5, epsilon=0.05, mrl=MRLConfig(base_dim=8, num_levels=6)),
        stage2=Stage2Config(alpha=0.6, num_questions=4),
    )
    embedder = HashingMRLEmbedder(dim=config.stage1.mrl.full_dim)
    return MultiScore(config, embedder=embedder, mllm=LexicalMLLM())


def test_toy_corpus_parses(toy):
    queries, candidates = toy
    assert len(candidates) == 24 and len(queries) == 6
    assert all(q.positives for q in queries)
    assert all(c.caption for c in candidates)
    assert candidates[0].modality == "v"


def test_stage1_only_run(engine, toy):
    queries, candidates = toy
    store = engine.index_candidates(candidates)
    results = engine.retrieve_all(queries, store, stage2=False, progress=False)

    assert len(results) == len(queries)
    for result in results:
        assert len(result.ranked_ids) == engine.config.stage1.top_k
        assert set(result.ranked_ids) <= {c.id for c in candidates}
        assert result.stage1 is not None and result.stage1.iterations > 0


def test_full_two_stage_run(engine, toy):
    queries, candidates = toy
    store = engine.index_candidates(candidates)
    results = engine.retrieve_all(queries, store, candidates=candidates, progress=False)

    for result in results:
        assert result.stage2 is not None
        assert len(result.ranked_ids) == len(set(result.ranked_ids))
        assert np.all(np.diff(result.scores) <= 1e-6)
        assert result.explain()


def test_stage2_reranks_within_the_stage1_shortlist(engine, toy):
    queries, candidates = toy
    store = engine.index_candidates(candidates)
    result = engine.retrieve(queries[0], store, candidates=candidates)
    assert set(result.ranked_ids) == set(result.stage1.ids)


def test_metrics_run_over_results(engine, toy):
    queries, candidates = toy
    store = engine.index_candidates(candidates)
    results = engine.retrieve_all(queries, store, candidates=candidates, progress=False)

    metrics = evaluate_results(results, queries)
    assert set(metrics) == {"R@1", "R@5", "R@10", "nDCG@10", "MRR"}
    assert all(0.0 <= v <= 100.0 for v in metrics.values())


def test_efficiency_report_and_histogram(engine, toy):
    queries, candidates = toy
    store = engine.index_candidates(candidates)
    results = engine.retrieve_all(queries, store, stage2=False, progress=False)

    report = efficiency_report(results)
    assert report["mean_naive_cost_macs"] > 0
    histogram = level_histogram(results, num_levels=engine.config.stage1.mrl.num_levels)
    assert sum(histogram.values()) > 0


def test_naive_stage1_matches_exhaustive_scoring(engine, toy):
    queries, candidates = toy
    store = engine.index_candidates(candidates)
    result = engine.retrieve(queries[0], store, stage2=False, naive_stage1=True)
    expected = np.argsort(-(store.embeddings @ engine.encode_query(queries[0])))[: len(result.ranked_ids)]
    assert result.ranked_ids == [store.ids[i] for i in expected]


def test_stage2_requires_candidates(engine, toy):
    queries, candidates = toy
    store = engine.index_candidates(candidates)
    with pytest.raises(ValueError):
        engine.retrieve(queries[0], store, candidates=None, stage2=True)


def test_result_serialisation_roundtrips(engine, toy):
    queries, candidates = toy
    store = engine.index_candidates(candidates)
    payload = engine.retrieve(queries[0], store, candidates=candidates).to_dict()
    assert payload["query_id"] == "q1"
    assert "stage1" in payload and "stage2" in payload
    assert payload["stage1"]["speedup"] > 0
