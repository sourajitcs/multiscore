"""Algorithm 1: admissibility, convergence and eps-bounded correctness."""

from __future__ import annotations

import numpy as np
import pytest

from multiscore.config import Stage1Config
from multiscore.stage1.naive import naive_full_scale_rank
from multiscore.stage1.pyramid_rank import max_iterations, pyramid_rank
from tests.conftest import make_store


def test_returns_exactly_top_k(mrl, queries):
    store = make_store(mrl, n=512)
    output = pyramid_rank(queries[0], store, Stage1Config(top_k=20, epsilon=0.05))
    assert len(output.indices) == 20
    assert len(set(output.indices.tolist())) == 20


def test_scores_are_sorted_descending(mrl, queries):
    store = make_store(mrl, n=512)
    output = pyramid_rank(queries[0], store, Stage1Config(top_k=32, epsilon=0.05))
    assert np.all(np.diff(output.scores) <= 1e-6)


def test_pruning_is_admissible(mrl, queries):
    """Nothing is pruned above the terminal threshold.

    A candidate leaves the index set only when ``U^{(l)} < tau``, and since the
    bound is admissible its true level-``L`` similarity is below that ``tau``,
    which never exceeds the terminal ``tau_min``.  This holds level by level and
    is the property the appendix calls *admissibility*.
    """

    store = make_store(mrl, n=512)
    for query in queries[:8]:
        output = pyramid_rank(query, store, Stage1Config(top_k=20, epsilon=0.02))
        true_similarity = store.embeddings @ query
        pruned = np.setdiff1d(np.arange(len(store)), output.survivor_indices)
        if pruned.size:
            assert np.all(true_similarity[pruned] < output.tau_min + 1e-5)


def test_epsilon_bound_is_exact_when_bounds_are_exact(queries):
    """With a single level the bound *is* the similarity, so Appendix A.2 is tight.

    Any discarded item ``j`` must then satisfy
    ``<q, c_j> <= <q, c_K> + eps`` with no slack at all.
    """

    from multiscore.config import MRLConfig

    single = MRLConfig(base_dim=32, num_levels=1)
    store = make_store(single, n=512, seed=11)
    rng = np.random.default_rng(3)
    epsilon = 0.05
    for _ in range(8):
        query = rng.standard_normal(single.full_dim).astype(np.float32)
        query /= np.linalg.norm(query)
        output = pyramid_rank(query, store, Stage1Config(top_k=20, epsilon=epsilon))
        true_similarity = store.embeddings @ query
        worst_retrieved = float(true_similarity[output.indices].min())
        discarded = np.setdiff1d(np.arange(len(store)), output.indices)
        assert np.all(true_similarity[discarded] <= worst_retrieved + epsilon + 1e-5)


def test_epsilon_bounded_correctness(paper_mrl):
    """The same bound on the pyramid geometry the paper actually uses (d=32, L=6).

    Here the retained bounds come from several levels at once, so the guarantee
    relies on the final level-``L`` re-scoring of the surviving set
    (``Stage1Config.final_rescore``).
    """

    store = make_store(paper_mrl, n=4096, seed=3)
    rng = np.random.default_rng(5)
    epsilon = 0.02
    for _ in range(8):
        query = rng.standard_normal(paper_mrl.full_dim).astype(np.float32)
        query *= 1.0 / (1.0 + np.arange(paper_mrl.full_dim, dtype=np.float32))
        query /= np.linalg.norm(query)
        output = pyramid_rank(query, store, Stage1Config(top_k=100, epsilon=epsilon))
        true_similarity = store.embeddings @ query
        worst_retrieved = float(true_similarity[output.indices].min())
        discarded = np.setdiff1d(np.arange(len(store)), output.indices)
        assert np.all(true_similarity[discarded] <= worst_retrieved + epsilon + 1e-4)


def test_converges_within_the_bisection_bound(mrl, queries):
    store = make_store(mrl, n=512)
    for epsilon in (0.4, 0.1, 0.02):
        output = pyramid_rank(queries[0], store, Stage1Config(top_k=16, epsilon=epsilon))
        assert output.iterations <= max_iterations(-1.0, 1.0, epsilon)


def test_convergence_is_independent_of_database_size(mrl, queries):
    iterations = set()
    for n in (256, 1024, 4096):
        store = make_store(mrl, n=n, seed=n)
        iterations.add(
            pyramid_rank(queries[0], store, Stage1Config(top_k=16, epsilon=0.02)).iterations
        )
    assert max(iterations) <= max_iterations(-1.0, 1.0, 0.02)


def test_cheaper_than_naive_full_resolution(paper_mrl):
    """On the paper's geometry and a database worth filtering, MACs go down."""

    store = make_store(paper_mrl, n=20_000, seed=3, decay=0.5)
    rng = np.random.default_rng(5)
    queries = rng.standard_normal((8, paper_mrl.full_dim)).astype(np.float32)
    queries *= 1.0 / (1.0 + np.arange(paper_mrl.full_dim, dtype=np.float32)) ** 0.5
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    config = Stage1Config(top_k=100, epsilon=0.02, mrl=paper_mrl)
    outputs = [pyramid_rank(q, store, config) for q in queries]
    assert all(o.cost < o.naive_cost for o in outputs)
    assert np.mean([o.speedup for o in outputs]) > 1.5


def test_filtering_only_pays_off_when_the_database_is_large(mrl, queries):
    """Sanity check on the regime, so the cost model is not over-claimed.

    With ``K`` a large fraction of ``N`` and a narrow pyramid there is nothing
    to filter, and the bisection can cost *more* than scoring everything once.
    Pyramid Rank targets the opposite regime (``N`` in the millions, ``K = 100``).
    """

    store = make_store(mrl, n=4096)
    outputs = [pyramid_rank(q, store, Stage1Config(top_k=50, epsilon=0.02)) for q in queries[:4]]
    assert all(o.cost > 0 for o in outputs)
    assert np.mean([o.speedup for o in outputs]) < 1.0


def test_recovers_naive_top_k_for_a_tight_epsilon(mrl, queries):
    """With a small eps, Pyramid Rank's shortlist covers the exact top-K."""

    store = make_store(mrl, n=1024)
    k = 25
    overlaps = []
    for query in queries[:8]:
        approx = pyramid_rank(query, store, Stage1Config(top_k=k, epsilon=1e-3))
        exact = naive_full_scale_rank(query, store, top_k=k)
        overlaps.append(len(set(approx.indices.tolist()) & set(exact.indices.tolist())) / k)
    assert np.mean(overlaps) >= 0.8


def test_final_rescore_returns_exact_similarities(mrl, queries):
    store = make_store(mrl, n=1024)
    output = pyramid_rank(queries[0], store, Stage1Config(top_k=20, epsilon=0.02))
    assert output.rescored
    np.testing.assert_allclose(
        output.scores, (store.embeddings @ queries[0])[output.indices], atol=1e-5
    )


def test_faithful_mode_ranks_by_upper_bounds(mrl, queries):
    """`final_rescore=False` reproduces Algorithm 1 line for line."""

    store = make_store(mrl, n=1024)
    output = pyramid_rank(
        queries[0], store, Stage1Config(top_k=20, epsilon=0.02, final_rescore=False)
    )
    assert not output.rescored
    exact = (store.embeddings @ queries[0])[output.indices]
    assert np.all(output.scores >= exact - 1e-5)  # they are upper bounds


def test_exact_padding_matches_fast_path(mrl, queries):
    store = make_store(mrl, n=256)
    fast = pyramid_rank(queries[0], store, Stage1Config(top_k=10, epsilon=0.05))
    slow = pyramid_rank(
        queries[0], store, Stage1Config(top_k=10, epsilon=0.05, exact_padding=True)
    )
    np.testing.assert_array_equal(fast.indices, slow.indices)


def test_small_database_falls_back_to_exhaustive(mrl, queries):
    store = make_store(mrl, n=8)
    output = pyramid_rank(queries[0], store, Stage1Config(top_k=16, epsilon=0.02))
    assert len(output.indices) == 8
    assert output.final_level == mrl.num_levels


def test_levels_only_ever_increase(mrl, queries):
    store = make_store(mrl, n=2048)
    output = pyramid_rank(queries[0], store, Stage1Config(top_k=32, epsilon=0.01))
    assert output.level_history == sorted(output.level_history)
    assert max(output.level_history) <= mrl.num_levels


def test_invalid_arguments(mrl, queries):
    store = make_store(mrl, n=32)
    with pytest.raises(ValueError):
        pyramid_rank(queries[0], store, Stage1Config(top_k=0))
    with pytest.raises(ValueError):
        pyramid_rank(np.zeros(3), store, Stage1Config())
