"""The bound of Eq. 6 must be (a) correct and (b) admissible at every level."""

from __future__ import annotations

import numpy as np
import pytest

from multiscore.stage1.upper_bound import (
    similarity_upper_bound,
    similarity_upper_bound_padded,
)
from tests.conftest import make_store


def test_fast_and_padded_implementations_agree(mrl, queries):
    store = make_store(mrl, n=128)
    for level in range(1, mrl.num_levels + 1):
        dim = mrl.level_dim(level)
        for query in queries[:4]:
            fast = similarity_upper_bound(
                query[:dim],
                store.level(level),
                query_sqnorm=float(np.dot(query[:dim], query[:dim])),
                candidate_sqnorms=store.prefix_sqnorm(level),
            )
            reference = similarity_upper_bound_padded(query, store.embeddings, dim)
            np.testing.assert_allclose(fast, reference, atol=1e-5)


def test_bound_is_admissible_at_every_level(mrl, queries):
    """U^{(l)} >= <x_q^{(L)}, x_c^{(L)}> for all candidates, all levels."""

    store = make_store(mrl, n=256)
    for query in queries[:8]:
        true_similarity = store.embeddings @ query
        for level in range(1, mrl.num_levels + 1):
            dim = mrl.level_dim(level)
            bound = similarity_upper_bound(
                query[:dim], store.level(level), candidate_sqnorms=store.prefix_sqnorm(level)
            )
            assert np.all(bound >= true_similarity - 1e-5)


def test_bound_is_tight_at_the_finest_level(mrl, queries):
    """At l = L the residual term vanishes and the bound becomes the similarity."""

    store = make_store(mrl, n=64)
    level = mrl.num_levels
    dim = mrl.level_dim(level)
    for query in queries[:4]:
        bound = similarity_upper_bound(
            query[:dim], store.level(level), candidate_sqnorms=store.prefix_sqnorm(level)
        )
        np.testing.assert_allclose(bound, store.embeddings @ query, atol=1e-5)


def test_bound_tightens_monotonically_on_average(mrl, queries):
    """Deeper levels give tighter (smaller) bounds."""

    store = make_store(mrl, n=256)
    query = queries[0]
    means = []
    for level in range(1, mrl.num_levels + 1):
        dim = mrl.level_dim(level)
        bound = similarity_upper_bound(
            query[:dim], store.level(level), candidate_sqnorms=store.prefix_sqnorm(level)
        )
        means.append(float(bound.mean()))
    assert all(a >= b - 1e-6 for a, b in zip(means, means[1:]))


def test_level_mismatch_raises(mrl):
    store = make_store(mrl, n=8)
    with pytest.raises(ValueError):
        similarity_upper_bound(np.zeros(mrl.level_dim(1)), store.level(2))
