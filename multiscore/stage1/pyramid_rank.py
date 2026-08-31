"""Pyramid Rank -- Stage-1 candidate filtering (Algorithm 1 in the paper).

The algorithm bisects a relevance threshold ``tau`` over ``[tau_min, tau_max]``
while walking up the Matryoshka pyramid.  At each step it keeps only the
candidates whose *upper bound* on the level-``L`` similarity clears ``tau``:

* **Admissibility** -- a candidate is dropped only when
  ``U_{q,c}^{(l)} < tau``, and since ``<x_q^{(L)}, x_c^{(L)}> <= U_{q,c}^{(l)}``
  no candidate's true relevance is ever underestimated.
* **Convergence** -- each iteration halves the width of the threshold interval,
  so the loop terminates in at most ``ceil(log2((tau_max - tau_min) / eps))``
  steps, independent of the database size ``N``.
* **eps-bounded correctness** -- at termination any discarded candidate ``j``
  satisfies ``<x_q, x_{c_j}> <= <x_q, x_{c_K}> + eps``.

Only the surviving set is ever promoted to a finer (longer) level, which is
where the speed-up over full-resolution scoring comes from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from multiscore.config import Stage1Config
from multiscore.stage1.mrl import MRLEmbeddingStore
from multiscore.stage1.upper_bound import (
    similarity_upper_bound,
    similarity_upper_bound_padded,
)


@dataclass
class Stage1Output:
    """Result of one Pyramid Rank call.

    Attributes
    ----------
    indices:
        Row indices of the top-``K`` candidates, sorted by decreasing bound.
    scores:
        Their similarity upper bounds ``U_q`` (the Stage-1 ranking signal).
    ids:
        External identifiers of ``indices``, when the store carries them.
    iterations:
        Number of bisection steps actually executed.
    final_level:
        Deepest MRL level reached for this query.
    level_history:
        Level used at each iteration -- reproduces the level histogram of the
        efficiency analysis.
    survivors_history:
        ``|I|`` after each iteration.
    cost:
        Multiply-accumulate operations spent on similarity computation.
        Counts every row each level pass touched -- including rows already
        pruned but not yet compacted away -- plus the final level-``L``
        re-scoring of the survivors.
    naive_cost:
        ``N * D``, the cost of scoring the whole database at level ``L``.
    tau_min, tau_max:
        Terminal threshold interval, ``tau_max - tau_min <= eps``.
    num_survivors:
        ``|I|`` at termination, before truncation to ``K``.
    survivor_indices:
        The surviving index set ``I`` itself.  Everything outside it was pruned,
        and every pruned candidate provably scores below ``tau_min``.
    rescored:
        Whether the returned ``scores`` are exact level-``L`` similarities
        (``final_rescore``) rather than mixed-level upper bounds.
    """

    indices: np.ndarray
    scores: np.ndarray
    ids: List[str] = field(default_factory=list)
    iterations: int = 0
    final_level: int = 1
    level_history: List[int] = field(default_factory=list)
    survivors_history: List[int] = field(default_factory=list)
    cost: int = 0
    naive_cost: int = 0
    tau_min: float = -1.0
    tau_max: float = 1.0
    num_survivors: int = 0
    survivor_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    rescored: bool = False

    @property
    def speedup(self) -> float:
        """Cost reduction against naive full-resolution MRL scoring."""

        return float(self.naive_cost) / float(max(1, self.cost))

    def level_histogram(self, num_levels: int) -> Dict[int, int]:
        return {
            level: int(sum(1 for used in self.level_history if used == level))
            for level in range(1, num_levels + 1)
        }


def max_iterations(tau_min: float, tau_max: float, epsilon: float) -> int:
    """``ceil(log2(w0 / eps))`` -- the convergence bound of Section 3.1."""

    width = float(tau_max) - float(tau_min)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if width <= epsilon:
        return 0
    return int(math.ceil(math.log2(width / epsilon)))


class _Working:
    """Contiguous view of the candidates still in contention.

    Pyramid Rank shrinks its index set, but *gathering* those rows out of the
    database is random-access memory traffic and is easily more expensive than
    the arithmetic it saves.  So while the surviving set is still a sizeable
    fraction of the database we scan the whole contiguous prefix matrix and mask
    afterwards -- the wasted work is bounded and sequential -- and only compact
    into a smaller buffer once the set has genuinely collapsed.

    ``scored_rows`` records how many rows each level pass actually touched, so
    the reported cost stays honest about that trade-off.
    """

    __slots__ = ("mrl", "ids", "embeddings", "_sqnorms", "_prefix", "_store")

    def __init__(self, store: MRLEmbeddingStore) -> None:
        self.mrl = store.mrl
        self.ids = np.arange(len(store), dtype=np.int64)
        self.embeddings = store.embeddings
        self._store = store
        self._sqnorms = None
        self._prefix = {}

    @property
    def size(self) -> int:
        return int(self.ids.size)

    def prefix(self, level: int) -> np.ndarray:
        """Contiguous level-``level`` prefixes for every row in the buffer."""

        if self._store is not None:
            return self._store.level(level)
        cached = self._prefix.get(level)
        if cached is None:
            dim = self.mrl.level_dim(level)
            cached = (
                self.embeddings
                if dim == self.mrl.full_dim
                else np.ascontiguousarray(self.embeddings[:, :dim])
            )
            self._prefix[level] = cached
        return cached

    def sqnorm(self, level: int) -> np.ndarray:
        if self._store is not None:
            return self._store.prefix_sqnorm(level)
        return self._sqnorms[level - 1]

    def compact(self, keep: np.ndarray, store: MRLEmbeddingStore) -> "_Working":
        """Gather the rows selected by the boolean mask ``keep`` into a new buffer."""

        compacted = _Working.__new__(_Working)
        compacted.mrl = self.mrl
        compacted.ids = self.ids[keep]
        compacted.embeddings = np.ascontiguousarray(
            np.take(store.embeddings, compacted.ids, axis=0)
        )
        compacted._sqnorms = np.ascontiguousarray(
            np.take(store.prefix_sqnorms.T, compacted.ids, axis=1)
        )
        compacted._prefix = {}
        compacted._store = None
        return compacted


def pyramid_rank(
    query_embedding: np.ndarray,
    store: MRLEmbeddingStore,
    config: Optional[Stage1Config] = None,
    top_k: Optional[int] = None,
    epsilon: Optional[float] = None,
    compact_fraction: float = 0.05,
) -> Stage1Output:
    """Run Algorithm 1 for a single query.

    Parameters
    ----------
    query_embedding:
        ``(D,)`` level-``L`` query embedding.  Assumed unit-norm (Eq. 3); it is
        re-normalised defensively.
    store:
        Candidate database as an :class:`MRLEmbeddingStore`.
    config:
        Stage-1 hyper-parameters; ``top_k``/``epsilon`` override the config.
    compact_fraction:
        Implementation detail with no effect on the returned ranking: the
        surviving candidates are gathered into a smaller buffer once they fall
        below this fraction of the current one.  See :class:`_Working`.
    """

    config = config or Stage1Config()
    mrl = store.mrl
    k = int(top_k if top_k is not None else config.top_k)
    eps = float(epsilon if epsilon is not None else config.epsilon)
    if k <= 0:
        raise ValueError("top_k must be positive")

    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    if query.shape[0] != mrl.full_dim:
        raise ValueError(
            f"query dim {query.shape[0]} != MRL full_dim {mrl.full_dim}"
        )
    if mrl.normalize:
        norm = float(np.linalg.norm(query))
        if norm > 0:
            query = query / norm

    num_candidates = len(store)
    # ---- Init: tau_min = -1, tau_max = 1, l = 1, I = {1..N}, U_q = zeros(N) ----
    tau_min, tau_max = float(config.tau_min), float(config.tau_max)
    level = 1
    # The paper initialises U_q to zeros; we use -inf so that candidates which
    # never survived a filtering step can never outrank one that did (identical
    # behaviour whenever the surviving bounds are non-negative).
    bounds = np.full(num_candidates, -np.inf, dtype=np.float32)

    output = Stage1Output(
        indices=np.empty(0, dtype=np.int64),
        scores=np.empty(0, dtype=np.float32),
        naive_cost=num_candidates * mrl.full_dim,
    )

    if num_candidates <= k:
        # Nothing to filter: score everything at level L and return.
        full_scores = store.embeddings @ query
        order = np.argsort(-full_scores, kind="stable")
        output.indices = order.astype(np.int64)
        output.scores = full_scores[order].astype(np.float32)
        output.ids = [store.ids[i] for i in output.indices]
        output.cost = output.naive_cost
        output.final_level = mrl.num_levels
        output.num_survivors = num_candidates
        output.survivor_indices = np.arange(num_candidates, dtype=np.int64)
        output.rescored = True
        return output

    working = _Working(store)
    alive = np.ones(working.size, dtype=bool)
    index_set = working.ids

    while tau_max - tau_min > eps:
        tau = (tau_min + tau_max) / 2.0
        level_dim = mrl.level_dim(level)

        # ---- U_{q,c_i}^{(l)} over the working buffer (Eq. 6) ----
        if config.exact_padding:
            step_bounds = similarity_upper_bound_padded(
                query, working.embeddings, level_dim
            )
        else:
            step_bounds = similarity_upper_bound(
                query[:level_dim],
                working.prefix(level),
                query_sqnorm=float(np.dot(query[:level_dim], query[:level_dim])),
                candidate_sqnorms=working.sqnorm(level),
            )

        output.cost += working.size * int(level_dim)
        output.iterations += 1
        output.level_history.append(level)

        keep = alive & (step_bounds >= tau)
        num_survivors = int(np.count_nonzero(keep))

        if num_survivors >= k:
            tau_min = tau  # tighten
            if level < mrl.num_levels:
                level += 1
            alive = keep
            index_set = working.ids[alive]
            bounds[index_set] = step_bounds[alive]

            if num_survivors <= compact_fraction * working.size:
                working = working.compact(alive, store)
                alive = np.ones(working.size, dtype=bool)
        else:
            tau_max = tau  # loosen

        output.survivors_history.append(int(index_set.size))

    output.final_level = level
    output.tau_min, output.tau_max = tau_min, tau_max
    output.num_survivors = int(index_set.size)
    output.survivor_indices = index_set.astype(np.int64)

    if config.final_rescore:
        # U^{(L)} for the survivors only: exact similarity, |I| * D MACs.
        exact = np.take(store.embeddings, index_set, axis=0) @ query
        bounds[index_set] = exact.astype(np.float32)
        output.cost += int(index_set.size) * int(mrl.full_dim)
        output.rescored = True

    order = index_set[np.argsort(-bounds[index_set], kind="stable")][:k]
    output.indices = order.astype(np.int64)
    output.scores = bounds[order].astype(np.float32)
    output.ids = [store.ids[i] for i in output.indices]
    return output


def pyramid_rank_batch(
    query_embeddings: np.ndarray,
    store: MRLEmbeddingStore,
    config: Optional[Stage1Config] = None,
    top_k: Optional[int] = None,
    epsilon: Optional[float] = None,
    progress: bool = False,
) -> List[Stage1Output]:
    """Apply :func:`pyramid_rank` to each row of ``query_embeddings``.

    Pyramid Rank is query-adaptive -- the level at which a candidate is scored
    depends on the query -- so queries are processed independently.  In the
    paper's deployment they are sharded across GPUs; here they are simply looped.
    """

    query_embeddings = np.atleast_2d(np.asarray(query_embeddings, dtype=np.float32))
    iterator: Sequence[int] = range(query_embeddings.shape[0])
    if progress:
        try:
            from tqdm.auto import tqdm  # type: ignore

            iterator = tqdm(iterator, desc="stage-1", unit="query")
        except ImportError:
            pass
    return [
        pyramid_rank(query_embeddings[i], store, config=config, top_k=top_k, epsilon=epsilon)
        for i in iterator
    ]
