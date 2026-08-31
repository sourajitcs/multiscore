"""Matryoshka (MRL) representation utilities.

Notation follows Section 3.1 of the paper:

* ``x_c^{(L)}`` is the unit-norm level-``L`` (finest) embedding of candidate ``c``;
* ``x_c^{(l)} = x_c^{(L)}[: 2 ** (l - 1) * d]`` is its nested level-``l`` prefix (Eq. 2);
* ``z_c^{(l)}`` is ``x_c^{(l)}`` zero-padded back to ``D`` dimensions (Eq. 4).

Because ``z`` only ever appears inside inner products and norms, the padding is
mathematically inert: ``<z_q^{(l)}, z_c^{(l)}>`` equals the inner product of the
two length-``2 ** (l - 1) * d`` prefixes, and ``||z_c^{(l)}||`` equals the prefix
norm.  ``MRLEmbeddingStore`` exploits this by caching cumulative prefix norms,
which is what makes Pyramid Rank cheap in practice.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from multiscore.config import MRLConfig


def level_slice(embeddings: np.ndarray, dim: int) -> np.ndarray:
    """Return the nested level prefix of ``embeddings`` (Eq. 2)."""

    if dim > embeddings.shape[-1]:
        raise ValueError(
            f"requested prefix of {dim} dims from {embeddings.shape[-1]}-dim vectors"
        )
    return embeddings[..., :dim]


def zero_pad(embeddings: np.ndarray, full_dim: int) -> np.ndarray:
    """Zero-pad a level prefix back to ``full_dim`` dimensions (Eq. 4)."""

    prefix_dim = embeddings.shape[-1]
    if prefix_dim == full_dim:
        return embeddings
    pad_width = [(0, 0)] * (embeddings.ndim - 1) + [(0, full_dim - prefix_dim)]
    return np.pad(embeddings, pad_width, mode="constant", constant_values=0.0)


def l2_normalize(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation, enforcing the unit-norm assumption of Eq. 3."""

    norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    return embeddings / np.maximum(norms, eps)


class MRLEmbeddingStore:
    """A frozen matrix of level-``L`` MRL embeddings plus cached prefix norms.

    Parameters
    ----------
    embeddings:
        ``(N, D)`` array of level-``L`` embeddings.  Normalised on construction
        unless ``mrl.normalize`` is ``False``.
    mrl:
        Pyramid geometry.  ``mrl.full_dim`` must match ``D``.
    ids:
        Optional external identifiers, aligned with the rows of ``embeddings``.
    cache_prefix_dim:
        Levels this narrow or narrower get their own contiguous copy.  Row-major
        storage means that reading a 32-column prefix out of a ``D``-wide matrix
        still touches every cache line, so without these copies a level-1 pass
        costs the same memory traffic as a full-resolution one -- the arithmetic
        saving would not show up as wall-clock.  The default caches levels up to
        128 dimensions, which is where most candidates are eliminated, for about
        22% extra memory at ``d = 32, D = 1024``.  Set to ``0`` to disable.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        mrl: Optional[MRLConfig] = None,
        ids: Optional[Sequence[str]] = None,
        cache_prefix_dim: int = 128,
    ) -> None:
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"expected a 2-D array, got shape {embeddings.shape}")

        self.mrl = mrl or MRLConfig()
        if embeddings.shape[1] != self.mrl.full_dim:
            raise ValueError(
                f"embedding dim {embeddings.shape[1]} != MRL full_dim {self.mrl.full_dim}"
            )
        if self.mrl.normalize:
            embeddings = l2_normalize(embeddings)

        self.embeddings = embeddings
        self.ids = list(ids) if ids is not None else [str(i) for i in range(len(embeddings))]
        if len(self.ids) != len(embeddings):
            raise ValueError("ids and embeddings must have the same length")

        # _prefix_sqnorms[j] = ||z^{(j + 1)}||_2^2 for every candidate -- computed
        # once, reused by every query and every bisection step.  Stored one
        # contiguous row per level so that gathering a subset is a fast take.
        cumulative = np.cumsum(np.square(embeddings, dtype=np.float64), axis=1)
        self._prefix_sqnorms = np.ascontiguousarray(
            np.stack([cumulative[:, dim - 1] for dim in self.mrl.level_dims], axis=0),
            dtype=np.float32,
        )

        self._prefix_cache: Dict[int, np.ndarray] = {}
        for level_index, dim in enumerate(self.mrl.level_dims, start=1):
            if 0 < dim <= cache_prefix_dim < self.mrl.full_dim:
                self._prefix_cache[level_index] = np.ascontiguousarray(embeddings[:, :dim])

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self.embeddings.shape[0]

    @property
    def num_levels(self) -> int:
        return self.mrl.num_levels

    @property
    def prefix_sqnorms(self) -> np.ndarray:
        """``(N, L)`` view of ``||z^{(l)}||_2^2``, one column per level."""

        return self._prefix_sqnorms.T

    def level(self, level: int, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Level-``level`` prefixes, optionally restricted to ``indices``.

        Returns a contiguous array whenever it can: the cached copy for a cached
        level, a gather out of it for a subset, and a gather out of the full
        matrix otherwise.
        """

        dim = self.mrl.level_dim(level)
        cached = self._prefix_cache.get(level)
        selects_all = indices is None or indices.size == self.embeddings.shape[0]

        if cached is not None:
            return cached if selects_all else np.take(cached, indices, axis=0)
        if selects_all:
            return self.embeddings[:, :dim]
        return np.ascontiguousarray(np.take(self.embeddings, indices, axis=0)[:, :dim])

    def prefix_sqnorm(self, level: int, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """``||z^{(level)}||_2^2`` for the requested rows (cached, O(1) lookup)."""

        row = self._prefix_sqnorms[level - 1]
        if indices is None or indices.size == self.embeddings.shape[0]:
            return row
        return np.take(row, indices)

    def residual_norm(self, level: int, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """``||x^{(L)} - z^{(level)}||_2 = sqrt(1 - ||z^{(level)}||^2)`` (Eq. 15)."""

        return np.sqrt(np.maximum(0.0, 1.0 - self.prefix_sqnorm(level, indices)))

    @classmethod
    def from_file(cls, path: str, mrl: Optional[MRLConfig] = None) -> "MRLEmbeddingStore":
        """Load a store written by :meth:`save` (``.npz`` with ``embeddings``/``ids``)."""

        with np.load(path, allow_pickle=False) as payload:
            embeddings = payload["embeddings"]
            ids = [str(i) for i in payload["ids"]] if "ids" in payload else None
        return cls(embeddings, mrl=mrl, ids=ids)

    def save(self, path: str) -> None:
        np.savez(path, embeddings=self.embeddings, ids=np.array(self.ids, dtype=object).astype(str))
