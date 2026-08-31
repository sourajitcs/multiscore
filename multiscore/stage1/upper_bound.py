"""The admissible similarity upper bound U_{q,c}^{(l)} (Eq. 6 / Appendix A.1).

Starting from the exact decomposition

    <x_q^{(L)}, x_c^{(L)}> = <z_q^{(l)}, z_c^{(l)}> + <x_q^{(L)} - z_q^{(l)}, x_c^{(L)} - z_c^{(l)}>,

Cauchy-Schwarz bounds the second (unknown) term by the product of the residual
norms.  Because ``z^{(l)}`` and ``x^{(L)} - z^{(l)}`` are orthogonal and
``||x^{(L)}|| = 1``, that residual norm is itself known in closed form,
``sqrt(1 - ||z^{(l)}||^2)``, giving

    U_{q,c}^{(l)} = <z_q^{(l)}, z_c^{(l)}> + sqrt((1 - ||z_q^{(l)}||^2)(1 - ||z_c^{(l)}||^2))

with the guarantee ``<x_q^{(L)}, x_c^{(L)}> <= U_{q,c}^{(l)}`` at every level.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from multiscore.stage1.mrl import zero_pad


def similarity_upper_bound(
    query_prefix: np.ndarray,
    candidate_prefixes: np.ndarray,
    query_sqnorm: Optional[float] = None,
    candidate_sqnorms: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Vectorised ``U_{q,c}^{(l)}`` for one query against many candidates.

    Parameters
    ----------
    query_prefix:
        ``(m,)`` level-``l`` query prefix, ``m = 2 ** (l - 1) * d``.
    candidate_prefixes:
        ``(n, m)`` level-``l`` candidate prefixes.
    query_sqnorm, candidate_sqnorms:
        Pre-computed squared prefix norms.  Supplying them (as
        :class:`~multiscore.stage1.mrl.MRLEmbeddingStore` does) avoids an
        O(n * m) recomputation per bisection step.

    Returns
    -------
    ``(n,)`` array of upper bounds.
    """

    query_prefix = np.asarray(query_prefix, dtype=np.float32).reshape(-1)
    candidate_prefixes = np.asarray(candidate_prefixes, dtype=np.float32)
    if candidate_prefixes.ndim == 1:
        candidate_prefixes = candidate_prefixes[None, :]
    if candidate_prefixes.shape[1] != query_prefix.shape[0]:
        raise ValueError(
            "query and candidate prefixes must share a level: "
            f"{query_prefix.shape[0]} vs {candidate_prefixes.shape[1]}"
        )

    known = candidate_prefixes @ query_prefix  # <z_q, z_c>

    if query_sqnorm is None:
        query_sqnorm = float(np.dot(query_prefix, query_prefix))
    if candidate_sqnorms is None:
        candidate_sqnorms = np.einsum("nm,nm->n", candidate_prefixes, candidate_prefixes)

    q_residual = np.sqrt(max(0.0, 1.0 - float(query_sqnorm)))
    c_residual = np.sqrt(np.maximum(0.0, 1.0 - np.asarray(candidate_sqnorms, dtype=np.float32)))
    return known + q_residual * c_residual


def similarity_upper_bound_padded(
    query_full: np.ndarray,
    candidates_full: np.ndarray,
    level_dim: int,
) -> np.ndarray:
    """Reference implementation that literally builds the ``z`` vectors of Eq. 4.

    Slower than :func:`similarity_upper_bound` and kept only so the equations in
    the paper can be checked line by line (``tests/test_upper_bound.py`` asserts
    the two agree).
    """

    query_full = np.asarray(query_full, dtype=np.float32).reshape(-1)
    candidates_full = np.asarray(candidates_full, dtype=np.float32)
    full_dim = query_full.shape[0]

    z_q = zero_pad(query_full[:level_dim], full_dim)
    z_c = zero_pad(candidates_full[:, :level_dim], full_dim)

    known = z_c @ z_q
    q_residual = np.linalg.norm(query_full - z_q)
    c_residual = np.linalg.norm(candidates_full - z_c, axis=1)
    return known + q_residual * c_residual
