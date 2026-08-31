from __future__ import annotations

import numpy as np
import pytest

from multiscore.config import MRLConfig
from multiscore.stage1.mrl import MRLEmbeddingStore


@pytest.fixture
def mrl() -> MRLConfig:
    """Small pyramid (d=4, L=4 -> D=32) so tests stay fast but keep the structure."""

    return MRLConfig(base_dim=4, num_levels=4)


@pytest.fixture
def paper_mrl() -> MRLConfig:
    """The geometry actually used in the paper: d=32, L=6, D=1024."""

    return MRLConfig(base_dim=32, num_levels=6)


def make_store(mrl: MRLConfig, n: int = 512, seed: int = 0, decay: float = 1.0) -> MRLEmbeddingStore:
    """Random unit-norm embeddings with front-loaded energy, like real MRL vectors."""

    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, mrl.full_dim)).astype(np.float32)
    raw *= (1.0 / (1.0 + np.arange(mrl.full_dim, dtype=np.float32)) ** decay)
    return MRLEmbeddingStore(raw, mrl=mrl)


@pytest.fixture
def store(mrl: MRLConfig) -> MRLEmbeddingStore:
    return make_store(mrl)


@pytest.fixture
def queries(mrl: MRLConfig) -> np.ndarray:
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((16, mrl.full_dim)).astype(np.float32)
    raw *= (1.0 / (1.0 + np.arange(mrl.full_dim, dtype=np.float32)))
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)
