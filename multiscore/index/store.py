"""On-disk MRL embedding index.

Layout (one directory per database)::

    <index_dir>/
        embeddings.npy   float32 [N, D], unit-norm level-L vectors
        ids.json         the N candidate ids, in row order
        meta.json        embedder name, MRL geometry, build timestamp

``EmbeddingIndex.load`` memory-maps ``embeddings.npy`` by default, so inspecting
an index is cheap regardless of its size.  Note that ``as_store()`` currently
realises the array in memory: a 5.7M x 1024 float32 index is ~23 GB, so at that
scale either shard the database across workers (as the paper does) or extend
:class:`~multiscore.stage1.mrl.MRLEmbeddingStore` to read level prefixes
directly from the memory map -- Pyramid Rank only ever touches the leading
columns of the surviving rows.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from multiscore.config import MRLConfig
from multiscore.data.schema import _Item
from multiscore.models.base import EmbeddingBackend
from multiscore.stage1.mrl import MRLEmbeddingStore
from multiscore.utils.io import ensure_dir, load_json, write_json
from multiscore.utils.logging import get_logger

LOGGER = get_logger(__name__)

EMBEDDINGS_FILE = "embeddings.npy"
IDS_FILE = "ids.json"
META_FILE = "meta.json"


@dataclass
class EmbeddingIndex:
    """A built index plus the metadata needed to interpret it."""

    embeddings: np.ndarray
    ids: List[str]
    mrl: MRLConfig
    meta: dict

    def __len__(self) -> int:
        return len(self.ids)

    def as_store(self) -> MRLEmbeddingStore:
        """Wrap as an :class:`MRLEmbeddingStore` ready for Pyramid Rank.

        Materialises the embeddings in memory; see the module docstring for the
        large-index caveat.
        """

        return MRLEmbeddingStore(np.asarray(self.embeddings), mrl=self.mrl, ids=self.ids)

    # ------------------------------------------------------------------ #
    def save(self, index_dir: str) -> str:
        ensure_dir(index_dir)
        np.save(os.path.join(index_dir, EMBEDDINGS_FILE), np.asarray(self.embeddings, dtype=np.float32))
        write_json(os.path.join(index_dir, IDS_FILE), self.ids)
        write_json(os.path.join(index_dir, META_FILE), self.meta)
        LOGGER.info("wrote index with %d items to %s", len(self.ids), index_dir)
        return index_dir

    @classmethod
    def load(cls, index_dir: str, mmap: bool = True) -> "EmbeddingIndex":
        embeddings = np.load(
            os.path.join(index_dir, EMBEDDINGS_FILE), mmap_mode="r" if mmap else None
        )
        ids = [str(i) for i in load_json(os.path.join(index_dir, IDS_FILE))]
        meta_path = os.path.join(index_dir, META_FILE)
        meta = load_json(meta_path) if os.path.exists(meta_path) else {}
        mrl_meta = meta.get("mrl", {})
        mrl = MRLConfig(
            base_dim=int(mrl_meta.get("base_dim", 32)),
            num_levels=int(mrl_meta.get("num_levels", 6)),
            normalize=bool(mrl_meta.get("normalize", True)),
        )
        return cls(embeddings=embeddings, ids=ids, mrl=mrl, meta=meta)


def build_index(
    items: Sequence[_Item],
    embedder: EmbeddingBackend,
    mrl: Optional[MRLConfig] = None,
    batch_size: int = 128,
    index_dir: Optional[str] = None,
    is_query: bool = False,
) -> EmbeddingIndex:
    """Encode ``items`` with ``embedder`` and (optionally) persist the result.

    ``items`` must already be captioned -- :meth:`_Item.stage1_text` is what
    gets embedded.
    """

    mrl = mrl or MRLConfig()
    if embedder.full_dim != mrl.full_dim:
        raise ValueError(
            f"embedder emits {embedder.full_dim}-dim vectors but MRL config "
            f"expects D={mrl.full_dim}"
        )

    texts = [item.stage1_text() for item in items]
    empty = sum(1 for text in texts if not text)
    if empty:
        LOGGER.warning("%d/%d items have no Stage-1 text (missing caption?)", empty, len(texts))

    embeddings = embedder.encode(texts, batch_size=batch_size, is_query=is_query)
    index = EmbeddingIndex(
        embeddings=np.asarray(embeddings, dtype=np.float32),
        ids=[item.id for item in items],
        mrl=mrl,
        meta={
            "embedder": getattr(embedder, "name", type(embedder).__name__),
            "num_items": len(items),
            "mrl": {
                "base_dim": mrl.base_dim,
                "num_levels": mrl.num_levels,
                "normalize": mrl.normalize,
            },
            "is_query": is_query,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    if index_dir:
        index.save(index_dir)
    return index
