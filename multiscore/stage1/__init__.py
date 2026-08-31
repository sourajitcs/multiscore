"""Stage-1: efficient candidate filtering with multi-scale MRL representations."""

from multiscore.stage1.mrl import MRLEmbeddingStore, level_slice, zero_pad
from multiscore.stage1.naive import naive_full_scale_rank
from multiscore.stage1.pyramid_rank import Stage1Output, pyramid_rank
from multiscore.stage1.upper_bound import (
    similarity_upper_bound,
    similarity_upper_bound_padded,
)

__all__ = [
    "MRLEmbeddingStore",
    "Stage1Output",
    "level_slice",
    "naive_full_scale_rank",
    "pyramid_rank",
    "similarity_upper_bound",
    "similarity_upper_bound_padded",
    "zero_pad",
]
