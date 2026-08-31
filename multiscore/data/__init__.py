"""Dataset schema, loaders and the benchmark registry.

Nothing here downloads data.  MULTI-SCORE consumes a small, uniform JSONL
format (see :mod:`multiscore.data.schema`); ``docs/DATA.md`` explains how to
convert each public benchmark into it.
"""

from multiscore.data.loaders import load_candidates, load_qrels, load_queries
from multiscore.data.registry import (
    BenchmarkSpec,
    all_datasets,
    all_tasks,
    get_benchmark,
    list_benchmarks,
)
from multiscore.data.schema import Candidate, Query, modality_key

__all__ = [
    "BenchmarkSpec",
    "Candidate",
    "Query",
    "all_datasets",
    "all_tasks",
    "get_benchmark",
    "list_benchmarks",
    "load_candidates",
    "load_qrels",
    "load_queries",
    "modality_key",
]
