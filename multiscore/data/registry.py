"""The 12 MMIR tasks / 32 dataset-task pairs evaluated in the paper.

This is a *description* of the benchmark suite -- names, modality signatures,
query/candidate counts and the caption modality each side needs -- not a
downloader.  Use it to drive preprocessing and evaluation loops, and see
``docs/DATA.md`` for how to obtain each corpus (all of them are public but
several require accepting a licence).

Counts are the test-time sizes reported in the paper (``**`` there means the
LSMDC split is distributed under a restricted licence; we leave it as ``None``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

#: The 12 MMIR tasks, written as ``query_modality -> candidate_modality``.
TASKS: Tuple[str, ...] = (
    "t2i",
    "t2t",
    "t2it",
    "t2v",
    "i2t",
    "i2i",
    "it2t",
    "it2i",
    "it2it",
    "v2t",
    "t2a",
    "t2av",
)

#: Task key -> (query modality, candidate modality) in the paper's notation.
TASK_MODALITIES: Dict[str, Tuple[str, str]] = {
    "t2i": ("t", "i"),
    "t2t": ("t", "t"),
    "t2it": ("t", "it"),
    "t2v": ("t", "v"),
    "i2t": ("i", "t"),
    "i2i": ("i", "i"),
    "it2t": ("it", "t"),
    "it2i": ("it", "i"),
    "it2it": ("it", "it"),
    "v2t": ("v", "t"),
    "t2a": ("t", "a"),
    "t2av": ("t", "av"),
}

#: Families used for the alpha default and for grouped reporting.
TASK_FAMILY: Dict[str, str] = {
    "t2i": "image",
    "t2t": "image",
    "t2it": "image",
    "i2t": "image",
    "i2i": "image",
    "it2t": "image",
    "it2i": "image",
    "it2it": "image",
    "t2v": "video",
    "v2t": "video",
    "t2a": "audio",
    "t2av": "audio",
}


@dataclass(frozen=True)
class BenchmarkSpec:
    """One (dataset, task) evaluation unit."""

    dataset: str
    task: str
    num_queries: Optional[int]
    num_candidates: Optional[int]
    collection: str = ""
    notes: str = ""

    @property
    def key(self) -> str:
        return f"{self.dataset}:{self.task}"

    @property
    def query_modality(self) -> str:
        return TASK_MODALITIES[self.task][0]

    @property
    def candidate_modality(self) -> str:
        return TASK_MODALITIES[self.task][1]

    @property
    def family(self) -> str:
        return TASK_FAMILY[self.task]


_SPECS: Tuple[BenchmarkSpec, ...] = (
    # ---- t -> i -------------------------------------------------------- #
    BenchmarkSpec("VisualNews", "t2i", 20_000, 542_000, "M-BEIR"),
    BenchmarkSpec("MSCOCO", "t2i", 24_800, 5_000, "M-BEIR"),
    BenchmarkSpec("Fashion200K", "t2i", 1_700, 201_000, "M-BEIR"),
    BenchmarkSpec("Urban-1K", "t2i", 1_000, 1_000, "Long-CLIP"),
    BenchmarkSpec("Flickr30K", "t2i", 1_000, 5_000, "Flickr30K"),
    # ---- t -> t -------------------------------------------------------- #
    BenchmarkSpec("WebQA", "t2t", 2_400, 544_000, "M-BEIR"),
    # ---- t -> (i,t) ---------------------------------------------------- #
    BenchmarkSpec("EDIS", "t2it", 3_200, 1_000_000, "M-BEIR"),
    BenchmarkSpec("WebQA", "t2it", 2_500, 403_000, "M-BEIR"),
    # ---- t -> v -------------------------------------------------------- #
    BenchmarkSpec("MSRVTT-1kA", "t2v", 1_000, 1_000, "video"),
    BenchmarkSpec("MSVD", "t2v", 26_000, 670, "video"),
    BenchmarkSpec("LSMDC", "t2v", None, None, "video", "restricted licence"),
    BenchmarkSpec("DiDeMo", "t2v", 4_000, 1_000, "video"),
    # ---- i -> t -------------------------------------------------------- #
    BenchmarkSpec("VisualNews", "i2t", 20_000, 537_000, "M-BEIR"),
    BenchmarkSpec("MSCOCO", "i2t", 5_000, 25_000, "M-BEIR"),
    BenchmarkSpec("Fashion200K", "i2t", 4_800, 61_000, "M-BEIR"),
    BenchmarkSpec("Urban-1K", "i2t", 1_000, 1_000, "Long-CLIP"),
    BenchmarkSpec("Flickr30K", "i2t", 5_000, 1_000, "Flickr30K"),
    # ---- i -> i -------------------------------------------------------- #
    BenchmarkSpec("NIGHTS", "i2i", 2_000, 40_000, "M-BEIR"),
    # ---- (i,t) -> t ---------------------------------------------------- #
    BenchmarkSpec("OVEN", "it2t", 50_000, 676_000, "M-BEIR"),
    BenchmarkSpec("InfoSeek", "it2t", 11_000, 611_000, "M-BEIR"),
    # ---- (i,t) -> i ---------------------------------------------------- #
    BenchmarkSpec("FashionIQ", "it2i", 6_000, 74_000, "M-BEIR"),
    BenchmarkSpec("CIRR", "it2i", 4_000, 21_000, "M-BEIR"),
    BenchmarkSpec("GeneCIS", "it2i", 8_000, 15, "GeneCIS", "10-15 candidates per query"),
    # ---- (i,t) -> (i,t) ------------------------------------------------ #
    BenchmarkSpec("OVEN", "it2it", 14_700, 335_000, "M-BEIR"),
    BenchmarkSpec("InfoSeek", "it2it", 17_600, 481_000, "M-BEIR"),
    # ---- v -> t -------------------------------------------------------- #
    BenchmarkSpec("MSRVTT-1kA", "v2t", 1_000, 1_000, "video"),
    BenchmarkSpec("MSVD", "v2t", 670, 26_000, "video"),
    BenchmarkSpec("LSMDC", "v2t", None, None, "video", "restricted licence"),
    BenchmarkSpec("DiDeMo", "v2t", 1_000, 4_000, "video"),
    # ---- t -> a / t -> (a,v) ------------------------------------------- #
    BenchmarkSpec("AudioCaps", "t2a", 4_895, 979, "audio"),
    BenchmarkSpec("Clotho", "t2a", 5_225, 1_045, "audio"),
    BenchmarkSpec("AudioCaps", "t2av", 4_895, 979, "audio"),
)

BENCHMARKS: Dict[str, BenchmarkSpec] = {spec.key: spec for spec in _SPECS}


def list_benchmarks(task: Optional[str] = None, collection: Optional[str] = None) -> List[BenchmarkSpec]:
    specs = list(_SPECS)
    if task:
        specs = [s for s in specs if s.task == task]
    if collection:
        specs = [s for s in specs if s.collection.lower() == collection.lower()]
    return specs


def get_benchmark(key: str) -> BenchmarkSpec:
    """Look up a spec by ``"<dataset>:<task>"``."""

    if key not in BENCHMARKS:
        raise KeyError(f"unknown benchmark '{key}'; try one of {sorted(BENCHMARKS)[:5]} ...")
    return BENCHMARKS[key]


def all_tasks() -> List[str]:
    return list(TASKS)


def all_datasets() -> List[str]:
    seen: List[str] = []
    for spec in _SPECS:
        if spec.dataset not in seen:
            seen.append(spec.dataset)
    return seen


def total_candidates() -> int:
    """~5.7M items when every benchmark database is pooled (universal retrieval)."""

    return sum(spec.num_candidates or 0 for spec in _SPECS)
