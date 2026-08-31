from __future__ import annotations

import os

import pytest

from multiscore.config import MRLConfig
from multiscore.data.registry import (
    BENCHMARKS,
    TASKS,
    all_datasets,
    get_benchmark,
    list_benchmarks,
    total_candidates,
)
from multiscore.data.schema import Candidate, Query, modality_key
from multiscore.index.store import EmbeddingIndex, build_index
from multiscore.models.stub import HashingMRLEmbedder
from multiscore.utils.io import read_jsonl, write_jsonl

TOY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "toy_data")


def test_registry_matches_the_paper_scale():
    assert len(TASKS) == 12
    assert len(BENCHMARKS) == 32
    assert 5.0e6 < total_candidates() < 6.5e6  # ~5.7M pooled items
    assert "MSCOCO" in all_datasets()


def test_every_benchmark_has_a_known_task():
    for spec in list_benchmarks():
        assert spec.task in TASKS
        assert spec.family in {"image", "video", "audio"}
        assert spec.query_modality and spec.candidate_modality


def test_get_benchmark_rejects_unknown_keys():
    assert get_benchmark("MSCOCO:t2i").collection == "M-BEIR"
    with pytest.raises(KeyError):
        get_benchmark("NotADataset:t2i")


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"id": "1", "text": "hi"}, "t"),
        ({"id": "2", "image": "a.jpg"}, "i"),
        ({"id": "3", "image": "a.jpg", "text": "hi"}, "it"),
        ({"id": "4", "video": "a.mp4"}, "v"),
        ({"id": "5", "audio": "a.wav"}, "a"),
        ({"id": "6", "audio": "a.wav", "video": "a.mp4"}, "av"),
    ],
)
def test_modality_key(payload, expected):
    assert modality_key(Candidate.from_dict(payload)) == expected


def test_query_roundtrip_keeps_positives_and_extras():
    query = Query.from_dict({"id": "q", "text": "t", "positives": ["a", "b"], "split": "test"})
    assert query.positives == ["a", "b"]
    assert query.meta["split"] == "test"
    assert Query.from_dict(query.to_dict()).positives == ["a", "b"]


def test_stage1_text_prefers_caption():
    candidate = Candidate.from_dict({"id": "c", "text": "raw", "caption": "cap"})
    assert candidate.stage1_text() == "cap raw"
    assert candidate.media() == []


def test_jsonl_roundtrip(tmp_path):
    rows = list(read_jsonl(os.path.join(TOY, "candidates.jsonl")))
    path = str(tmp_path / "out.jsonl")
    assert write_jsonl(path, rows) == len(rows)
    assert list(read_jsonl(path)) == rows


def test_index_build_save_load(tmp_path):
    from multiscore.data.loaders import load_candidates

    mrl = MRLConfig(base_dim=8, num_levels=6)
    candidates = load_candidates(os.path.join(TOY, "candidates.jsonl"))
    index = build_index(
        candidates, HashingMRLEmbedder(dim=mrl.full_dim), mrl=mrl, index_dir=str(tmp_path / "idx")
    )
    reloaded = EmbeddingIndex.load(str(tmp_path / "idx"))

    assert len(reloaded) == len(candidates)
    assert reloaded.ids == index.ids
    assert reloaded.as_store().embeddings.shape == (len(candidates), mrl.full_dim)
    assert reloaded.meta["embedder"] == "stub-mrl"


def test_index_rejects_dimension_mismatch():
    from multiscore.data.loaders import load_candidates

    candidates = load_candidates(os.path.join(TOY, "candidates.jsonl"), limit=2)
    with pytest.raises(ValueError):
        build_index(candidates, HashingMRLEmbedder(dim=64), mrl=MRLConfig(base_dim=32, num_levels=6))
