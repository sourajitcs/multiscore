"""Configuration objects for the MULTI-SCORE pipeline.

Every knob that appears in the paper has a home here.  Configs can be built in
Python or loaded from the YAML files under ``configs/``.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MRLConfig:
    """Geometry of the Matryoshka representation pyramid.

    Defaults follow the paper: Qwen3-MRL 0.6B exposes ``L = 6`` nested levels,
    the smallest being ``d = 32`` (level 1) and the largest ``D = 1024``
    (level 6), with ``dim(level) = 2 ** (level - 1) * d``.
    """

    base_dim: int = 32  # d, level 1
    num_levels: int = 6  # L
    normalize: bool = True  # unit-norm level-L vectors (Eq. 3)

    @property
    def full_dim(self) -> int:
        """D, the dimensionality of the level-L (finest) representation."""
        return self.base_dim * 2 ** (self.num_levels - 1)

    def level_dim(self, level: int) -> int:
        """Length of a level-``level`` vector, ``2 ** (level - 1) * d`` (Eq. 1)."""
        if not 1 <= level <= self.num_levels:
            raise ValueError(
                f"level must be in [1, {self.num_levels}], got {level}"
            )
        return self.base_dim * 2 ** (level - 1)

    @property
    def level_dims(self) -> List[int]:
        return [self.level_dim(level) for level in range(1, self.num_levels + 1)]


@dataclass
class Stage1Config:
    """Pyramid Rank (Algorithm 1) settings."""

    top_k: int = 100  # K, candidate budget handed to Stage-2
    epsilon: float = 0.02  # tolerance; optimal value reported in the paper
    tau_min: float = -1.0
    tau_max: float = 1.0
    mrl: MRLConfig = field(default_factory=MRLConfig)
    # Reference implementation materialises the zero-padded vectors of Eq. 4;
    # the fast path uses prefix slices and cached prefix norms instead.  Both
    # produce identical numbers up to float error.
    exact_padding: bool = False
    # Re-score the surviving set at level L before the final sort.  Algorithm 1
    # ranks by the upper bounds U_q, which are collected at whatever level each
    # candidate was last evaluated at; the eps-bounded correctness argument in
    # the appendix is written in terms of U^{(L)}.  Re-scoring restores that
    # condition for the returned ranking and costs only |I| * D MACs, where |I|
    # is the (small) surviving set.  Set to False to reproduce Algorithm 1
    # line for line.
    final_rescore: bool = True
    embedder: str = "qwen3-mrl"  # see multiscore.models.registry


@dataclass
class Stage2Config:
    """Re-ranking settings (Bidirectional-CoT + QA relevance)."""

    backend: str = "qwen2.5-omni-7b"  # MLLM used for both scores
    alpha: float = 0.6  # convex weight on S_CoT in Eq. 9
    num_questions: int = 7  # M yes/no questions per query
    num_cot_examples: int = 2  # in-context CoT demonstrations
    emb_token: str = "<emb>"  # marker token; hidden state *before* it is pooled
    max_answer_tokens: int = 5
    temperature: float = 0.0
    cot_batch_size: int = 8
    qa_batch_size: int = 4
    enable_cot: bool = True
    enable_qa: bool = True


@dataclass
class MultiScoreConfig:
    """Top-level config: Stage-1 + Stage-2 + bookkeeping."""

    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    task: str = "t2i"
    dataset: Optional[str] = None
    # Optional data paths, so a config can be self-contained.
    queries: Optional[str] = None
    candidates: Optional[str] = None
    qrels: Optional[str] = None
    index_dir: str = "runs/index"
    output_dir: str = "runs/latest"
    seed: int = 0
    device: str = "auto"
    dtype: str = "bfloat16"

    # ------------------------------------------------------------------ #
    # (de)serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MultiScoreConfig":
        payload = dict(payload or {})
        stage1 = dict(payload.pop("stage1", {}) or {})
        stage2 = dict(payload.pop("stage2", {}) or {})
        mrl = dict(stage1.pop("mrl", {}) or {})
        return cls(
            stage1=Stage1Config(mrl=MRLConfig(**mrl), **stage1),
            stage2=Stage2Config(**stage2),
            **payload,
        )


def _read_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "PyYAML is required to read YAML configs: pip install pyyaml"
        ) from exc
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: str, **overrides: Any) -> MultiScoreConfig:
    """Load a config from ``.yaml``/``.json`` and apply flat top-level overrides."""

    if path.endswith((".yaml", ".yml")):
        payload = _read_yaml(path)
    elif path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        raise ValueError(f"unsupported config extension: {path}")

    base = dict(payload.pop("base", {}) or {}) if isinstance(payload, dict) else {}
    if base:
        parent = os.path.join(os.path.dirname(os.path.abspath(path)), base["path"])
        merged = _read_yaml(parent)
        merged.update(payload)
        payload = merged

    config = MultiScoreConfig.from_dict(payload)
    for key, value in overrides.items():
        if value is None:
            continue
        if hasattr(config.stage1, key):
            setattr(config.stage1, key, value)
        elif hasattr(config.stage2, key):
            setattr(config.stage2, key, value)
        elif hasattr(config, key):
            setattr(config, key, value)
        else:
            raise AttributeError(f"unknown config field: {key}")
    return config
