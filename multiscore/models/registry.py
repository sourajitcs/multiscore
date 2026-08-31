"""Name -> backend registry, so configs can stay stringly-typed."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from multiscore.models.base import EmbeddingBackend, MLLMBackend

_EMBEDDERS: Dict[str, Callable[..., EmbeddingBackend]] = {}
_MLLMS: Dict[str, Callable[..., MLLMBackend]] = {}


def register_embedder(name: str, factory: Callable[..., EmbeddingBackend]) -> None:
    _EMBEDDERS[name.lower()] = factory


def register_mllm(name: str, factory: Callable[..., MLLMBackend]) -> None:
    _MLLMS[name.lower()] = factory


def load_embedder(name: str, **kwargs: Any) -> EmbeddingBackend:
    key = name.lower()
    if key not in _EMBEDDERS:
        raise KeyError(f"unknown embedder '{name}'; available: {sorted(_EMBEDDERS)}")
    return _EMBEDDERS[key](**kwargs)


def load_mllm(name: str, **kwargs: Any) -> MLLMBackend:
    key = name.lower()
    if key not in _MLLMS:
        raise KeyError(f"unknown MLLM '{name}'; available: {sorted(_MLLMS)}")
    return _MLLMS[key](**kwargs)


def available_backends() -> Dict[str, List[str]]:
    return {"embedders": sorted(_EMBEDDERS), "mllms": sorted(_MLLMS)}


# --------------------------------------------------------------------- #
# Lazy factories: importing this module must not import torch.
# --------------------------------------------------------------------- #
def _qwen3_mrl(**kwargs: Any) -> EmbeddingBackend:
    from multiscore.models.qwen3_mrl import Qwen3MRLEmbedder

    return Qwen3MRLEmbedder(**kwargs)


def _resnet_mrl(**kwargs: Any) -> EmbeddingBackend:
    from multiscore.models.qwen3_mrl import ResNetMRLEmbedder

    return ResNetMRLEmbedder(**kwargs)


def _stub_mrl(**kwargs: Any) -> EmbeddingBackend:
    from multiscore.models.stub import HashingMRLEmbedder

    return HashingMRLEmbedder(**kwargs)


def _stub_mllm(**kwargs: Any) -> MLLMBackend:
    from multiscore.models.stub import LexicalMLLM

    return LexicalMLLM(**kwargs)


def _qwen_mllm_factory(default_name: str) -> Callable[..., MLLMBackend]:
    def factory(**kwargs: Any) -> MLLMBackend:
        from multiscore.models.qwen_mllm import QwenMLLM

        kwargs.setdefault("model_name", default_name)
        return QwenMLLM(**kwargs)

    return factory


register_embedder("qwen3-mrl", _qwen3_mrl)
register_embedder("resnet-mrl", _resnet_mrl)
register_embedder("stub-mrl", _stub_mrl)

register_mllm("stub-mllm", _stub_mllm)
for _alias in (
    "qwen2.5-omni-7b",
    "qwen2.5-omni-3b",
    "qwen3-vl-8b",
    "qwen2-audio-7b",
    "qwen3-omni-30b-thinking",
):
    register_mllm(_alias, _qwen_mllm_factory(_alias))
