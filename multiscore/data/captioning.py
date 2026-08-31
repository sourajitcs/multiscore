"""Offline pre-processing: turn every modality into text for Stage-1.

Qwen3-MRL is text-only, so images, videos and audio are captioned once, offline,
and only the captions are embedded.  The paper measures this as a one-time cost
(~20 GPU-hours for 5.7M items across 32 GPUs) and shows the conversion costs
little retrieval quality (R@100 73.26 -> 72.98 on the image-MRL comparison).

Stage-2 does *not* use these captions in place of the media: it re-introduces
the native modality, which is where the fine-grained gains come from.
"""

from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Sequence

from multiscore.data.schema import _Item
from multiscore.models.base import MLLMBackend
from multiscore.stage2.prompts import CAPTION_PROMPTS, caption_prompt
from multiscore.utils.logging import get_logger

LOGGER = get_logger(__name__)

#: Captioning model used per modality in the paper (see Appendix A.4).
CAPTION_MODEL_BY_MODALITY: Dict[str, str] = {
    key: model for key, (_, model) in CAPTION_PROMPTS.items()
}


class Captioner:
    """Wraps an MLLM backend with the paper's per-modality caption prompts."""

    def __init__(
        self,
        backend: MLLMBackend,
        max_new_tokens: int = 768,
        temperature: float = 0.0,
        skip_text_only: bool = True,
    ) -> None:
        self.backend = backend
        # The ablation on caption length finds >600 tokens best across
        # image / video / audio, hence the generous default budget.
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.skip_text_only = skip_text_only

    def caption_item(self, item: _Item) -> Optional[str]:
        """Caption one item; returns ``None`` for text-only items."""

        modality = item.modality
        if modality == "t":
            return None if self.skip_text_only else item.text
        prompt = caption_prompt(modality)
        if item.text:
            prompt = f"{prompt}\n\nAccompanying text: {item.text}"
        return self.backend.generate(
            prompt,
            media=item.media(),
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        ).strip()

    def caption_all(
        self,
        items: Sequence[_Item],
        overwrite: bool = False,
        progress: bool = True,
    ) -> List[_Item]:
        """Caption a collection in place, skipping items that already have one."""

        iterator: Iterable[_Item] = items
        if progress:
            try:
                from tqdm.auto import tqdm  # type: ignore

                iterator = tqdm(items, desc="captioning", unit="item")
            except ImportError:
                pass

        for item in iterator:
            if item.caption and not overwrite:
                continue
            try:
                item.caption = self.caption_item(item)
            except Exception as exc:  # keep long offline jobs alive
                LOGGER.warning("captioning failed for %s: %s", item.id, exc)
                item.caption = item.text
        return list(items)


def caption_stream(
    items: Iterable[_Item],
    captioner: Captioner,
    overwrite: bool = False,
) -> Iterator[Dict[str, object]]:
    """Generator variant for streaming large shards straight to JSONL."""

    for item in items:
        if overwrite or not item.caption:
            item.caption = captioner.caption_item(item)
        yield item.to_dict()
