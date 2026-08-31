"""The one data format MULTI-SCORE reads.

Both queries and candidates are a bag of optional modalities plus an id::

    {"id": "coco:val:139", "text": "a man riding a horse", "image": "images/139.jpg"}
    {"id": "msrvtt:7010", "video": "videos/7010.mp4", "caption": "a chef ..."}

``caption`` is the offline text rendering produced by
:mod:`multiscore.data.captioning` -- it is what Stage-1 embeds.  The raw media
paths survive into Stage-2, which consumes the native modality directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

MODALITY_FIELDS = ("image", "video", "audio")

#: Canonical modality keys used to pick a captioning prompt.
MODALITY_KEYS = ("t", "i", "it", "v", "a", "av")


@dataclass
class _Item:
    id: str
    text: Optional[str] = None
    image: Optional[str] = None
    video: Optional[str] = None
    audio: Optional[str] = None
    caption: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def modality(self) -> str:
        """Modality key of this item (``t``, ``i``, ``it``, ``v``, ``a``, ``av``)."""

        return modality_key(self)

    def media(self) -> List[Dict[str, str]]:
        """Media payload handed to an MLLM backend in Stage-2."""

        payload: List[Dict[str, str]] = []
        for name in MODALITY_FIELDS:
            value = getattr(self, name)
            if value:
                payload.append({name: value})
        return payload

    def stage1_text(self) -> str:
        """Text handed to the MRL embedder: caption if available, else raw text."""

        parts = [p for p in (self.caption, self.text) if p]
        return " ".join(parts) if parts else ""

    def stage2_text(self) -> str:
        """Text handed to the MLLM in Stage-2 (media, if any, is passed alongside)."""

        return self.stage1_text()

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "_Item":
        known = {"id", "text", "image", "video", "audio", "caption"}
        meta = {k: v for k, v in payload.items() if k not in known}
        return cls(
            id=str(payload["id"]),
            text=payload.get("text"),
            image=payload.get("image"),
            video=payload.get("video"),
            audio=payload.get("audio"),
            caption=payload.get("caption"),
            meta=meta,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"id": self.id}
        for name in ("text", "image", "video", "audio", "caption"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        payload.update(self.meta)
        return payload


@dataclass
class Query(_Item):
    """A retrieval query; ``positives`` holds the gold candidate ids."""

    positives: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Query":
        base = _Item.from_dict(payload)
        positives = payload.get("positives") or payload.get("positive_ids") or []
        if isinstance(positives, str):
            positives = [positives]
        base.meta.pop("positives", None)
        base.meta.pop("positive_ids", None)
        return cls(
            id=base.id,
            text=base.text,
            image=base.image,
            video=base.video,
            audio=base.audio,
            caption=base.caption,
            meta=base.meta,
            positives=[str(p) for p in positives],
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = super().to_dict()
        payload["positives"] = list(self.positives)
        return payload


@dataclass
class Candidate(_Item):
    """A database item."""


def modality_key(item: _Item) -> str:
    """Map an item's populated fields onto a modality key.

    ``t`` text only, ``i`` image, ``it`` image+text, ``v`` video, ``a`` audio,
    ``av`` audio+video -- the five non-text combinations are exactly the ones
    the paper captions offline.
    """

    has_text = bool(item.text)
    has_image = bool(item.image)
    has_video = bool(item.video)
    has_audio = bool(item.audio)

    if has_audio and has_video:
        return "av"
    if has_audio:
        return "a"
    if has_video:
        return "v"
    if has_image and has_text:
        return "it"
    if has_image:
        return "i"
    return "t"


def as_items(rows: Sequence[Dict[str, Any]], kind: str = "candidate") -> List[_Item]:
    factory = Query if kind == "query" else Candidate
    return [factory.from_dict(row) for row in rows]
