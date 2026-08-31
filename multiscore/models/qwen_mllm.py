"""Qwen multimodal backends for Stage-2 (and for offline captioning).

The paper uses:

===============================  ====================================
Qwen3-VL-8B                      image / image+text / video captioning
Qwen2-Audio-7B                   audio transcription and captioning
Qwen2.5-Omni-7B                  audio+video captioning, and both
                                 Stage-2 scores (CoT + QA)
===============================  ====================================

All three are exposed through one wrapper because ``transformers`` gives them
the same chat/processor interface; only the processor class differs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from multiscore.models.base import MLLMBackend

MODEL_ALIASES = {
    "qwen2.5-omni-7b": "Qwen/Qwen2.5-Omni-7B",
    "qwen2.5-omni-3b": "Qwen/Qwen2.5-Omni-3B",
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen2-audio-7b": "Qwen/Qwen2-Audio-7B-Instruct",
    "qwen3-omni-30b-thinking": "Qwen/Qwen3-Omni-30B-A3B-Thinking",
}


def resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name.lower(), name)


class QwenMLLM(MLLMBackend):
    """Generation + ``<emb>`` hidden-state extraction on top of a Qwen MLLM."""

    name = "qwen-mllm"

    def __init__(
        self,
        model_name: str = "qwen2.5-omni-7b",
        device: str = "auto",
        dtype: str = "bfloat16",
        max_length: int = 32768,
        attn_implementation: Optional[str] = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "QwenMLLM needs `torch` and `transformers`: pip install -e '.[models]'"
            ) from exc

        self._torch = torch
        self.model_name = resolve_model_name(model_name)
        self.max_length = int(max_length)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        torch_dtype = getattr(torch, dtype) if device != "cpu" else torch.float32

        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype, "trust_remote_code": True}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs).to(device)
        self.model.eval()

    # ------------------------------------------------------------------ #
    def _build_inputs(self, prompt: str, media: Optional[Sequence[Dict[str, Any]]]):
        content: List[Dict[str, Any]] = []
        for item in media or []:
            for modality, value in item.items():
                content.append({"type": modality, modality: value})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        except TypeError:  # processors that require explicit modality kwargs
            inputs = self.processor(text=[text], images=None, return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")}

    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        media: Optional[Sequence[Dict[str, Any]]] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        torch = self._torch
        inputs = self._build_inputs(prompt, media)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True
        )[0].strip()

    def embed_with_marker(
        self,
        prompt: str,
        media: Optional[Sequence[Dict[str, Any]]] = None,
        marker: str = "<emb>",
    ) -> np.ndarray:
        """Return the hidden state of the token immediately preceding ``marker``.

        Following the one-word-limitation formulation, the prompt ends with the
        marker token and the state just before it aggregates the full context,
        so a single forward pass (no decoding) yields the embedding.
        """

        torch = self._torch
        if marker not in prompt:
            prompt = f"{prompt}\n{marker}"

        inputs = self._build_inputs(prompt, media)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        hidden = outputs.hidden_states[-1][0]  # (seq_len, hidden_dim)
        marker_pos = self._marker_position(inputs["input_ids"][0], marker)
        vector = hidden[max(0, marker_pos - 1)].float().cpu().numpy()
        norm = float(np.linalg.norm(vector))
        return (vector / norm).astype(np.float32) if norm > 0 else vector.astype(np.float32)

    def _marker_position(self, input_ids, marker: str) -> int:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        marker_ids = tokenizer.encode(marker, add_special_tokens=False)
        ids = input_ids.tolist()
        if marker_ids:
            first = marker_ids[0]
            for pos in range(len(ids) - 1, -1, -1):
                if ids[pos] == first:
                    return pos
        return len(ids) - 1  # fall back to the final position
