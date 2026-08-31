#!/usr/bin/env python3
"""Offline pre-processing: caption a JSONL shard so Stage-1 can embed it.

Every image / video / audio item is turned into text exactly once, offline.
The paper reports ~20 GPU-hours for the full 5.7M-item database when sharded
across 32 GPUs; use ``--shard i --num-shards n`` to reproduce that split.

Example
-------
    python scripts/preprocess_caption.py \
        --input  data/msrvtt/t2v/candidates.jsonl \
        --output data/msrvtt/t2v/candidates.captioned.jsonl \
        --backend qwen3-vl-8b --shard 0 --num-shards 8
"""

from __future__ import annotations

import argparse
import sys

from multiscore.data.captioning import CAPTION_MODEL_BY_MODALITY, Captioner
from multiscore.data.schema import Candidate, Query
from multiscore.models.registry import load_mllm
from multiscore.utils.io import read_jsonl, write_jsonl
from multiscore.utils.logging import get_logger, setup_logging

LOGGER = get_logger("caption")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="JSONL of queries or candidates")
    parser.add_argument("--output", required=True, help="destination JSONL")
    parser.add_argument(
        "--backend",
        default="qwen3-vl-8b",
        help="captioning MLLM; the paper uses qwen3-vl-8b (image/video), "
        "qwen2-audio-7b (audio) and qwen2.5-omni-7b (audio+video)",
    )
    parser.add_argument("--kind", choices=("candidate", "query"), default="candidate")
    parser.add_argument("--max-new-tokens", type=int, default=768,
                        help=">600 tokens works best in the caption-length ablation")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="re-caption items that already have one")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    factory = Query if args.kind == "query" else Candidate
    items = []
    for i, row in enumerate(read_jsonl(args.input)):
        if i % args.num_shards != args.shard:
            continue
        items.append(factory.from_dict(row))
        if args.limit is not None and len(items) >= args.limit:
            break

    modalities = sorted({item.modality for item in items})
    LOGGER.info("shard %d/%d: %d items, modalities=%s", args.shard, args.num_shards, len(items), modalities)
    for modality in modalities:
        if modality != "t":
            LOGGER.info("  %s -> paper uses %s", modality, CAPTION_MODEL_BY_MODALITY.get(modality, "?"))

    backend = load_mllm(args.backend, device=args.device, dtype=args.dtype)
    captioner = Captioner(backend, max_new_tokens=args.max_new_tokens)
    captioner.caption_all(items, overwrite=args.overwrite)

    written = write_jsonl(args.output, (item.to_dict() for item in items))
    LOGGER.info("wrote %d captioned items to %s", written, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
