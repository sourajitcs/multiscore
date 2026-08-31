#!/usr/bin/env python3
"""Build the Stage-1 MRL index over a captioned candidate database.

Embedding is the cheap half of offline pre-processing (~0.012 s per item; 0.6
GPU-hours for 5.7M items across 32 GPUs).  The index is written once and reused
by every retrieval run.

Example
-------
    python scripts/build_index.py \
        --candidates data/msrvtt/t2v/candidates.captioned.jsonl \
        --index-dir runs/index/msrvtt_t2v \
        --embedder qwen3-mrl
"""

from __future__ import annotations

import argparse
import sys

from multiscore.config import MRLConfig
from multiscore.data.loaders import load_candidates
from multiscore.index.store import build_index
from multiscore.models.registry import load_embedder
from multiscore.utils.logging import get_logger, setup_logging

LOGGER = get_logger("index")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", required=True, help="captioned candidates JSONL")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--embedder", default="qwen3-mrl", help="'stub-mrl' needs no downloads")
    parser.add_argument("--base-dim", type=int, default=32, help="d, the level-1 width")
    parser.add_argument("--num-levels", type=int, default=6, help="L, number of MRL levels")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    mrl = MRLConfig(base_dim=args.base_dim, num_levels=args.num_levels)
    candidates = load_candidates(args.candidates, limit=args.limit)
    LOGGER.info("encoding %d candidates at D=%d (levels %s)", len(candidates), mrl.full_dim, mrl.level_dims)

    kwargs = {"device": args.device, "dtype": args.dtype}
    if args.embedder == "stub-mrl":
        kwargs = {"dim": mrl.full_dim}
    embedder = load_embedder(args.embedder, **kwargs)

    build_index(candidates, embedder, mrl=mrl, batch_size=args.batch_size, index_dir=args.index_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
