#!/usr/bin/env python3
"""Run the two-stage MULTI-SCORE pipeline over a query set.

Writes one JSONL row per query (ranked ids, scores, per-stage diagnostics) plus
a ``metrics.json`` summary when the queries carry relevance judgements.

Examples
--------
    # full pipeline against a pre-built index
    python scripts/run_retrieval.py --config configs/video.yaml \
        --queries data/msrvtt/t2v/test.queries.jsonl \
        --candidates data/msrvtt/t2v/candidates.captioned.jsonl \
        --index-dir runs/index/msrvtt_t2v --output-dir runs/msrvtt_t2v

    # Stage-1 only, and the naive full-resolution baseline, on the toy corpus
    python scripts/run_retrieval.py --config configs/toy.yaml --stage1-only
    python scripts/run_retrieval.py --config configs/toy.yaml --stage1-only --naive-stage1
"""

from __future__ import annotations

import argparse
import os
import sys

from multiscore.config import load_config
from multiscore.data.loaders import attach_qrels, load_candidates, load_qrels, load_queries
from multiscore.eval.evaluate import efficiency_report, evaluate_results, level_histogram
from multiscore.index.store import EmbeddingIndex
from multiscore.models.registry import load_embedder, load_mllm
from multiscore.pipeline import MultiScore
from multiscore.utils.io import write_json, write_jsonl
from multiscore.utils.logging import get_logger, setup_logging
from multiscore.utils.seed import set_seed

LOGGER = get_logger("retrieval")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--queries", default=None)
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--index-dir", default=None, help="reuse a pre-built index instead of encoding")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-k", type=int, default=None, help="K handed to Stage-2")
    parser.add_argument("--epsilon", type=float, default=None, help="Stage-1 tolerance")
    parser.add_argument("--alpha", type=float, default=None, help="weight on S_CoT in Eq. 9")
    parser.add_argument("--stage1-only", action="store_true", help="skip re-ranking")
    parser.add_argument("--naive-stage1", action="store_true",
                        help="ablation: exhaustive full-resolution MRL instead of Pyramid Rank")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of queries")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    config = load_config(args.config, top_k=args.top_k, epsilon=args.epsilon, alpha=args.alpha)
    set_seed(config.seed)

    queries_path = args.queries or config.queries
    candidates_path = args.candidates or config.candidates
    if not queries_path or not candidates_path:
        raise SystemExit("--queries and --candidates are required unless the config sets them")

    queries = load_queries(queries_path, limit=args.limit)
    candidates = load_candidates(candidates_path)
    qrels_path = args.qrels or config.qrels
    if qrels_path:
        attach_qrels(queries, load_qrels(qrels_path))
    LOGGER.info("%d queries, %d candidates", len(queries), len(candidates))

    # ---- backends -------------------------------------------------------- #
    embedder_kwargs = {"device": config.device, "dtype": config.dtype}
    if config.stage1.embedder == "stub-mrl":
        embedder_kwargs = {"dim": config.stage1.mrl.full_dim}
    embedder = load_embedder(config.stage1.embedder, **embedder_kwargs)

    mllm = None
    if not args.stage1_only:
        mllm_kwargs = {} if config.stage2.backend == "stub-mllm" else {
            "device": config.device,
            "dtype": config.dtype,
        }
        mllm = load_mllm(config.stage2.backend, **mllm_kwargs)

    engine = MultiScore(config, embedder=embedder, mllm=mllm)

    # ---- index ----------------------------------------------------------- #
    index_dir = args.index_dir or config.index_dir
    if index_dir and os.path.exists(os.path.join(index_dir, "embeddings.npy")):
        LOGGER.info("loading index from %s", index_dir)
        store = EmbeddingIndex.load(index_dir).as_store()
    else:
        LOGGER.info("no index found; encoding %d candidates now", len(candidates))
        store = engine.index_candidates(candidates)

    # ---- retrieve -------------------------------------------------------- #
    results = engine.retrieve_all(
        queries,
        store,
        candidates=None if args.stage1_only else candidates,
        stage2=not args.stage1_only,
        naive_stage1=args.naive_stage1,
    )

    metrics = evaluate_results(results, queries)
    efficiency = efficiency_report(results)
    LOGGER.info("metrics: %s", metrics)
    LOGGER.info("stage-1 efficiency: %s", efficiency)

    output_dir = args.output_dir or config.output_dir
    write_jsonl(os.path.join(output_dir, "runs.jsonl"), (r.to_dict() for r in results))
    write_json(
        os.path.join(output_dir, "metrics.json"),
        {
            "metrics": metrics,
            "efficiency": efficiency,
            "level_histogram": level_histogram(results, config.stage1.mrl.num_levels),
            "config": config.to_dict(),
            "stage1_only": args.stage1_only,
            "naive_stage1": args.naive_stage1,
            "num_queries": len(queries),
            "num_candidates": len(candidates),
        },
    )
    LOGGER.info("wrote results to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
