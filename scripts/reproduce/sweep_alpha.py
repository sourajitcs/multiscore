#!/usr/bin/env python3
"""Sweep the Stage-2 mixing weight alpha (Eq. 9) and report R@1 per value.

Reproduces the alpha analysis: image retrieval and audio/video retrieval peak at
different alphas, so the sweep is what settles the per-modality default.  Both
Stage-2 scores are computed once per (query, candidate) pair and then re-mixed,
so the sweep costs one Stage-2 pass, not one per alpha.

    python scripts/reproduce/sweep_alpha.py --config configs/toy.yaml
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from multiscore.config import load_config
from multiscore.data.loaders import load_candidates, load_queries
from multiscore.eval.metrics import recall_at_k
from multiscore.models.registry import load_embedder, load_mllm
from multiscore.pipeline import MultiScore
from multiscore.stage1.pyramid_rank import pyramid_rank
from multiscore.stage2.cot_score import BidirectionalCoTScorer
from multiscore.stage2.qa_score import QARelevanceScorer
from multiscore.stage2.rerank import combine_scores
from multiscore.utils.io import write_json
from multiscore.utils.logging import setup_logging


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--queries", default=None)
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--k", type=int, default=1, help="R@k to report")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging()
    config = load_config(args.config)

    queries = load_queries(args.queries or config.queries, limit=args.limit)
    candidates = load_candidates(args.candidates or config.candidates)

    embedder_kwargs = ({"dim": config.stage1.mrl.full_dim} if config.stage1.embedder == "stub-mrl"
                       else {"device": config.device, "dtype": config.dtype})
    embedder = load_embedder(config.stage1.embedder, **embedder_kwargs)
    mllm_kwargs = {} if config.stage2.backend == "stub-mllm" else {"device": config.device, "dtype": config.dtype}
    mllm = load_mllm(config.stage2.backend, **mllm_kwargs)

    engine = MultiScore(config, embedder=embedder, mllm=mllm)
    store = engine.index_candidates(candidates)
    by_id = {c.id: c for c in candidates}

    cot_scorer = BidirectionalCoTScorer(mllm, config.stage2)
    qa_scorer = QARelevanceScorer(mllm, config.stage2)

    # One Stage-2 pass; the alphas only re-mix the cached component scores.
    shortlists, cot_all, qa_all, relevants = [], [], [], []
    query_embeddings = embedder.encode([q.stage1_text() for q in queries], is_query=True)
    for i, query in enumerate(queries):
        stage1 = pyramid_rank(query_embeddings[i], store, config.stage1)
        shortlist = [by_id[cid] for cid in stage1.ids if cid in by_id]
        texts = [c.stage2_text() for c in shortlist]
        media = [c.media() for c in shortlist]

        shortlists.append([c.id for c in shortlist])
        cot_all.append(cot_scorer.score_candidates(query.stage2_text(), texts,
                                                   query_media=query.media(), candidate_media=media))
        qa_all.append(qa_scorer.score_candidates(query.stage2_text(), texts,
                                                 candidate_ids=[c.id for c in shortlist],
                                                 query_media=query.media(), candidate_media=media))
        relevants.append(query.positives)

    curve = {}
    for alpha in args.alphas:
        rankings = []
        for ids, cot, qa in zip(shortlists, cot_all, qa_all):
            combined = combine_scores(cot, qa, alpha)
            rankings.append([ids[j] for j in np.argsort(-combined, kind="stable")])
        curve[alpha] = recall_at_k(rankings, relevants, args.k)

    best = max(curve, key=curve.get)
    print(f"{'alpha':>6}  R@{args.k}")
    for alpha, value in curve.items():
        marker = "  <- best" if alpha == best else ""
        print(f"{alpha:>6.2f}  {value:6.2f}{marker}")

    if args.output:
        write_json(args.output, {"k": args.k, "curve": {str(a): v for a, v in curve.items()}, "best_alpha": best})
    return 0


if __name__ == "__main__":
    sys.exit(main())
