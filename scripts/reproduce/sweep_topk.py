#!/usr/bin/env python3
"""Sweep the Stage-2 candidate budget K: recall and runtime.

The trade-off of the paper's K study -- a larger shortlist gives Stage-2 more to
work with (recall goes up) while Pyramid Rank keeps the Stage-1 cost of
producing it nearly flat, unlike naive full-resolution scoring.

    python scripts/reproduce/sweep_topk.py --config configs/toy.yaml --ks 1 2 5 10
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from multiscore.config import load_config
from multiscore.data.loaders import load_candidates, load_queries
from multiscore.eval.metrics import recall_at_k
from multiscore.models.registry import load_embedder
from multiscore.stage1.mrl import MRLEmbeddingStore
from multiscore.stage1.pyramid_rank import pyramid_rank
from multiscore.utils.io import write_json
from multiscore.utils.logging import setup_logging


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--queries", default=None)
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 25, 50, 100, 200])
    parser.add_argument("--recall-at", type=int, default=5)
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
    store = MRLEmbeddingStore(
        embedder.encode([c.stage1_text() for c in candidates]),
        mrl=config.stage1.mrl,
        ids=[c.id for c in candidates],
    )
    query_embeddings = embedder.encode([q.stage1_text() for q in queries], is_query=True)
    relevants = [q.positives for q in queries]

    rows = []
    for k in args.ks:
        start = time.perf_counter()
        outputs = [pyramid_rank(e, store, config.stage1, top_k=k) for e in query_embeddings]
        elapsed_ms = 1000 * (time.perf_counter() - start) / len(queries)
        rows.append({
            "K": k,
            # Candidate survival: does the gold item reach Stage-2 at all?
            "hit_at_K": recall_at_k([o.ids for o in outputs], relevants, k),
            f"R@{args.recall_at}": recall_at_k([o.ids for o in outputs], relevants, args.recall_at),
            "stage1_ms_per_query": elapsed_ms,
            "speedup": float(np.mean([o.speedup for o in outputs])),
        })

    print(f"{'K':>6} {'Hit@K':>8} {'R@'+str(args.recall_at):>8} {'ms/query':>10} {'speedup':>8}")
    for row in rows:
        print(f"{row['K']:>6} {row['hit_at_K']:8.2f} {row[f'R@{args.recall_at}']:8.2f} "
              f"{row['stage1_ms_per_query']:10.3f} {row['speedup']:7.2f}x")

    if args.output:
        write_json(args.output, {"rows": rows})
    return 0


if __name__ == "__main__":
    sys.exit(main())
