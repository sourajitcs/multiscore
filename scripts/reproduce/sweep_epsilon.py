#!/usr/bin/env python3
"""Sweep the Stage-1 tolerance epsilon: retrieval quality vs. Stage-1 cost.

A larger epsilon terminates the bisection sooner (cheaper, coarser); a smaller
one converges to the exact full-resolution ranking.  The paper reports the
optimum at eps = 0.02 and shows that even eps = 0.40 stays competitive.

    python scripts/reproduce/sweep_epsilon.py --config configs/toy.yaml
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
from multiscore.stage1.naive import naive_full_scale_rank
from multiscore.stage1.pyramid_rank import pyramid_rank
from multiscore.utils.io import write_json
from multiscore.utils.logging import setup_logging


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--queries", default=None)
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.4, 0.2, 0.1, 0.05, 0.02, 0.01])
    parser.add_argument("--k", type=int, default=5)
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
    # Reference row: exhaustive full-resolution MRL.
    start = time.perf_counter()
    naive = [naive_full_scale_rank(e, store, top_k=config.stage1.top_k) for e in query_embeddings]
    naive_ms = 1000 * (time.perf_counter() - start) / len(queries)
    rows.append({
        "epsilon": None,
        "recall": recall_at_k([o.ids for o in naive], relevants, args.k),
        "ms_per_query": naive_ms,
        "mean_cost": float(np.mean([o.cost for o in naive])),
        "speedup": 1.0,
    })

    for epsilon in args.epsilons:
        start = time.perf_counter()
        outputs = [pyramid_rank(e, store, config.stage1, epsilon=epsilon) for e in query_embeddings]
        elapsed_ms = 1000 * (time.perf_counter() - start) / len(queries)
        rows.append({
            "epsilon": epsilon,
            "recall": recall_at_k([o.ids for o in outputs], relevants, args.k),
            "ms_per_query": elapsed_ms,
            "mean_cost": float(np.mean([o.cost for o in outputs])),
            "speedup": float(np.mean([o.speedup for o in outputs])),
        })

    print(f"{'epsilon':>8} {'R@'+str(args.k):>8} {'ms/query':>10} {'MACs':>12} {'speedup':>8}")
    for row in rows:
        label = "naive" if row["epsilon"] is None else f"{row['epsilon']:.3f}"
        print(f"{label:>8} {row['recall']:8.2f} {row['ms_per_query']:10.3f} "
              f"{row['mean_cost']:12.0f} {row['speedup']:7.2f}x")

    if args.output:
        write_json(args.output, {"k": args.k, "rows": rows})
    return 0


if __name__ == "__main__":
    sys.exit(main())
