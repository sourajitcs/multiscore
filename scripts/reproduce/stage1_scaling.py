#!/usr/bin/env python3
"""Stage-1 scaling study: Pyramid Rank vs. naive full-resolution MRL.

Measures cost and wall-clock as the database grows, reproducing the shape of the
online Stage-1 cost table (1.8x at 100K rising to 3.3x at 5.7M) without needing
any real data: the geometry of the pyramid, not the semantics of the vectors,
is what drives the saving.  Recall preservation is measured against the exact
level-L top-K.

    python scripts/reproduce/stage1_scaling.py --sizes 10000 50000 100000
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from multiscore.config import MRLConfig, Stage1Config
from multiscore.stage1.mrl import MRLEmbeddingStore
from multiscore.stage1.naive import naive_full_scale_rank
from multiscore.stage1.pyramid_rank import pyramid_rank
from multiscore.utils.io import write_json
from multiscore.utils.logging import setup_logging


def synthetic_store(n: int, mrl: MRLConfig, seed: int, decay: float) -> MRLEmbeddingStore:
    """Unit-norm vectors whose energy decays across dimensions, as MRL embeddings do."""

    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n, mrl.full_dim)).astype(np.float32)
    raw *= (1.0 / (1.0 + np.arange(mrl.full_dim, dtype=np.float32)) ** decay)
    return MRLEmbeddingStore(raw, mrl=mrl)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[10_000, 50_000, 100_000, 500_000])
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--base-dim", type=int, default=32)
    parser.add_argument("--num-levels", type=int, default=6)
    parser.add_argument("--decay", type=float, default=0.5, help="how fast embedding energy front-loads")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging()

    mrl = MRLConfig(base_dim=args.base_dim, num_levels=args.num_levels)
    stage1 = Stage1Config(top_k=args.top_k, epsilon=args.epsilon, mrl=mrl)
    rng = np.random.default_rng(args.seed + 1)

    rows = []
    for n in args.sizes:
        store = synthetic_store(n, mrl, seed=args.seed, decay=args.decay)
        queries = rng.standard_normal((args.num_queries, mrl.full_dim)).astype(np.float32)
        queries *= (1.0 / (1.0 + np.arange(mrl.full_dim, dtype=np.float32)) ** args.decay)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        start = time.perf_counter()
        naive = [naive_full_scale_rank(q, store, top_k=args.top_k) for q in queries]
        naive_ms = 1000 * (time.perf_counter() - start) / len(queries)

        start = time.perf_counter()
        pyramid = [pyramid_rank(q, store, stage1) for q in queries]
        pyramid_ms = 1000 * (time.perf_counter() - start) / len(queries)

        overlap = np.mean([
            len(set(a.indices.tolist()) & set(b.indices.tolist())) / args.top_k
            for a, b in zip(pyramid, naive)
        ])
        rows.append({
            "N": n,
            "naive_ms": naive_ms,
            "pyramid_ms": pyramid_ms,
            "wallclock_speedup": naive_ms / max(pyramid_ms, 1e-9),
            "mac_speedup": float(np.mean([o.speedup for o in pyramid])),
            f"topK_preservation@{args.top_k}": float(overlap),
            "mean_iterations": float(np.mean([o.iterations for o in pyramid])),
        })

    header = f"{'N':>10} {'naive ms':>10} {'pyramid ms':>11} {'wall':>7} {'MACs':>7} {'top-K kept':>11}"
    print(header)
    for row in rows:
        print(f"{row['N']:>10} {row['naive_ms']:10.3f} {row['pyramid_ms']:11.3f} "
              f"{row['wallclock_speedup']:6.2f}x {row['mac_speedup']:6.2f}x "
              f"{row[f'topK_preservation@{args.top_k}']:11.3f}")

    if args.output:
        write_json(args.output, {"rows": rows, "top_k": args.top_k, "epsilon": args.epsilon})
    return 0


if __name__ == "__main__":
    sys.exit(main())
