#!/usr/bin/env python3
"""Score a saved ``runs.jsonl`` against a query file (R@1/5/10, nDCG@10).

Example
-------
    python scripts/evaluate.py --runs runs/msrvtt_t2v/runs.jsonl \
        --queries data/msrvtt/t2v/test.queries.jsonl
"""

from __future__ import annotations

import argparse
import sys

from multiscore.data.loaders import attach_qrels, load_qrels, load_queries
from multiscore.eval.metrics import format_metrics, summarize
from multiscore.utils.io import read_jsonl, write_json
from multiscore.utils.logging import get_logger, setup_logging

LOGGER = get_logger("evaluate")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--ndcg-ks", type=int, nargs="+", default=[10])
    parser.add_argument("--output", default=None, help="write metrics as JSON")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)

    queries = load_queries(args.queries)
    if args.qrels:
        attach_qrels(queries, load_qrels(args.qrels))
    positives = {q.id: q.positives for q in queries}

    rankings, relevants, missing = [], [], 0
    for row in read_jsonl(args.runs):
        qid = str(row["query_id"])
        if qid not in positives:
            missing += 1
            continue
        rankings.append([str(c) for c in row["ranked_ids"]])
        relevants.append(positives[qid])

    if missing:
        LOGGER.warning("%d run rows had no matching query", missing)

    metrics = summarize(rankings, relevants, ks=args.ks, ndcg_ks=args.ndcg_ks)
    print(format_metrics(metrics))
    if args.output:
        write_json(args.output, {"metrics": metrics, "num_queries": len(rankings)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
