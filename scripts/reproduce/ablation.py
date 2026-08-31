#!/usr/bin/env python3
"""Component ablation: naive MRL -> + Pyramid Rank -> + S_CoT -> + S_QA -> full.

Mirrors the ablation figure: each component is switched on in turn and the same
metric block is reported, so the contribution of each is visible.

    python scripts/reproduce/ablation.py --config configs/toy.yaml
"""

from __future__ import annotations

import argparse
import copy
import sys

from multiscore.config import load_config
from multiscore.data.loaders import load_candidates, load_queries
from multiscore.eval.evaluate import efficiency_report, evaluate_results
from multiscore.eval.metrics import format_metrics
from multiscore.models.registry import load_embedder, load_mllm
from multiscore.pipeline import MultiScore
from multiscore.utils.io import write_json
from multiscore.utils.logging import setup_logging

VARIANTS = (
    # (label, naive_stage1, enable_cot, enable_qa)
    ("naive MRL (full-scale, no re-rank)", True, False, False),
    ("+ Pyramid Rank", False, False, False),
    ("+ Pyramid Rank + S_CoT", False, True, False),
    ("+ Pyramid Rank + S_QA", False, False, True),
    ("MULTI-SCORE (all components)", False, True, True),
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--queries", default=None)
    parser.add_argument("--candidates", default=None)
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

    rows = []
    for label, naive_stage1, enable_cot, enable_qa in VARIANTS:
        variant = copy.deepcopy(config)
        variant.stage2.enable_cot = enable_cot
        variant.stage2.enable_qa = enable_qa

        engine = MultiScore(variant, embedder=embedder, mllm=mllm)
        store = engine.index_candidates(candidates)
        stage2 = enable_cot or enable_qa
        results = engine.retrieve_all(
            queries, store,
            candidates=candidates if stage2 else None,
            stage2=stage2, naive_stage1=naive_stage1, progress=False,
        )
        metrics = evaluate_results(results, queries)
        rows.append({"variant": label, "metrics": metrics, "efficiency": efficiency_report(results)})
        print(f"{label:<38} {format_metrics(metrics)}")

    if args.output:
        write_json(args.output, {"rows": rows, "config": config.to_dict()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
