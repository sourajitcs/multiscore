#!/usr/bin/env python3
"""A 60-second tour of MULTI-SCORE on a 24-item toy corpus.

Runs the whole pipeline -- Pyramid Rank then CoT/QA re-ranking -- with the
deterministic stub backends, so it needs no checkpoints, no GPU and no
benchmark downloads.  The retrieval numbers are meaningless; the point is to
show what each stage produces and how to read its diagnostics.

    python examples/toy_demo.py
"""

from __future__ import annotations

import os

from multiscore.config import MRLConfig, MultiScoreConfig, Stage1Config, Stage2Config
from multiscore.data.loaders import load_candidates, load_queries
from multiscore.eval.evaluate import efficiency_report, evaluate_results
from multiscore.eval.metrics import format_metrics
from multiscore.models.stub import HashingMRLEmbedder, LexicalMLLM
from multiscore.pipeline import MultiScore

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    config = MultiScoreConfig(
        stage1=Stage1Config(top_k=5, epsilon=0.05, mrl=MRLConfig(base_dim=8, num_levels=6)),
        stage2=Stage2Config(alpha=0.6, num_questions=4, num_cot_examples=2),
    )
    engine = MultiScore(
        config,
        embedder=HashingMRLEmbedder(dim=config.stage1.mrl.full_dim),
        mllm=LexicalMLLM(),
    )

    queries = load_queries(os.path.join(HERE, "toy_data", "queries.jsonl"))
    candidates = load_candidates(os.path.join(HERE, "toy_data", "candidates.jsonl"))
    print(f"corpus: {len(candidates)} candidates, {len(queries)} queries")
    print(f"MRL pyramid: levels {config.stage1.mrl.level_dims} (d={config.stage1.mrl.base_dim}, "
          f"L={config.stage1.mrl.num_levels}, D={config.stage1.mrl.full_dim})\n")

    store = engine.index_candidates(candidates)
    by_id = {c.id: c for c in candidates}

    # ---- Stage-1 in isolation ------------------------------------------ #
    query = queries[0]
    stage1_only = engine.retrieve(query, store, stage2=False)
    s1 = stage1_only.stage1
    print(f'query "{query.text}"  (gold: {query.positives})')
    print("\n[Stage-1] Pyramid Rank")
    print(f"  bisection steps      : {s1.iterations}")
    print(f"  levels visited       : {s1.level_history}")
    print(f"  survivors per step   : {s1.survivors_history}")
    print(f"  terminal threshold   : [{s1.tau_min:.3f}, {s1.tau_max:.3f}]")
    print(f"  cost vs naive        : {s1.cost} vs {s1.naive_cost} MACs  ({s1.speedup:.2f}x)")
    print("  (24 candidates for a budget of 5 leaves nothing to filter -- see the last")
    print("   section for the regime Pyramid Rank is built for)")
    for rank, (cid, score) in enumerate(zip(stage1_only.ranked_ids, stage1_only.scores), 1):
        print(f"    {rank}. {cid}  {score:+.4f}  {by_id[cid].caption[:64]}...")

    # ---- Stage-2 re-ranking -------------------------------------------- #
    full = engine.retrieve(query, store, candidates=candidates)
    print("\n[Stage-2] alpha * S_CoT + (1 - alpha) * S_QA"
          f"   (alpha={config.stage2.alpha})")
    print("  generated questions:")
    for pair in full.stage2.questions:
        print(f"    - {pair.question}  -> {pair.answer}")
    for rank, cid in enumerate(full.ranked_ids, 1):
        print(f"    {rank}. {cid}  S_rerank={full.scores[rank - 1]:+.4f}  "
              f"S_CoT={full.stage2.cot_scores[rank - 1]:+.4f}  "
              f"S_QA={full.stage2.qa_scores[rank - 1]:.2f}")

    print("\n  why the top candidate scored what it did:")
    print("   ", full.explain(top=1).replace("\n", "\n    "))

    # ---- Whole query set ------------------------------------------------ #
    results = engine.retrieve_all(queries, store, candidates=candidates, progress=False)
    print("\n[Evaluation] (stub backends -- numbers are illustrative only)")
    print("  ", format_metrics(evaluate_results(results, queries)))
    efficiency = efficiency_report(results)
    print(f"   Stage-1 mean cost ratio vs naive full-resolution MRL: {efficiency['mean_speedup']:.2f}x")

    scaling_demo(config)


def scaling_demo(config: MultiScoreConfig, num_candidates: int = 50_000) -> None:
    """Where Pyramid Rank actually pays: K << N.

    Uses synthetic unit-norm embeddings with a front-loaded energy profile, so no
    data is needed -- the saving comes from the geometry of the pyramid, not from
    the semantics of the vectors.
    """

    import numpy as np

    from multiscore.config import Stage1Config
    from multiscore.stage1.mrl import MRLEmbeddingStore
    from multiscore.stage1.pyramid_rank import pyramid_rank

    mrl = config.stage1.mrl
    rng = np.random.default_rng(0)
    decay = 1.0 / (1.0 + np.arange(mrl.full_dim, dtype=np.float32)) ** 0.5

    store = MRLEmbeddingStore(rng.standard_normal((num_candidates, mrl.full_dim)).astype("float32") * decay, mrl=mrl)
    queries = rng.standard_normal((8, mrl.full_dim)).astype("float32") * decay

    stage1 = Stage1Config(top_k=100, epsilon=0.02, mrl=mrl)
    outputs = [pyramid_rank(q, store, stage1) for q in queries]

    print(f"\n[Scaling] synthetic database, N={num_candidates:,}, K={stage1.top_k}, eps={stage1.epsilon}")
    print(f"   mean bisection steps : {np.mean([o.iterations for o in outputs]):.1f}")
    print(f"   mean deepest level   : {np.mean([o.final_level for o in outputs]):.1f} of {mrl.num_levels}")
    print(f"   mean cost saving     : {np.mean([o.speedup for o in outputs]):.2f}x fewer MACs than scoring all "
          f"{num_candidates:,} candidates at D={mrl.full_dim}")


if __name__ == "__main__":
    main()
