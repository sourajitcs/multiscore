# Reproducing the paper

What each script produces, what it costs and which result it corresponds to.

---

## 0. Without any data

These run in seconds on CPU and need no checkpoints:

```bash
make test                                   # guarantees, prompts, metrics, end-to-end
python examples/toy_demo.py                 # annotated walkthrough
python scripts/run_retrieval.py --config configs/toy.yaml
python scripts/reproduce/stage1_scaling.py --sizes 10000 100000 1000000
```

`stage1_scaling.py` is the only *quantitative* claim reproducible without data: the Stage-1
saving follows from the geometry of the pyramid, not the semantics of the vectors, so it can
be measured on synthetic unit-norm embeddings with a realistic energy decay. Expect the
speed-up to grow with `N`, as in the paper's cost table (1.8× at 100K → 3.3× at 5.7M).

Everything else needs the benchmarks ([`docs/DATA.md`](DATA.md)) and the Qwen backbones.

---

## 1. Full pipeline, one benchmark

```bash
python scripts/preprocess_caption.py --input data/<ds>/<task>/candidates.jsonl \
    --output data/<ds>/<task>/candidates.captioned.jsonl --backend qwen3-vl-8b
python scripts/build_index.py --candidates data/<ds>/<task>/candidates.captioned.jsonl \
    --index-dir runs/index/<ds>_<task>
python scripts/run_retrieval.py --config configs/<family>.yaml \
    --queries data/<ds>/<task>/test.queries.jsonl \
    --candidates data/<ds>/<task>/candidates.captioned.jsonl \
    --index-dir runs/index/<ds>_<task> --output-dir runs/<ds>_<task>
```

Use `configs/mbeir.yaml` for image-text tasks, `configs/video.yaml` for video and
`configs/audio.yaml` for audio — they differ only in `α` and bookkeeping.

Outputs: `runs/<ds>_<task>/runs.jsonl` (one row per query, with per-stage diagnostics) and
`metrics.json` (R@1/5/10, nDCG@10, Stage-1 efficiency, level histogram, the resolved config).

Hardware in the paper: Stage-1 on 8 clusters of 4 H100s (batch 128); Stage-2 on 4 clusters
of 16 L40S (CoT batch 8, QA batch 4); greedy decoding.

---

## 2. Analyses

| Script | Reproduces | Needs |
|---|---|---|
| `reproduce/ablation.py` | Component ablation: naive MRL → + Pyramid Rank → + `S_CoT` → + `S_QA` → full | 1 benchmark, Stage-2 backbone |
| `reproduce/sweep_alpha.py` | The α sweep and its per-modality optimum | 1 benchmark, Stage-2 backbone |
| `reproduce/sweep_epsilon.py` | ε vs. R@k and Stage-1 cost | 1 benchmark, Stage-1 only |
| `reproduce/sweep_topk.py` | K vs. recall, candidate survival and Stage-1 runtime | 1 benchmark, Stage-1 only |
| `reproduce/stage1_scaling.py` | Stage-1 speed-up vs. database size | nothing |

`sweep_alpha.py` computes `S_CoT` and `S_QA` once per (query, candidate) pair and re-mixes
them for each α, so a full sweep costs one Stage-2 pass rather than one per α.

The ε and K sweeps are Stage-1 only and therefore cheap: they need the index, not the MLLM.

### Backbone-agnostic gains

Point `stage2.backend` at `qwen2.5-omni-3b`, `qwen2.5-omni-7b` or `qwen2-audio-7b` and
re-run the same benchmark. The registry resolves the aliases to Hugging Face ids
(`models/qwen_mllm.py`); nothing else changes.

### Dataset difficulty

`metrics.json` contains the distribution of MRL levels Pyramid Rank actually used. Easy
datasets resolve at low levels with tight, right-shifted bounds; hard ones climb the
pyramid. `eval.evaluate.level_histogram` exposes the same numbers programmatically.

---

## 3. Expected wall-clock (from the paper)

| Phase | Cost |
|---|---|
| Offline captioning, 5.7M items, 32 GPUs | 20 GPU-hours |
| Offline embedding extraction | 0.6 GPU-hours |
| Online Stage-1, 5.7M candidates | 54.6 ms/query (naive full-resolution MRL: 179.2 ms) |
| Online Stage-2, `K = 100`, CoT ∥ QA | 0.82 s/query |
| **End-to-end per query** | **0.87 s** |

Stage-2 dominates online cost, and it scales linearly in `K` — which is exactly why
Stage-1 must produce a *small, high-recall* shortlist rather than a large one.

---

## 4. Differences you should expect

- **Sequential Stage-2.** This reference implementation loops over candidates. The paper
  runs `S_CoT` and `S_QA` in parallel with KV caching across a GPU cluster, so absolute
  latencies here will be much higher than the table above. Relative comparisons
  (Pyramid Rank vs. naive MRL, ablation rows) remain valid.
- **Decoding.** Greedy (`temperature = 0`) throughout, but MLLM outputs still vary with
  `transformers` version, dtype and attention implementation. Fix the versions before
  comparing runs.
- **Captions.** Retrieval quality depends on the captions, which depend on the captioning
  model's version. Regenerate captions with a single model version for a whole benchmark;
  do not mix.
- **`final_rescore`.** On by default (see the note in the README). It changes Stage-1
  rankings slightly relative to the literal pseudocode — for the better — at a small extra
  cost.
