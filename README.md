# MULTI-SCORE: Zero-Shot Multimodal Retrieval with Multi-Scale Contextual Representations

<p align="center">
  <a href="https://sourajitcs.github.io/multiscore/">
    <img src="https://img.shields.io/badge/%F0%9F%94%A5_Accepted_at-ACL_2026_%F0%9F%94%A5-b12a00?style=for-the-badge&labelColor=ffb300" alt="Accepted at ACL 2026">
  </a>
</p>

<!-- Add the arXiv badge once the preprint is posted:
[![Paper](https://img.shields.io/badge/arXiv-XXXX.XXXXX-red)](https://arxiv.org/abs/XXXX.XXXXX) -->

[![ACL 2026](https://img.shields.io/badge/ACL%202026-Main-blue)](https://sourajitcs.github.io/multiscore/)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://sourajitcs.github.io/multiscore/)
[![Models](https://img.shields.io/badge/HuggingFace-Qwen%20backbones-yellow)](https://huggingface.co/Qwen)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-12%20tasks%20%7C%2032%20datasets-orange)](docs/DATA.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

Official implementation of **"Zero-Shot Multimodal Retrieval with Multi-Scale Contextual Representations"**

Sourajit Saha, Tejas Gokhale — University of Maryland, Baltimore County

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Method](#method)
- [Data](#data)
- [Models](#models)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Reproducing the Paper](#reproducing-the-paper)
- [Implementation Notes](#implementation-notes)
- [Results](#results)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Overview

MULTI-SCORE is a **fine-tuning-free, two-stage** multimodal information retrieval (MMIR)
system. It retrieves across text, image, video and audio — unimodal, cross-modal and
composite query/candidate combinations — with no task-specific training, no auxiliary
fine-tuning and no extra annotation. We introduce:

- **Pyramid Rank**: a Stage-1 candidate filter that walks up a Matryoshka (MRL) embedding
  pyramid, eliminating most of the database with 32-dimensional vectors instead of
  1024-dimensional ones, with an admissible similarity upper bound and an `ε`-bounded
  quality guarantee.
- **Bidirectional-CoT Embedding Score (`S_CoT`)**: few-shot chain-of-thought prompts run in
  both query→candidate and candidate→query directions, embedded via the hidden state
  preceding an `<emb>` marker.
- **Question Answering Relevance Score (`S_QA`)**: the query is turned into yes/no
  questions, and a candidate is scored by how accurately they can be answered from it —
  which doubles as a natural-language explanation of the ranking.

```
                     N candidates (5.7M in the universal setting)
                                    │
   Stage-1  ┌─────────────────────────────────────────────────┐
  Pyramid   │  bisect a relevance threshold τ while climbing  │   admissible
   Rank     │  the Matryoshka pyramid: score survivors only,  │   ε-bounded
            │  at the shortest level that can still decide    │   O(log 1/ε) steps
            └─────────────────────────────────────────────────┘
                                    │  top-K  (K ≪ N)
   Stage-2  ┌─────────────────────────────────────────────────┐
   Multi-   │  S_CoT  bidirectional chain-of-thought embeddings│  native modality,
   modal    │  S_QA   yes/no question answering accuracy       │  no captions,
  re-rank   │  S = α·S_CoT + (1−α)·S_QA                        │  interpretable
            └─────────────────────────────────────────────────┘
                                    │
                            re-ranked top-K
```

Stage-1 is cheap and text-only: it runs on captions computed **once, offline**, and never
touches full-resolution embeddings for the whole database. Stage-2 is expensive but only
ever sees `K` candidates, and consumes their **native** modality, so nothing is lost to the
caption bottleneck.

## Key Features

- **Zero-shot** — no fine-tuning, no retrieval training data, no dataset-specific pre-training
- **Efficient** — 3.3× faster Stage-1 at 5.7M candidates with R@100 unchanged (86.4 → 86.2)
- **Guaranteed** — admissible pruning, convergence in `⌈log₂(w₀/ε)⌉` steps independent of `N`, and an `ε`-bounded quality loss versus exhaustive search
- **Universal** — one pipeline for 12 MMIR tasks over 32 datasets (~5.7M pooled candidates) across text, image, video and audio
- **Explainable** — Stage-2 keeps its QA logs, so every ranking decision comes with the questions a candidate answered and the ones it missed
- **Backbone-agnostic** — gains hold across Qwen2.5-Omni-3B/7B and Qwen2-Audio-7B; swap the backbone with one config field
- **Runs without downloads** — deterministic stub backends exercise the whole pipeline on CPU in seconds

## Installation

### Requirements
- Python 3.9+
- For the real backbones: CUDA-capable GPU (the paper uses H100s for Stage-1 and L40S for Stage-2)
- No GPU or checkpoint is needed for the tests, the toy demo, or the Stage-1 scaling study

### Setup

```bash
# Clone the repository
git clone https://github.com/sourajitcs/multiscore.git
cd multiscore

# Core install: numpy, pyyaml, tqdm
pip install -e .

# With the Qwen backbones (torch, transformers, media codecs)
pip install -e ".[models]"

# With the dev tooling (pytest, ruff, black)
pip install -e ".[dev]"
```

Or via the Makefile:

```bash
make install          # core + dev
make install-models   # core + dev + models
```

## Quick Start

No downloads, no GPU:

```bash
python examples/toy_demo.py                            # annotated walkthrough of both stages
python scripts/run_retrieval.py --config configs/toy.yaml
make test                                              # 71 unit tests, ~5 s
```

`configs/toy.yaml` runs the *whole* pipeline against a 24-item toy corpus using the
deterministic stub backends in `multiscore/models/stub.py`. It exercises every code path —
indexing, Pyramid Rank, both Stage-2 scores, re-ranking, metrics.
**The retrieval numbers it prints are meaningless**; they exist to show the plumbing works.

```text
[Stage-1] Pyramid Rank
  bisection steps      : 6
  levels visited       : [1, 2, 3, 3, 4, 4]
  survivors per step   : [14, 9, 9, 5, 5, 5]

[Stage-2] alpha * S_CoT + (1 - alpha) * S_QA   (alpha=0.6)
    1. c001  S_rerank=+0.7769  S_CoT=+0.9615  S_QA=0.50
  why the top candidate scored what it did:
      [OK  ] Does the candidate involve person?  gold=Yes pred=Yes
      [MISS] Does the candidate involve swimming?  gold=Yes pred=No

[Scaling] synthetic database, N=50,000, K=100, eps=0.02
   mean cost saving     : 3.62x fewer MACs than scoring all 50,000 candidates at D=256
```

The last section is the one quantitative claim the demo can make honestly: 24 candidates
for a budget of 5 leaves nothing to filter, so the saving only shows up once `K ≪ N`. It
uses synthetic unit-norm embeddings, because the saving follows from the geometry of the
pyramid rather than the semantics of the vectors.

## Method

### Stage-1: Pyramid Rank

Qwen3-MRL emits nested (Matryoshka) embeddings at `L = 6` levels, from `d = 32` up to
`D = 1024`, where level `ℓ` is the first `2^(ℓ-1)·d` coordinates of the level-`L` vector.
Writing `z^(ℓ)` for a level-`ℓ` prefix zero-padded back to `D`:

```
⟨x_q^(L), x_c^(L)⟩ = ⟨z_q^(ℓ), z_c^(ℓ)⟩ + ⟨x_q^(L) − z_q^(ℓ), x_c^(L) − z_c^(ℓ)⟩
                      └──── known ────┘   └────────── unknown ──────────┘
```

Cauchy–Schwarz bounds the remainder, and since `z^(ℓ) ⊥ (x^(L) − z^(ℓ))` with `‖x^(L)‖ = 1`,
the residual norm is `√(1 − ‖z^(ℓ)‖²)` — known in closed form. This gives a bound computable
entirely from short vectors:

```
⟨x_q^(L), x_c^(L)⟩  ≤  U_q,c^(ℓ)  =  ⟨z_q^(ℓ), z_c^(ℓ)⟩ + √((1 − ‖z_q^(ℓ)‖²)(1 − ‖z_c^(ℓ)‖²))
```

Algorithm 1 bisects a threshold `τ` over `[−1, 1]`, keeps the candidates whose bound clears
it, and climbs one level each time the surviving set is still larger than `K`.

| Guarantee | Statement |
|-----------|-----------|
| Admissibility | A candidate is dropped only when `U^(ℓ) < τ`, and `U` bounds the true similarity from above — nothing relevant is pruned |
| Convergence | Each step halves the interval: at most `⌈log₂(w₀/ε)⌉` steps, independent of `N` |
| ε-bounded correctness | Any discarded item `j` satisfies `⟨q, c_j⟩ ≤ ⟨q, c_K⟩ + ε` |

### Stage-2: Multimodal Re-ranking

| Score | What it computes |
|-------|------------------|
| `S_CoT` | Cosine between `z_q2c` and `z_c2q`, the hidden states preceding `<emb>` in the query→candidate and candidate→query CoT prompts |
| `S_QA` | Accuracy of `M` query-derived yes/no questions answered with the candidate as the only context |
| `S_rerank` | `α · S_CoT + (1 − α) · S_QA` |

See [`docs/METHOD.md`](docs/METHOD.md) for the full derivation and
[`docs/PROMPTS.md`](docs/PROMPTS.md) for every prompt.

## Data

**No data ships with this repository and nothing here downloads it.** See
[`docs/DATA.md`](docs/DATA.md) for how to obtain each corpus and convert it.

### Data Format

```jsonl
{"id": "c001", "video": "clips/c001.mp4", "caption": "A person in a red life vest swims ..."}
{"id": "q1",   "text": "a person is swimming in some white water rapids", "positives": ["c001"]}
```

| Field | Description |
|-------|-------------|
| `id` | Unique within the file; candidate ids must match those in `positives` |
| `text` | Raw text, if the item has any |
| `image` / `video` / `audio` | Paths to media |
| `caption` | Filled in by `scripts/preprocess_caption.py`; this is what Stage-1 embeds |
| `positives` | Queries only: the relevant candidate ids |

### Benchmark Suite

12 tasks, 32 (dataset, task) pairs, ~5.7M pooled candidates.

| Task | Datasets | Source |
|------|----------|--------|
| `t → i` | VisualNews, MSCOCO, Fashion200K | [M-BEIR](https://huggingface.co/datasets/TIGER-Lab/M-BEIR) |
| `t → i` | Urban-1K, Flickr30K | [Long-CLIP](https://github.com/beichenzbc/Long-CLIP), [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/) |
| `t → t` | WebQA | M-BEIR |
| `t → (i,t)` | EDIS, WebQA | M-BEIR |
| `t → v` | MSRVTT-1kA, MSVD, LSMDC, DiDeMo | original releases |
| `i → t` | VisualNews, MSCOCO, Fashion200K, Urban-1K, Flickr30K | M-BEIR / originals |
| `i → i` | NIGHTS | [DreamSim](https://dreamsim-nights.github.io/) |
| `(i,t) → t` | OVEN, InfoSeek | M-BEIR |
| `(i,t) → i` | FashionIQ, CIRR, GeneCIS | M-BEIR / [GeneCIS](https://github.com/facebookresearch/genecis) |
| `(i,t) → (i,t)` | OVEN, InfoSeek | M-BEIR |
| `v → t` | MSRVTT-1kA, MSVD, LSMDC, DiDeMo | original releases |
| `t → a` | AudioCaps, Clotho | [AudioCaps](https://audiocaps.github.io/), [Clotho](https://zenodo.org/records/3490684) |
| `t → (a,v)` | AudioCaps | AudioCaps + source video |

The same table lives in code as `multiscore/data/registry.py`, including test-time
query/candidate counts.

## Models

MULTI-SCORE is training-free, so it releases **no checkpoints of its own** — it composes
off-the-shelf Qwen models, fetched by `transformers` on first use.

| Role | Model | HuggingFace Link | Wrapper |
|------|-------|------------------|---------|
| Stage-1 embeddings (MRL, `L=6`, `d=32`, `D=1024`) | Qwen3-Embedding-0.6B | [Link](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | `models/qwen3_mrl.py` |
| Image / image+text / video captioning | Qwen3-VL-8B-Instruct | [Link](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `data/captioning.py` |
| Audio captioning | Qwen2-Audio-7B-Instruct | [Link](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) | `data/captioning.py` |
| Audio+video captioning, Stage-2 `S_CoT` and `S_QA` | Qwen2.5-Omni-7B | [Link](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) | `models/qwen_mllm.py` |
| Stage-2 ablation backbone | Qwen2.5-Omni-3B | [Link](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | `models/qwen_mllm.py` |

**Backend aliases** (usable directly in configs): `qwen3-mrl`, `qwen2.5-omni-7b`,
`qwen2.5-omni-3b`, `qwen3-vl-8b`, `qwen2-audio-7b`, `qwen3-omni-30b-thinking`, plus
`stub-mrl` / `stub-mllm` for the dependency-free stubs.

## Usage

### Step 1 — Offline captioning

Every non-text item is turned into text once, offline. Shardable across GPUs.

```bash
python scripts/preprocess_caption.py \
    --input  data/msrvtt/t2v/candidates.jsonl \
    --output data/msrvtt/t2v/candidates.captioned.jsonl \
    --backend qwen3-vl-8b --shard 0 --num-shards 8
```

### Step 2 — Offline indexing

```bash
python scripts/build_index.py \
    --candidates data/msrvtt/t2v/candidates.captioned.jsonl \
    --index-dir runs/index/msrvtt_t2v \
    --embedder qwen3-mrl
```

### Step 3 — Online retrieval

```bash
python scripts/run_retrieval.py --config configs/video.yaml \
    --queries data/msrvtt/t2v/test.queries.jsonl \
    --candidates data/msrvtt/t2v/candidates.captioned.jsonl \
    --index-dir runs/index/msrvtt_t2v \
    --output-dir runs/msrvtt_t2v
```

Useful flags: `--stage1-only` (skip re-ranking), `--naive-stage1` (exhaustive
full-resolution baseline), `--top-k`, `--epsilon`, `--alpha`, `--limit`.

### Step 4 — Evaluation

```bash
python scripts/evaluate.py \
    --runs runs/msrvtt_t2v/runs.jsonl \
    --queries data/msrvtt/t2v/test.queries.jsonl
```

### Python API

```python
from multiscore import MultiScore, MultiScoreConfig
from multiscore.data.loaders import load_candidates, load_queries

engine = MultiScore(MultiScoreConfig())          # loads Qwen3-MRL + Qwen2.5-Omni lazily
candidates = load_candidates("data/msrvtt/t2v/candidates.captioned.jsonl")
queries = load_queries("data/msrvtt/t2v/test.queries.jsonl")

store = engine.index_candidates(candidates)
result = engine.retrieve(queries[0], store, candidates=candidates)

print(result.ranked_ids[:5])
print(result.explain())                          # QA logs behind the ranking
```

### Configuration

Configs live in `configs/`: `default.yaml`, `mbeir.yaml` (image-text), `video.yaml`,
`audio.yaml`, `universal.yaml` (all 32 datasets pooled) and `toy.yaml`.

| Field | Default | Meaning |
|-------|---------|---------|
| `stage1.top_k` | `100` | `K`, the shortlist Stage-2 re-ranks |
| `stage1.epsilon` | `0.02` | Bisection tolerance; larger = faster and coarser |
| `stage1.mrl.base_dim` / `num_levels` | `32` / `6` | The pyramid: `D = d·2^(L−1) = 1024` |
| `stage1.final_rescore` | `true` | Re-score survivors at level `L` before sorting |
| `stage2.backend` | `qwen2.5-omni-7b` | Stage-2 MLLM |
| `stage2.alpha` | `0.6` | Weight on `S_CoT` versus `S_QA` |
| `stage2.num_questions` | `7` | `M`; the ablation prefers `≥ 7` |
| `stage2.num_cot_examples` | `2` | In-context CoT demonstrations; 2 is the reported optimum |

## Repository Structure

```
multiscore/
├── config.py               dataclass configs; mirrors configs/*.yaml
├── pipeline.py             MultiScore: the two stages glued together
├── stage1/
│   ├── mrl.py              nested prefixes, zero-padding, cached prefix norms
│   ├── upper_bound.py      U_q,c^(ℓ) — fast path and a literal reference version
│   ├── pyramid_rank.py     Algorithm 1, with cost/level/threshold diagnostics
│   └── naive.py            exhaustive full-resolution baseline
├── stage2/
│   ├── prompts.py          every prompt: captioning, QA generation, bidirectional CoT
│   ├── cot_score.py        S_CoT
│   ├── qa_score.py         S_QA (+ interpretable per-question logs)
│   └── rerank.py           the convex combination and the re-ordering
├── models/                 Qwen3-MRL, Qwen MLLMs, and dependency-free stubs
├── data/                   JSONL schema, loaders, offline captioning, benchmark registry
├── index/                  build / save / memory-map the MRL index
└── eval/                   R@k, nDCG@k, efficiency reports

scripts/                    caption → index → retrieve → evaluate, plus reproduce/
configs/                    default, mbeir, video, audio, universal, toy
docs/                       METHOD.md, DATA.md, REPRODUCING.md, PROMPTS.md
tests/                      guarantees, prompts, metrics, end-to-end
examples/                   toy corpus and an annotated demo
```

## Reproducing the Paper

```bash
python scripts/reproduce/ablation.py       --config configs/video.yaml   # component ablation
python scripts/reproduce/sweep_alpha.py    --config configs/mbeir.yaml   # α sweep
python scripts/reproduce/sweep_epsilon.py  --config configs/mbeir.yaml   # ε vs. cost
python scripts/reproduce/sweep_topk.py     --config configs/video.yaml   # K vs. recall/runtime
python scripts/reproduce/stage1_scaling.py --sizes 10000 100000 1000000  # Stage-1 speed-up curve
```

| Script | Reproduces | Needs |
|--------|------------|-------|
| `reproduce/ablation.py` | naive MRL → + Pyramid Rank → + `S_CoT` → + `S_QA` → full | 1 benchmark, Stage-2 backbone |
| `reproduce/sweep_alpha.py` | The α sweep and its per-modality optimum | 1 benchmark, Stage-2 backbone |
| `reproduce/sweep_epsilon.py` | ε vs. R@k and Stage-1 cost | 1 benchmark, Stage-1 only |
| `reproduce/sweep_topk.py` | K vs. recall, candidate survival and Stage-1 runtime | 1 benchmark, Stage-1 only |
| `reproduce/stage1_scaling.py` | Stage-1 speed-up vs. database size | nothing |

`sweep_alpha.py` computes `S_CoT` and `S_QA` once per pair and re-mixes them, so a full
sweep costs one Stage-2 pass rather than one per α. See
[`docs/REPRODUCING.md`](docs/REPRODUCING.md) for what each analysis costs.

## Implementation Notes

- **`final_rescore` (on by default).** Algorithm 1 ranks survivors by upper bounds `U_q`
  collected at mixed levels, while the ε-correctness argument in the appendix is written in
  terms of `U^(L)`. This implementation re-scores the surviving set at level `L` before the
  final sort — `|I|·D` extra MACs on a small set. Set `stage1.final_rescore: false` to
  reproduce the pseudocode line for line; `tests/test_pyramid_rank.py` covers both modes.
- **`exact_padding`.** `upper_bound.py` ships both the fast prefix-slicing path and a literal
  implementation that materialises the zero-padded `z` vectors, and a test asserts they agree.
- **`alpha` per modality.** The paper's Implementation Details give `α = 0.6` for image
  retrieval and `α = 0.3` for video/audio, while the α-sweep figure and its discussion state
  the opposite assignment. The configs follow Implementation Details; `sweep_alpha.py`
  re-derives the value from data.
- **Stage-1 cost model.** `Stage1Output.cost` counts every row a level pass touched,
  including rows pruned but not yet compacted away, so `speedup` is conservative. On this
  NumPy/CPU reference expect a 2–3× MAC saving; the paper's 3.3× is end-to-end wall-clock
  for a batched GPU deployment at 5.7M candidates. Filtering only pays off when `K ≪ N`.
- **Batching.** Stage-2 loops over candidates sequentially here. The paper runs `S_CoT` and
  `S_QA` in parallel with KV caching; `MLLMBackend.batch_generate` is the hook for a batched
  backend.
- **Stub backends.** `multiscore/models/stub.py` provides a hashing MRL encoder and a
  lexical-overlap "MLLM" so tests, CI and the demo need no checkpoints. They are not
  baselines and their scores mean nothing.

## Results

For full results across 12 MMIR tasks and 32 datasets, please refer to our
[paper](https://sourajitcs.github.io/multiscore/). Reported compute costs:

| Phase | Cost |
|-------|------|
| Offline captioning (5.7M items, 32 GPUs) | 20 GPU-hours |
| Offline embedding extraction | 0.6 GPU-hours |
| Online Stage-1 at 5.7M candidates | 54.6 ms/query (naive MRL: 179.2 ms) |
| Online Stage-2, `K = 100`, CoT ∥ QA | 0.82 s/query |
| **Total per query** | **0.87 s** |

Hardware: Stage-1 on 8 × 4 NVIDIA H100 (batch 128); Stage-2 on 4 × 16 NVIDIA L40S
(CoT batch 8, QA batch 4); greedy decoding, 5 max tokens per QA answer.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{saha2026multiscore,
  title={Zero-Shot Multimodal Retrieval with Multi-Scale Contextual Representations},
  author={Saha, Sourajit and Gokhale, Tejas},
  booktitle={Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026},
  url={https://sourajitcs.github.io/multiscore/},
}
```

## License

MIT — see [LICENSE](LICENSE). The datasets and model checkpoints referenced here carry their
own licences.

## Acknowledgments

- [Qwen](https://github.com/QwenLM/Qwen) for the embedding and omni-modal base models
- [Matryoshka Representation Learning](https://github.com/RAIVNLab/MRL) for the nested representation hierarchy
- [M-BEIR / UniIR](https://huggingface.co/datasets/TIGER-Lab/M-BEIR) for the unified multimodal retrieval benchmark
- [HuggingFace Transformers](https://github.com/huggingface/transformers) for model serving


