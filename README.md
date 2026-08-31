# MULTI-SCORE

**Zero-Shot Multimodal Retrieval with Multi-Scale Contextual Representations**

Sourajit Saha, Tejas Gokhale — University of Maryland, Baltimore County
[Project page](https://sourajitcs.github.io/multiscore/)

MULTI-SCORE is a **fine-tuning-free, two-stage** multimodal information retrieval (MMIR)
system. It retrieves across text, image, video and audio — unimodal, cross-modal and
composite query/candidate combinations — with no task-specific training, no auxiliary
fine-tuning and no extra annotation.

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

---

## Highlights

| | |
|---|---|
| **Zero-shot** | No fine-tuning, no retrieval training data, no dataset-specific pre-training. |
| **Efficient** | Pyramid Rank scores most candidates with 32-dim vectors instead of 1024-dim ones: 3.3× faster Stage-1 at 5.7M candidates with R@100 unchanged (86.4 → 86.2). |
| **Guaranteed** | The similarity upper bound is *admissible*, the bisection converges in `⌈log₂(w₀/ε)⌉` steps regardless of `N`, and the quality loss versus exhaustive full-resolution search is bounded by a controllable `ε`. |
| **Universal** | One pipeline for 12 MMIR tasks over 32 datasets (~5.7M pooled candidates) across text, image, video and audio. |
| **Explainable** | Stage-2 keeps its QA logs, so every ranking decision comes with the questions a candidate answered and the ones it missed. |
| **Backbone-agnostic** | The gains hold across Qwen2.5-Omni-3B/7B and Qwen2-Audio-7B; swap the backbone via one config field. |

---

## Install

```bash
git clone https://github.com/sourajitcs/multiscore.git
cd multiscore
pip install -e .              # core: numpy, pyyaml, tqdm
pip install -e ".[models]"    # + torch, transformers, and the media codecs
pip install -e ".[dev]"       # + pytest, ruff, black
```

Python ≥ 3.9. The core install is enough for Stage-1, the metrics, the test suite and the
toy demo; only `[models]` pulls in PyTorch and the Qwen checkpoints.

## Quickstart — no downloads, no GPU

```bash
python examples/toy_demo.py            # annotated walkthrough of both stages
python scripts/run_retrieval.py --config configs/toy.yaml
make test                              # 70+ unit tests, ~5 s
```

`configs/toy.yaml` runs the *whole* pipeline against a 24-item toy corpus using the
deterministic stub backends in `multiscore/models/stub.py`. It exercises every code path —
indexing, Pyramid Rank, both Stage-2 scores, re-ranking, metrics — in a few seconds on CPU.
**The retrieval numbers it prints are meaningless**; they exist to show the plumbing works.
Real numbers need the Qwen backbones and the public benchmarks.

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

---

## The method in 30 lines

**Stage-1 — Pyramid Rank.** Qwen3-MRL emits nested (Matryoshka) embeddings at `L = 6`
levels, from `d = 32` up to `D = 1024`, where level `ℓ` is simply the first
`2^(ℓ-1)·d` coordinates of the level-`L` vector. Writing `z^(ℓ)` for a level-`ℓ` prefix
zero-padded back to `D`, the exact decomposition

```
⟨x_q^(L), x_c^(L)⟩ = ⟨z_q^(ℓ), z_c^(ℓ)⟩ + ⟨x_q^(L) − z_q^(ℓ), x_c^(L) − z_c^(ℓ)⟩
                      └──── known ────┘   └────────── unknown ──────────┘
```

plus Cauchy–Schwarz — and the fact that `z^(ℓ) ⊥ (x^(L) − z^(ℓ))` with `‖x^(L)‖ = 1`, so
the residual norm is `√(1 − ‖z^(ℓ)‖²)` — gives a bound computable entirely from short
vectors:

```
⟨x_q^(L), x_c^(L)⟩  ≤  U_q,c^(ℓ)  =  ⟨z_q^(ℓ), z_c^(ℓ)⟩ + √((1 − ‖z_q^(ℓ)‖²)(1 − ‖z_c^(ℓ)‖²))
```

Algorithm 1 bisects a threshold `τ` over `[−1, 1]`, keeps the candidates whose bound clears
it, and climbs one level each time the surviving set is still larger than `K`. Because the
bound is admissible, nothing relevant is ever pruned; because each step halves the interval,
it terminates in `⌈log₂(w₀/ε)⌉` steps regardless of `N`.

**Stage-2 — multimodal re-ranking.** Two complementary scores over the `K` survivors:

- `S_CoT` — few-shot chain-of-thought prompts in *both* directions (query→candidate and
  candidate→query), each terminated by an `<emb>` marker whose preceding hidden state is
  pooled as an embedding; the score is the cosine between the two.
- `S_QA` — the MLLM turns the query into `M` discriminative yes/no questions, then answers
  them using each candidate as the only context; the score is the answer accuracy.
- Combined as `S = α·S_CoT + (1 − α)·S_QA`.

`docs/METHOD.md` walks through the derivation, the guarantees and the design choices.

---

## Repository layout

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

---

## Running on real benchmarks

Data never ships with this repository and nothing here downloads it — see
[`docs/DATA.md`](docs/DATA.md) for how to obtain each corpus and convert it into the JSONL
schema below.

```jsonl
{"id": "c001", "video": "clips/c001.mp4", "caption": "A person in a red life vest swims ..."}
{"id": "q1",   "text": "a person is swimming in some white water rapids", "positives": ["c001"]}
```

Then, per benchmark:

```bash
# 1. Offline: caption every non-text item once (shardable across GPUs)
python scripts/preprocess_caption.py \
    --input  data/msrvtt/t2v/candidates.jsonl \
    --output data/msrvtt/t2v/candidates.captioned.jsonl \
    --backend qwen3-vl-8b --shard 0 --num-shards 8

# 2. Offline: build the MRL index
python scripts/build_index.py \
    --candidates data/msrvtt/t2v/candidates.captioned.jsonl \
    --index-dir runs/index/msrvtt_t2v --embedder qwen3-mrl

# 3. Online: retrieve (Stage-1 + Stage-2)
python scripts/run_retrieval.py --config configs/video.yaml \
    --queries data/msrvtt/t2v/test.queries.jsonl \
    --candidates data/msrvtt/t2v/candidates.captioned.jsonl \
    --index-dir runs/index/msrvtt_t2v --output-dir runs/msrvtt_t2v

# 4. Score a saved run
python scripts/evaluate.py --runs runs/msrvtt_t2v/runs.jsonl \
    --queries data/msrvtt/t2v/test.queries.jsonl
```

Useful flags: `--stage1-only` (skip re-ranking), `--naive-stage1` (exhaustive
full-resolution baseline), `--top-k`, `--epsilon`, `--alpha`, `--limit`.

### Models used in the paper

| Role | Model | Where |
|---|---|---|
| Stage-1 embeddings | `Qwen/Qwen3-Embedding-0.6B` (MRL, `L=6`, `d=32`, `D=1024`) | `models/qwen3_mrl.py` |
| Image / image+text / video captioning | `Qwen3-VL-8B` | `data/captioning.py` |
| Audio captioning | `Qwen2-Audio-7B` | `data/captioning.py` |
| Audio+video captioning | `Qwen2.5-Omni-7B` | `data/captioning.py` |
| Stage-2 `S_CoT` and `S_QA` | `Qwen2.5-Omni-7B` | `models/qwen_mllm.py` |

No checkpoint is bundled or downloaded by this repository; `transformers` fetches them on
first use.

### Key hyper-parameters

| Field | Default | Meaning |
|---|---|---|
| `stage1.top_k` | `100` | `K`, the shortlist Stage-2 re-ranks. |
| `stage1.epsilon` | `0.02` | Bisection tolerance; larger = faster and coarser. |
| `stage1.mrl.base_dim` / `num_levels` | `32` / `6` | The pyramid: `D = d·2^(L−1) = 1024`. |
| `stage2.alpha` | `0.6` | Weight on `S_CoT` versus `S_QA`. |
| `stage2.num_questions` | `7` | `M`; the ablation prefers `≥ 7`. |
| `stage2.num_cot_examples` | `2` | In-context CoT demonstrations; 2 is the reported optimum. |

---

## Reproducing the paper's analyses

```bash
python scripts/reproduce/ablation.py       --config configs/video.yaml   # component ablation
python scripts/reproduce/sweep_alpha.py    --config configs/mbeir.yaml   # α sweep
python scripts/reproduce/sweep_epsilon.py  --config configs/mbeir.yaml   # ε vs. cost
python scripts/reproduce/sweep_topk.py     --config configs/video.yaml   # K vs. recall/runtime
python scripts/reproduce/stage1_scaling.py --sizes 10000 100000 1000000  # Stage-1 speed-up curve
```

`stage1_scaling.py` needs no data at all — the Stage-1 saving is a property of the pyramid
geometry, so it can be measured on synthetic embeddings. Everything else takes a config and
a benchmark. [`docs/REPRODUCING.md`](docs/REPRODUCING.md) lists the compute each analysis
needs and which table or figure it corresponds to.

### Reported costs (from the paper)

| | |
|---|---|
| Offline captioning (5.7M items, 32 GPUs) | 20 GPU-hours |
| Offline embedding extraction | 0.6 GPU-hours |
| Online Stage-1 at 5.7M candidates | 54.6 ms/query (naive MRL: 179.2 ms) |
| Online Stage-2, `K = 100`, CoT ∥ QA | 0.82 s/query |
| **Total per query** | **0.87 s** |

Hardware: Stage-1 on 8 × 4 NVIDIA H100 (batch 128); Stage-2 on 4 × 16 NVIDIA L40S
(CoT batch 8, QA batch 4); greedy decoding, 5 max tokens per QA answer.

---

## Notes on this implementation

- **`final_rescore` (on by default).** Algorithm 1 ranks the survivors by the upper bounds
  `U_q`, which are collected at whatever level each candidate was last evaluated at, while
  the ε-bounded correctness argument in the appendix is written in terms of `U^(L)`. This
  implementation therefore re-scores the surviving set at level `L` before the final sort —
  `|I|·D` extra MACs on a set that is only a small multiple of `K`. Set
  `stage1.final_rescore: false` to reproduce the pseudocode line for line;
  `tests/test_pyramid_rank.py` covers both modes.
- **`exact_padding`.** `upper_bound.py` ships both the fast prefix-slicing path and a literal
  implementation that materialises the zero-padded `z` vectors, and a test asserts they agree —
  useful when checking the equations against the paper.
- **`alpha` per modality.** The paper's Implementation Details give `α = 0.6` for image
  retrieval and `α = 0.3` for video/audio, while the α-sweep figure and its discussion state
  the opposite assignment. The configs follow Implementation Details
  (`configs/mbeir.yaml`: `0.6`; `configs/video.yaml` and `configs/audio.yaml`: `0.3`);
  `scripts/reproduce/sweep_alpha.py` re-derives the value from data.
- **Stub backends.** `multiscore/models/stub.py` provides a hashing MRL encoder and a
  lexical-overlap "MLLM" so that CI, the tests and the demo need no checkpoints. They are
  not baselines and their scores mean nothing.
- **Stage-1 cost model.** `Stage1Output.cost` counts every row a level pass touched,
  including rows already pruned but not yet compacted away, so `speedup` is conservative.
  On this NumPy/CPU reference implementation expect a 2–3× MAC saving; the paper's 3.3× is
  end-to-end wall-clock for a batched GPU deployment at 5.7M candidates. Filtering only pays
  off when `K ≪ N` — on a few thousand candidates the bisection overhead can exceed
  exhaustive scoring, which `tests/test_pyramid_rank.py` pins down in both directions.
- **Batching.** Stage-2 loops over candidates sequentially in this reference implementation.
  The paper runs `S_CoT` and `S_QA` in parallel with KV caching, which is where the
  `max(6.5, 8.2) ms` per-candidate figure comes from; `MLLMBackend.batch_generate` is the
  hook for a batched backend.

## Citation

```bibtex
@inproceedings{saha2026multiscore,
  title     = {Zero-Shot Multimodal Retrieval with Multi-Scale Contextual Representations},
  author    = {Saha, Sourajit and Gokhale, Tejas},
  booktitle = {Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE). The datasets and model checkpoints referenced here carry their
own licences.

## Acknowledgments

This work was funded in part by DARPA's SciFy program (agreement HR00112520301). We
acknowledge high-performance computing support from UMBC HPCF and a Lambda Inc. award.
The views and conclusions are those of the authors and should not be interpreted as
representing official policies or endorsements of employers, funding agencies, or
governments.
