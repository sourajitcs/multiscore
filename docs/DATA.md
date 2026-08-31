# Data

**This repository ships no data and downloads none.** Every benchmark below is public, but
several require accepting a licence or requesting access from the original authors. Obtain
each corpus from its own source, then convert it into the JSONL schema described here.

---

## The schema

Two files per (dataset, task), plus optional relevance judgements.

`candidates.jsonl` — the retrieval database:

```jsonl
{"id": "coco:val:139", "image": "images/COCO_val2014_000000000139.jpg"}
{"id": "msrvtt:7010", "video": "clips/video7010.mp4"}
{"id": "audiocaps:91139", "audio": "wav/91139.wav"}
{"id": "webqa:d20431", "text": "The Cheyenne Mountain Complex is ...", "image": "imgs/20431.jpg"}
```

`test.queries.jsonl` — queries with their gold candidate ids:

```jsonl
{"id": "q1", "text": "a person is swimming in some white water rapids", "positives": ["msrvtt:7010"]}
{"id": "q2", "image": "imgs/dress.jpg", "text": "make it sleeveless", "positives": ["fiq:8842"]}
```

| Field | Meaning |
|---|---|
| `id` | Unique within the file. Candidate ids must match those in `positives`. |
| `text` | Raw text, if the item has any. |
| `image` / `video` / `audio` | Paths to media, relative to wherever you run from. |
| `caption` | Filled in by `scripts/preprocess_caption.py`; this is what Stage-1 embeds. |
| `positives` | Queries only: the relevant candidate ids. |

Any other key is preserved in `Item.meta` and round-trips through the loaders, so dataset
bookkeeping (splits, categories, original ids) can ride along.

Relevance judgements may instead live in a separate file, JSONL
(`{"query_id": ..., "candidate_id": ..., "relevance": 1}`) or TREC-style
(`qid _ did rel`), passed with `--qrels`.

The modality of an item is inferred from which fields are populated —
`t`, `i`, `it`, `v`, `a`, `av` — and decides which captioning prompt applies. See
`multiscore/data/schema.py`.

---

## Expected layout

`multiscore.data.loaders.resolve_split` assumes:

```
data/
└── <dataset>/
    └── <task>/
        ├── candidates.jsonl            # raw
        ├── candidates.captioned.jsonl  # after step 1
        ├── test.queries.jsonl
        └── test.qrels.jsonl            # optional
```

Nothing enforces this; every script takes explicit paths.

---

## The benchmark suite

12 tasks, 32 (dataset, task) pairs, ~5.7M pooled candidates. `multiscore/data/registry.py`
holds the same table in code, including the test-time query/candidate counts.

| Task | Datasets | Source |
|---|---|---|
| `t → i` | VisualNews, MSCOCO, Fashion200K | [M-BEIR](https://huggingface.co/datasets/TIGER-Lab/M-BEIR) |
| `t → i` | Urban-1K | [Long-CLIP](https://github.com/beichenzbc/Long-CLIP) |
| `t → i` | Flickr30K | [Flickr30K](https://shannon.cs.illinois.edu/DenotationGraph/) |
| `t → t` | WebQA | M-BEIR |
| `t → (i,t)` | EDIS, WebQA | M-BEIR |
| `t → v` | MSRVTT-1kA, MSVD, LSMDC, DiDeMo | original releases |
| `i → t` | VisualNews, MSCOCO, Fashion200K, Urban-1K, Flickr30K | M-BEIR / originals |
| `i → i` | NIGHTS | M-BEIR / [DreamSim](https://dreamsim-nights.github.io/) |
| `(i,t) → t` | OVEN, InfoSeek | M-BEIR |
| `(i,t) → i` | FashionIQ, CIRR, GeneCIS | M-BEIR / [GeneCIS](https://github.com/facebookresearch/genecis) |
| `(i,t) → (i,t)` | OVEN, InfoSeek | M-BEIR |
| `v → t` | MSRVTT-1kA, MSVD, LSMDC, DiDeMo | original releases |
| `t → a` | AudioCaps, Clotho | [AudioCaps](https://audiocaps.github.io/), [Clotho](https://zenodo.org/records/3490684) |
| `t → (a,v)` | AudioCaps | AudioCaps + the source YouTube video |

Notes:

- **M-BEIR** already provides a unified format; a converter mostly needs to rename fields
  and prefix ids with the dataset name to keep them unique in the pooled setting.
- **LSMDC** is distributed under a restricted licence — request access from the authors.
  The registry records it with `None` counts.
- **GeneCIS** evaluates against a small per-query gallery (10–15 candidates), so Stage-1 is
  effectively a no-op there and Stage-2 does the work.
- **AudioCaps `t → (a,v)`** needs the source videos as well as the audio; some YouTube
  sources have since been removed, so recall is measured over whatever is retrievable.

---

## Universal retrieval

The universal setting pools every candidate file into a single database, so a query must be
matched against ~5.7M items spanning all modalities. Concatenate the captioned candidate
files (ids must already be globally unique), build one index over the result, and point
`configs/universal.yaml` at it. This is the setting where Pyramid Rank matters most: the
paper measures a 3.3× Stage-1 speed-up at that scale with no loss in R@100.

---

## Offline pre-processing cost

Captioning is a one-time cost, and the only expensive part of indexing:

| Modality | Model | Avg. time per item |
|---|---|---|
| Image | Qwen3-VL-8B | 0.18 s |
| Video | Qwen2.5-Omni-7B | 0.65 s |
| Audio | Qwen2-Audio-7B | 0.35 s |

≈20 GPU-hours for 5.7M items across 32 GPUs, plus 0.6 GPU-hours for embedding extraction
(0.012 s per item). Use `--shard i --num-shards n` in `scripts/preprocess_caption.py` to
split the work; shards write independent JSONL files that can simply be concatenated.
