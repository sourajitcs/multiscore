# Prompts

Every prompt MULTI-SCORE uses lives in [`multiscore/stage2/prompts.py`](../multiscore/stage2/prompts.py).
The captioning and QA-generation prompts are reproduced verbatim from the paper's appendix;
the bidirectional CoT templates follow the description in Section 3.2 and Figure 3.

---

## Offline captioning (modality → text)

Applied once, offline, to every non-text query and candidate. `<i>`, `<t>`, `<v>`, `<a>`
mark where the media is spliced in.

| Key | Modality | Model in the paper | Constant |
|---|---|---|---|
| `i` | image | Qwen3-VL-8B | `CAPTION_IMAGE` |
| `it` | image + text | Qwen3-VL-8B | `CAPTION_IMAGE_TEXT` |
| `v` | video | Qwen3-VL-8B | `CAPTION_VIDEO` |
| `a` | audio | Qwen2-Audio-7B | `CAPTION_AUDIO` |
| `av` | audio + video | Qwen2.5-Omni-7B | `CAPTION_AUDIO_VIDEO` |

```text
You are given an image: <i>
Generate a detailed caption describing the key objects, their attributes, spatial layout, and interactions.
Include information about the scene type, context, and any salient visual cues that convey intent or activity.
Output a detailed caption that captures the essential meaning of the image.
```

The caption-length ablation prefers **> 600 tokens** across image, video and audio, which
is why `Captioner` defaults to a generous `max_new_tokens`.

---

## Stage-2a: QA generation and answering

Generation, once per query:

```text
You are given a multimodal query: {query}
Generate {num_questions} Yes/No questions that capture essential semantics from the above query.
For each question, also provide its correct Yes/No answer. Format:
Q1:..? A: Yes/No; Q2:..? A: Yes/No; ..
```

Answering, once per (question, candidate) pair — the candidate is the *only* context, and
decoding is capped at 5 tokens:

```text
You are given a database candidate as context.

Candidate: {candidate}

Using only the information in the candidate, answer the following question.
Answer with only Yes or No.
{question}
```

`parse_qa_pairs` tolerates the `;`-separated single-line format, newline-separated `Q1.`/
`A:` variants and mixed capitalisation; `normalize_yes_no` maps anything unparseable to
`No` rather than crashing a long run.

---

## Stage-2b: bidirectional CoT

Shared preamble (`COT_INSTRUCTION`):

```text
You judge how well a retrieval query and a database candidate align.
Reason step by step: list the concepts in the source, check whether each one is supported by the target, and conclude with the alignment.
If every concept in the source is found in the target, the alignment is high; if concepts are missing or contradicted, the alignment is low.
Summarise the alignment in one word.
```

Then two demonstrations — one aligned pair, one misaligned pair — and the live pair,
terminated by the marker:

```text
Query: a person is swimming in some white water rapids
Candidate: A kayaker paddles hard through churning white water rapids while a swimmer in a red life vest is carried downstream beside the boat.
Reasoning: Concepts in the query: (1) a person, (2) swimming, (3) white water rapids. ...
Alignment: high

...

Query: {query}
Candidate: {candidate}
Reasoning:
Alignment in one word: <emb>
```

The c2q prompt is **not** a reordering of the q2c prompt: the candidate is presented first
*and* the demonstrations reason from candidate concepts to the query, because alignment is
asymmetric. Each entry in `COT_EXAMPLES` therefore carries both a `q2c_reasoning` and a
`c2q_reasoning`.

The hidden state of the token immediately before `<emb>` is pooled as the embedding
(`MLLMBackend.embed_with_marker`); nothing is decoded. Two demonstrations is the reported
optimum — three or four do not help — and `Stage2Config.num_cot_examples` controls it.

---

## Changing the prompts

Edit `prompts.py`; everything downstream reads from it. If you add a modality, add both the
prompt and its key to `CAPTION_PROMPTS` — `caption_prompt()` raises on unknown keys rather
than silently falling back, and `tests/test_stage2.py` checks every registered key.
