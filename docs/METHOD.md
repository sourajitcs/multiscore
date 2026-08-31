# Method

A code-oriented walkthrough of MULTI-SCORE. Section numbers refer to the paper.

---

## 1. Setup

Given a query `q` and a database `C = {c_1 ... c_N}`, return the `K` candidates most
relevant to `q`. Queries and candidates may each be text, image, video, audio, or a
combination — 12 distinct task signatures in the paper's benchmark suite.

The pipeline is:

1. **Offline.** Caption every non-text item once, then embed all captions with an MRL text
   encoder and store the level-`L` vectors. (`data/captioning.py`, `index/store.py`)
2. **Stage-1, online.** Pyramid Rank filters `N → K` using short embedding prefixes.
   (`stage1/pyramid_rank.py`)
3. **Stage-2, online.** Re-rank those `K` with two MLLM scores computed on the *native*
   modality. (`stage2/`)

Nothing is trained at any point.

---

## 2. Stage-1: Pyramid Rank

### 2.1 The pyramid

Qwen3-MRL produces nested embeddings: level `ℓ` is literally the first `2^(ℓ-1)·d`
coordinates of the level-`L` vector.

```python
x_c_l = x_c_L[: 2 ** (l - 1) * d]      # Eq. 2
```

With `d = 32` and `L = 6`, the levels are `[32, 64, 128, 256, 512, 1024]`. Level-`L`
vectors are unit-norm. `z^(ℓ)` denotes a level-`ℓ` prefix zero-padded back to `D`.

`MRLEmbeddingStore` never materialises `z`: since `z` only appears inside inner products
and norms, `⟨z_q^(ℓ), z_c^(ℓ)⟩` is the inner product of the two prefixes and `‖z^(ℓ)‖` is a
prefix norm. Prefix norms for all `L` levels are computed once at construction
(`_prefix_sqnorms`), so each bisection step is one matrix-vector product and a lookup.

### 2.2 The upper bound

Split the level-`L` similarity into a computable part and a remainder:

```
⟨x_q^(L), x_c^(L)⟩ = ⟨z_q^(ℓ), z_c^(ℓ)⟩ + ⟨x_q^(L) − z_q^(ℓ), x_c^(L) − z_c^(ℓ)⟩
```

Cauchy–Schwarz bounds the remainder by the product of the two residual norms. And the
residual norm is not unknown: `z^(ℓ)` and `x^(L) − z^(ℓ)` occupy disjoint coordinates, so
they are orthogonal, and with `‖x^(L)‖ = 1`,

```
‖x^(L) − z^(ℓ)‖ = √(1 − ‖z^(ℓ)‖²)
```

giving a bound made entirely of quantities available from short prefixes (Eq. 6):

```
U_q,c^(ℓ) = ⟨z_q^(ℓ), z_c^(ℓ)⟩ + √((1 − ‖z_q^(ℓ)‖²)(1 − ‖z_c^(ℓ)‖²))
```

Two properties fall out, both asserted in `tests/test_upper_bound.py`:

- `U^(ℓ) ≥ ⟨x_q^(L), x_c^(L)⟩` at every level — the bound is **admissible**.
- At `ℓ = L` the residual vanishes and `U^(L)` *is* the similarity — the bound is **tight**
  at the top of the pyramid, and tightens monotonically on the way there.

### 2.3 The algorithm

```
τ_min, τ_max = −1, 1;  ℓ = 1;  I = {1..N}
while τ_max − τ_min > ε:
    τ = (τ_min + τ_max) / 2
    U = upper_bound(q, C[I], level=ℓ)
    Î = {i ∈ I : U_i ≥ τ}
    if |Î| ≥ K:  τ_min = τ;  ℓ = min(ℓ+1, L);  I = Î;  U_q[I] = U[Î]
    else:        τ_max = τ
return top-K of I by U_q
```

The loop does two things at once: it searches for a threshold that isolates roughly `K`
candidates, and it spends longer vectors only on the candidates that are still in
contention. Most of the database is eliminated at level 1 with 32-dim arithmetic.

### 2.4 Guarantees

| Property | Statement | Test |
|---|---|---|
| Admissibility | A candidate is dropped only when `U^(ℓ) < τ`; since `U` bounds the true similarity from above, nothing relevant is pruned. Everything outside the surviving set provably scores below the terminal `τ_min`. | `test_pruning_is_admissible` |
| Convergence | Each step halves the interval, so termination takes at most `⌈log₂(w₀/ε)⌉` steps — independent of `N`. | `test_converges_within_the_bisection_bound`, `test_convergence_is_independent_of_database_size` |
| ε-bounded correctness | Any discarded item `j` satisfies `⟨q, c_j⟩ ≤ ⟨q, c_K⟩ + ε`. | `test_epsilon_bound_is_exact_when_bounds_are_exact`, `test_epsilon_bounded_correctness` |

**Implementation note.** The ε-correctness argument is written in terms of `U^(L)`, but
Algorithm 1's `U_q` mixes bounds taken at different levels. `Stage1Config.final_rescore`
(default `True`) re-scores the surviving set at level `L` before the final sort, which
restores that condition at a cost of `|I|·D` MACs. Set it to `False` for the literal
pseudocode.

### 2.5 Cost

Naive full-resolution scoring costs `N·D`. Pyramid Rank costs `Σ_t (rows scored at step t)·dim(ℓ_t)`,
which `Stage1Output` tracks alongside `naive_cost`, so `output.speedup` is measured, not
assumed. If levels were uniformly distributed the saving would be a factor of `L`; in
practice the distribution is dataset-dependent — easy datasets resolve at low levels, hard
ones climb — which is what makes the level histogram a readable proxy for dataset difficulty.

Two implementation facts shape what you will actually observe:

- **Masked scans, late compaction.** Gathering the surviving rows out of the database is
  random-access memory traffic, and at low levels it costs more than the arithmetic it
  saves. So while the surviving set is still a sizeable fraction of the buffer, the whole
  contiguous prefix matrix is scanned and masked afterwards; the buffer is compacted only
  once the set has collapsed (`pyramid_rank(..., compact_fraction=0.05)`). `cost` counts
  those masked-out rows, so the reported speed-up is conservative rather than idealised.
- **Cached prefixes.** Row-major storage means reading a 32-column prefix out of a 1024-wide
  matrix still touches every cache line. `MRLEmbeddingStore` therefore keeps contiguous
  copies of the narrow levels (`cache_prefix_dim`, default 128 → ~22% extra memory), without
  which the arithmetic saving would not translate into wall-clock at all.

Expect the measured MAC saving to be roughly 2–3× on this reference NumPy/CPU
implementation, where large scans are memory-bandwidth bound. The paper's 3.3× is
end-to-end wall-clock for a batched GPU deployment at 5.7M candidates.

---

## 3. Stage-2: multimodal re-ranking

Stage-1 works on captions. Stage-2 puts the modality back: the `K` survivors are scored
with their images, video or audio attached, which is where the fine-grained gains come from.

### 3.1 Bidirectional-CoT Embedding Score (`S_CoT`)

Two prompts per (query, candidate) pair:

- **q2c** — "here is a query, here is a candidate, reason about which query concepts the
  candidate supports";
- **c2q** — the same with the roles swapped, reasoning from candidate concepts to the query.

Both carry two in-context CoT demonstrations (one aligned, one not) and end with an `<emb>`
marker. The hidden state of the token *preceding* the marker is pooled as an embedding — a
single forward pass, no decoding, following the one-word-limitation formulation. The score
is the cosine between `z_q2c` and `z_c2q` (Eq. 7).

Direction matters: alignment is asymmetric. A candidate can support every concept in a
short query while introducing a great deal the query never asked for, and only the c2q
direction sees that.

### 3.2 Question Answering Relevance Score (`S_QA`)

The MLLM converts the query into `M` discriminative yes/no question-answer pairs, once per
query. Each candidate is then the sole context for answering them, and the score is answer
accuracy (Eq. 8). `QARelevanceScorer` keeps a `QALog` per candidate — question, gold answer,
predicted answer — so `result.explain()` prints exactly which semantics a candidate matched
or missed.

### 3.3 Combination

```
S_rerank(q, c_k) = α · S_CoT(q, c_k) + (1 − α) · S_QA(q, c_k)
```

`α` is the only Stage-2 knob. `scripts/reproduce/sweep_alpha.py` computes both components
once per pair and re-mixes them, so a full sweep costs one Stage-2 pass.

---

## 4. Choice of MRL embedding model

Pyramid Rank needs only a *nested* representation hierarchy; it is otherwise
modality-agnostic. At the time of the paper, Qwen3-Embedding-0.6B was the only public
foundation embedding model exposing an MRL pyramid, and it is text-only — hence the
offline captioning step. The appendix compares it against the image-native ResNet-MRL of
Kusupati et al. on NIGHTS (`i→i`) and finds the captioned foundation encoder stronger,
despite the modality conversion.

`models/qwen3_mrl.py` contains a `ResNetMRLEmbedder` stub for anyone wanting to plug in an
image-native pyramid: implement `encode`, keep the prefixes nested and the level-`L` vectors
unit-norm, and Stage-1 needs no changes.
