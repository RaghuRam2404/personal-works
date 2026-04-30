# Week 13 Assignment — KV Cache and Sampling from Scratch

## Setup Checklist

- [ ] Your Week 12 `model_v2.py` (modernized nanoGPT) or Week 11 `model.py` as base
- [ ] A trained checkpoint from Week 11 or 12 (for inference benchmarking)
- [ ] GitHub branch `week-13-kv-cache-sampling`
- [ ] W&B project `week-13-kv-cache-sampling`
- [ ] Colab Free or Mac (inference runs fast on CPU for this model size)

---

## Task 1 — Add KV Cache to CausalSelfAttention

**Goal:** Modify your attention module to accept and return a KV cache.

**Requirements:**
- Change `CausalSelfAttention.forward` signature to: `forward(self, x, kv_cache=None) -> (output, new_kv_cache)`
  - `kv_cache`: None (first step) or `(k_past, v_past)` where each has shape `[B, n_heads, T_past, d_k]`
  - `new_kv_cache`: `(k_full, v_full)` — the concatenated cache for next step
- When `kv_cache` is not None, concatenate past K/V with current K/V along the sequence dimension
- During inference (kv_cache mode), x has shape `[B, 1, C]` (single new token) — handle this correctly
- No causal mask needed when kv_cache is provided (explain why in a code comment)
- Train mode (no kv_cache): behavior identical to Week 11/12 (causal mask applied normally)

**Unit test:** Run a forward pass of length 10 with no cache. Then run 10 individual forward passes with cache (one token at a time). Assert the output logits are identical (within 1e-4).

**Deliverable:** Updated `model_v2.py` with KV cache support.

---

## Task 2 — Update GPT.generate() to Use KV Cache

**Goal:** Rewrite `generate()` to use the per-layer KV cache.

**Requirements:**
- `generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None)`
- Initialize `kv_caches = [None] * self.config.n_layer`
- On the first call: process the full prompt as a single batch (no cache yet); collect initial KV caches from all layers
- On subsequent calls: process only the single new token; pass and update caches
- Return generated token IDs
- The `Block` and `CausalSelfAttention` must thread the cache through: `GPT` collects `[(k0,v0), (k1,v1), ...]` for each layer

**Deliverable:** Updated `generate()` in `model_v2.py`

---

## Task 3 — Implement Sampling Strategies

**Goal:** Standalone `sample_next_token(logits, temperature, top_k, top_p)` function.

**Requirements:**
- `temperature`: divide logits by this before anything else. Default 1.0.
- `top_k`: if not None, zero out all logits except the top k. Set the rest to `-inf`.
- `top_p`: if not None, sort probabilities descending, compute cumulative sum, mask out tokens beyond the cumulative threshold `p`. At least one token must always survive (handle edge case where first token already exceeds p).
- Return: a sampled token index (integer).
- Greedy = `sample_next_token(logits, temperature=1.0, top_k=1, top_p=None)`

**Unit tests:**
- `top_k=1` always returns the argmax token
- `top_p=1.0` never masks anything (all tokens survive)
- `top_p=0.0` always returns only the single highest-probability token
- Returned token is always a valid vocabulary index (0 ≤ token < vocab_size)

**Deliverable:** `sampling.py` with tests.

---

## Task 4 — Benchmark KV Cache Speedup

**Goal:** Measure actual inference speed with and without KV cache.

**Requirements:**
- Load your trained SQL model checkpoint
- Generate 200 tokens from the same prompt using:
  - Without KV cache: standard `generate()` that re-runs full forward pass each step
  - With KV cache: your new `generate()` with cached K/V
- Measure wall-clock time for each (use `time.perf_counter()` — not `timeit`, not a profiler)
- Report: tokens/second for each method and the speedup ratio
- Acceptance criteria: KV-cached inference must be at least 5x faster than non-cached for a 200-token generation from a 50-token prompt

**Hint:** The speedup depends on model size and context length. A small nanoGPT may show only 3–5x speedup (the overhead of Python loops matters at this scale). In production 7B models, the speedup is 20–100x.

**Deliverable:** `benchmark.py` with printed timing table. Example output:
```
Without KV cache: 200 tokens in 12.3s = 16.3 tok/s
With KV cache:    200 tokens in 1.8s  = 111 tok/s
Speedup: 6.8x
```
GitHub commit `week-13-kv-cache-sampling`.

---

## Task 5 — Compare Sampling Strategies on SQL

**Goal:** Generate 10 SQL queries using each strategy. Qualitatively compare.

**Requirements:**
- Use your trained SQL model
- Generate 10 queries (50 tokens each) with:
  - Greedy (top_k=1)
  - Temperature 0.3, top_k=10
  - Temperature 0.7, top_p=0.9
- Save all 30 outputs to `sampling_comparison.txt`
- Write 3–5 sentences at the top of the file: which strategy produces the most syntactically valid SQL?

**Deliverable:** `sampling_comparison.txt`

---

## Stretch Goals

- Implement beam search with beam width 3. Compare its SQL quality to top-p sampling. Is it better? Is it slower?
- Profile where time is spent during non-cached inference (use `torch.profiler`). Which operation dominates?
- Implement a KV cache eviction policy: when the cache exceeds `block_size`, drop the oldest tokens.
