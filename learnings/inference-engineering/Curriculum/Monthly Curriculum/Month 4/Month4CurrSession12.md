[← Month 4 overview](<../Month4.md>) · [← Session 11](<Month4CurrSession11.md>) · [Session 13 →](<Month4CurrSession13.md>)

## Session 12 — Attention version 2: vectorized

### What you should have working at the end

A vectorized causal attention function that is mathematically identical to Session 11's loops, proven by forward and gradient parity across 20 seeds, a behavioral leakage test, and a timing table with written analysis.

Plan for roughly 3 hours. Natural stopping checkpoint: after Part 4.

### Part 1 — Predict the geometry

Before coding, write in `notes/session-12.md`:

```text
q:                [B, T, D]
k.transpose:      [B, D, T]
scores:           [B, T, T]
causal_mask:      [T, T] or broadcastable equivalent
weights:          [B, T, T]
v:                [B, T, Dv]
out:              [B, T, Dv]
```

Explain why `[B, T, D] @ [B, D, T]` sums out `D`, and what `scores[b, i, j]` means. Row `i` is the asking position, column `j` is the candidate position, and `b` is an independent sequence.

### Part 2 — Implement `attention/vector_attention.py`

Use the same interface as `loop_attention`. The core is:

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(D)
scores = scores.masked_fill(~causal_mask, float("-inf"))
weights = F.softmax(scores, dim=-1)
out = weights @ v
```

Requirements:

1. Build the mask as `torch.tril(torch.ones(T, T, dtype=torch.bool))`, where `True` means allowed.
2. Explain why `exp(-inf) = 0` makes masked entries exactly zero, and why a causal row can never be fully masked: the diagonal is always allowed, so no `nan` rows occur.
3. Add a `return_intermediates=True` option to **both** `loop_attention` and `vector_attention` that also returns `q`, `k`, `v`, and the pre-mask scores. In the loop version, leave entries above the diagonal as `nan` and compare only the lower triangle.

### Part 3 — Forward parity harness

Extend `tests/test_attention_forward.py`. Use identical `x`, `Wq`, `Wk`, and `Wv` for both implementations and compare:

- Projected `q`, `k`, and `v`.
- Pre-mask allowed scores (lower triangle only).
- Attention weights.
- Outputs.

Run seeds 0–19 over at least these cases:

| Case | `(B, T, C, D, Dv)` | Purpose |
|---|---|---|
| Minimal | `(1, 1, 4, 4, 4)` | `B=1, T=1` |
| Batched | `(3, 8, 8, 8, 8)` | `B>1` |
| Mismatched value width | `(2, 7, 6, 4, 5)` | `D != Dv` and non-power-of-two `T` |
| Longer odd length | `(2, 13, 8, 4, 4)` | Non-power-of-two `T` |
| Repeated token | `(2, 6, 8, 4, 4)` with every row of `x` identical | All keys equal, so weights must be uniform over each prefix |

### Part 4 — Gradient parity

Create `tests/test_attention_gradients.py`.

1. Create separate leaf copies of `x`, `Wq`, `Wk`, and `Wv` with `requires_grad=True` for each implementation.
2. Use a fixed random tensor `G` with the shape of `out` and the loss `(out * G).sum()`. Weighting the outputs unequally makes every output element contribute non-symmetrically, so a bug that permutes or averages outputs cannot cancel out.
3. Compare `.grad` for `x`, `Wq`, `Wk`, and `Wv` across the same 20 seeds and shape cases.

Done when: forward and gradient parity pass for every seed and case.

### Part 5 — Behavioral causal leakage test

Create `tests/test_causality.py`, parametrized over both implementations. For a random cut position `i`:

1. Clone the input.
2. Replace all future positions after `i` with unrelated values.
3. Run both inputs.
4. Assert outputs through position `i` are unchanged.
5. Confirm at least one later output usually changes, which proves the test is capable of observing a difference.

### Part 6 — Prove the tests have teeth

Introduce each bug separately and confirm a test fails, then revert:

1. `masked_fill(causal_mask, ...)`, a reversed mask polarity. The leakage test must fail.
2. `F.softmax(scores, dim=-2)`. The row-sum test must fail.
3. `k.transpose(-1, -2)` replaced by `k` with a reshape that happens to fit. Record whether any test catches it when `T == D`, and why distinct dimension sizes matter.

### Part 7 — Benchmark (Assignment M4.12)

Create `attention/benchmark.py`:

- Use `T ∈ {8, 16, 32, 64, 128, 256}`, `B=4`, `C=32`, `D=Dv=16`, under `torch.no_grad()`.
- Do 3 warm-up calls, then report the median of at least 10 timed calls using `time.perf_counter`.
- On CPU no synchronization is needed. On CUDA call `torch.cuda.synchronize()`, and on Apple MPS call `torch.mps.synchronize()`, before reading the timer.
- Also record the memory of the `[B, T, T]` weights tensor (`B·T²·bytes_per_element`) for each `T`.

Record the trend rather than expecting a fixed speedup. Optionally plot time against `T` on log-log axes and estimate the slope for each implementation.

### Part 8 — Optional reference check

In `attention/reference.py`, compare `vector_attention` against `F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)`. This is validation only; do not replace your core implementation.

PyTorch's current scaled-dot-product-attention API treats `True` in its boolean attention mask as allowed to participate, while other attention APIs may use `True` to mean masked; always verify the API rather than assuming mask polarity.[cite:221]

### Part 9 — Write `outputs/attention_parity.md`

Include the parity results (seeds, cases, maximum absolute differences), the leakage test result, the timing table, and answers to:

1. Did vectorization change asymptotic `T^2` work?
2. Why can it still be dramatically faster?
3. Which intermediate tensor causes quadratic memory growth?
4. Why does a lower-triangular mask preserve parallel training?
5. Why are boolean-mask semantics easy to reverse across APIs?

Address these points in your answers:

- Both implementations perform `O(T²)` work. The loop version is slow because each `(b, i, j)` step pays Python-interpreter and tiny-operation overhead, not because it does asymptotically more arithmetic.
- The `[B, T, T]` score/weight tensor is what grows quadratically in memory.
- In eager PyTorch, masking, softmax, and the value matmul run as separate operations, so the full `[B, T, T]` tensor is written to and read back from GPU main memory (HBM) between steps. That traffic, not arithmetic, becomes the bottleneck at long sequence lengths. FlashAttention (Phase F) avoids materializing the full matrix by processing tiles in on-chip SRAM.

### Deliverable

- `attention/vector_attention.py`.
- `return_intermediates` support in both implementations.
- 20-seed forward and gradient parity results.
- Causal leakage test.
- Small timing table.
- `outputs/attention_parity.md`.
- `notes/session-12.md`.

### Exit criteria

- Loop and vector outputs and gradients match.
- Causality is tested behaviorally, not only by inspecting the mask.
- All axes of `[B, T, T]` attention weights are explainable.
