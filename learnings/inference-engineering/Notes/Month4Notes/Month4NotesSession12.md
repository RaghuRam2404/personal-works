[← Month 4 notes](<../Month4Notes.md>) · [← Session 11](<Month4NotesSession11.md>)

## Session 12 — Attention version 2: vectorized

Curriculum: [Month4CurrSession12.md](../../Curriculum/Monthly%20Curriculum/Month%204/Month4CurrSession12.md). Code: [attention.py](../../Month%204/gpt/attention.py) (`loop_attention`, `vector_attention`), [9 session12.py](../../Month%204/9%20session12.py) (benchmark), [9 session12_paritytest.py](../../Month%204/9%20session12_paritytest.py) (parity, leakage, teeth tests).

**Scope.** Done: Parts 1-3, 5, 6, 7 (a rough version) and 9. Skipped, on purpose: Part 4 (gradient parity, `test_attention_gradients.py`), Part 8 (the `F.scaled_dot_product_attention` reference check), and the proper benchmark protocol (warm-up, median of 10, device sync). Also not done: a loop-version timing column, so the timing table below covers `vector_attention` only.

### Part 1 — Geometry

```text
x:                [B, T, C]
Wq, Wk:           [D, C]        Wv: [Dv, C]      (code: x @ W.T)
q, k:             [B, T, D]
k.transpose:      [B, D, T]
scores:           [B, T, T]
causal_mask:      [T, T]        broadcasts over B
weights:          [B, T, T]
v:                [B, T, Dv]
out:              [B, T, Dv]
```

- `[B, T, D] @ [B, D, T]` is a batched matmul. For each `b` it contracts the shared inner axis `D`, so `D` is summed out and `[T, T]` is left.
- `scores[b, i, j] = q[b, i] · k[b, j] / sqrt(D)`. `b` is an independent sequence, **row `i` is the asking position**, **column `j` is the candidate position**. Softmax runs over `j` (`dim=-1`) because each asker needs a distribution over candidates, so every row sums to 1.
- `weights @ v` contracts `j`: `out[b, i] = Σ_j weights[b, i, j] · v[b, j]`.

### Part 2 — Implementation notes

```python
Q = x @ Wq.T;  K = x @ Wk.T;  V = x @ Wv.T
scores = Q @ K.transpose(1, 2) / dk ** 0.5
masked = scores.masked_fill(~causal_mask, float("-inf"))   # causal_mask = tril(ones) == 1.0, True = allowed
weights = torch.softmax(masked, dim=2)
out = weights @ V
```

- `exp(-inf) = 0`, so a masked entry adds exactly 0 to the softmax numerator and denominator. It is exactly zero, not "small".
- A causal row can never be fully masked. The diagonal `j == i` is always allowed, so every row has at least one finite score. A fully `-inf` row would give `exp(-inf) / 0 = 0/0 = nan`.
- `return_intermediates=True` returns `(Q, K, V, scores, weights, out)` in both versions. `scores` is the *pre-mask* score in the vector version, and the loop version leaves everything above the diagonal as `nan` (it only ever fills `j <= i`). So the comparison uses the lower triangle only.
- Loop version detail: the softmax is computed per row on `scores[b, i, :i+1]` with the max subtracted first (stable softmax). The "mask" is never materialized, it is just the slice `:i+1`.
- The loop version also carries `assert torch.allclose(...)` against the vectorized matmul after each stage. These are handy while building, but they make the loop version depend on the thing it is meant to check. Worth removing once parity tests exist.

### Part 3 — Forward parity

`float64`, `atol=1e-10`, seeds 0-19, five cases `(B, T, C, D, Dv)`: minimal `(1,1,4,4,4)`, batched `(3,8,8,8,8)`, `D != Dv` `(2,7,6,4,5)`, longer odd `(2,13,8,4,4)`, repeated token `(2,6,8,4,4)`. Compared: `q`, `k`, `v`, lower-triangle scores, weights, outputs. Extras: loop scores are `nan` strictly above the diagonal, the shapes and dtype are right, the default return `(out, weights)` equals the intermediates version, and the repeated-token case gives uniform weights `1/(i+1)` over each prefix.

Why the repeated-token case matters: all keys are identical, so all raw scores in a row are equal and the only thing that shapes the row is the mask. It checks mask + softmax with no learned structure to hide behind.

### Part 5 — Behavioral leakage test

`test_future_does_not_change_past`, parametrized over both implementations and 20 seeds. For a random cut `i` (at least one future position), clone `x`, overwrite positions `> i` with unrelated values (`randn * 3 + 5`), rerun, and assert `out[:, :i+1]` is unchanged. Then assert later outputs *did* change, which shows the test is able to see a difference. This checks causality by behavior, not by reading the mask.

### Part 6 — Tests have teeth

Done with a `buggy_attention(..., bug=...)` helper. First, with `bug=None` it must match `vector_attention`, so a failure can only come from the injected bug.

| Bug | What catches it | Why |
|---|---|---|
| `masked_fill(causal_mask, -inf)` (polarity reversed) | Leakage test | Allowed entries are masked and future ones kept, so the past depends on the future. |
| `softmax(dim=-2)` | Row-sum test (`weights.sum(-1) != 1`) | It normalizes columns instead of rows. |
| `k.reshape(B, D, T)` instead of `k.transpose(-1, -2)` | Depends on shapes, see below | |

**The reshape bug.** `[B, T, D]` and `[B, D, T]` hold the same number of elements, so `reshape` never raises a shape error. But it re-chops memory instead of swapping axes.

- When `T == D` the target shape is the *same* shape, so the reshape does nothing and the code computes `q @ k`, which is `[T, D] @ [T, D]`. This is a shape-valid but wrong matmul. The result is still masked and each row still sums to 1, so the **row-sum test does not catch it**. Parity against `vector_attention` catches it, and so does the leakage test (`q @ k` mixes positions, so future keys leak in).
- When `T != D` both parity and leakage catch it.
- Lesson: use distinct sizes for `T`, `D`, `Dv`, `C`, `B` in tests (the `(2, 7, 6, 4, 5)` case does this). If two sizes coincide, a bug that confuses those two axes can pass silently. Invariants like row sums are blind to some bugs, which is why parity against an independent implementation matters.

### Part 7 — Benchmark (rough, `vector_attention` only)

Setup: `B=4, C=32, D=Dv=16`, float32, CPU, 4 rounds with only the last printed, so one timed call per `T` and the earlier rounds act as warm-up. This is **not** the median-of-10 protocol from the curriculum, so treat the numbers as noisy.

| T | time (ms) | weights `[B,T,T]` |
|---|---|---|
| 8 | 0.133 | 1.00 KB |
| 16 | 0.123 | 4.00 KB |
| 32 | 0.115 | 16.00 KB |
| 64 | 0.146 | 64.00 KB |
| 128 | 0.389 | 256.00 KB |
| 256 | 0.645 | 1024.00 KB |

- Memory is exactly `B·T²·4` bytes and grows 4x per doubling of `T` (check: `4·256²·4 = 1 MiB`).
- Time is flat from `T=8` to `64`: this is fixed per-call overhead (Python dispatch, kernel launches), not arithmetic. Only at `T >= 128` does the work start to show. From 128 to 256 the time grew 1.66x (not 4x), and the log-log slope between 64 and 256 is about 1.07. The quadratic regime has not started yet at these sizes, so the slope is far below 2. That is the point of the curriculum's "record the trend, not a fixed speedup".
- Missing, to do properly: time the loop version at the same `T`, use `time.perf_counter` with 3 warm-ups and the median of >= 10 calls, and wrap in `torch.no_grad()`.

### Part 9 — Answers

**1. Did vectorization change the asymptotic `T²` work?**
No. Both versions compute the same `B·T²` query-key dot products (each of length `D`), the same `T²` softmax entries, and the same `T²·Dv` weighted sum. The loop version even does about half the score arithmetic (it only visits `j <= i`), while the vectorized version computes the full `[T, T]` matrix and then masks it. So the work is `O(B·T²·D)` in both. Vectorization changes the constant and where the time is spent, not the complexity class.

**2. Why can it still be dramatically faster?**
Because the loop version is dominated by overhead, not arithmetic. Each `(b, i, j)` step is a Python-interpreter iteration that launches a tiny tensor operation (index, multiply, sum, divide), each paying dispatch cost that dwarfs the 16 multiplies it does. The loop also does the projections as four nested Python loops over scalars. The vectorized version hands PyTorch a few big batched operations, where one `@` runs an optimized BLAS kernel (SIMD, cache blocking, multiple threads) over all `B·T²` dot products at once. The arithmetic is the same, but it gets done at hardware speed instead of one interpreter step at a time. The benchmark shows this from the other side: even at `T=64` the vector version spends about 0.13 ms, i.e. almost all fixed overhead.

**3. Which intermediate tensor causes quadratic memory growth?**
The `[B, T, T]` score / weight tensor. `q`, `k`, `v`, and `out` are `[B, T, D or Dv]` and grow linearly in `T`. The scores, the masked scores, and the weights are each `[B, T, T]`, and eager PyTorch keeps several of them alive (and the weights are needed for backward). Measured: 1 KB at `T=8` up to 1 MiB at `T=256` for `B=4` in float32. Per head and per layer in a real model this multiplies, which is why long context is a memory problem.

**4. Why does a lower-triangular mask preserve parallel training?**
Because causality is enforced by *which entries of the score matrix are kept*, not by *the order in which positions are computed*. The mask makes row `i` depend only on positions `<= i`, so every row is a valid "predict the next token given the prefix" computation, and all `T` of them are independent of each other. So one forward pass computes all `T` training examples at once, with next-token targets for every position, and one backward pass trains on them all. Without the mask, position `i` would see the answer (token `i+1`) and the training would be trivial and useless. The alternative, generating step by step, would need `T` sequential passes. (Inference is sequential, which is exactly why the KV cache exists, but training is not.)

**5. Why are boolean-mask semantics easy to reverse across APIs?**
There is no universal convention for what `True` means. Some APIs use `True` = "allowed to participate" (`F.scaled_dot_product_attention` boolean `attn_mask`), while others use `True` = "masked out" (e.g. `masked_fill`, whose mask marks the entries to *overwrite*, and `nn.MultiheadAttention`'s boolean masks). My own code shows it: I keep `causal_mask` as "True = allowed" and so have to write `masked_fill(~causal_mask, -inf)`. Drop one `~` and the mask is exactly reversed (Part 6, bug 1). The reversed version still has the right shapes and every row still sums to 1 (the softmax normalizes whatever is left), so it fails silently. That is why the behavioral leakage test is necessary and why the polarity should be checked against the API docs rather than assumed.

### Takeaways

- The vectorized version is the same math as the loops. Parity tests with distinct axis sizes, a non-trivial case (repeated token) and an independent behavioral test (leakage) are what make "same math" a checked claim.
- Shape-valid bugs (`reshape` vs `transpose`, wrong softmax axis, reversed mask) are the dangerous ones. Each needs at least one test whose invariant it breaks.
- Speed at small `T` is about overhead. Memory at large `T` is about the `[B, T, T]` tensor. The second is what FlashAttention (Phase F) attacks by never materializing it in HBM.
