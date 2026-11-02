[← Month 4 overview](<../Month4.md>) · [← Session 12](<Month4CurrSession12.md>) · [Session 14 →](<Month4CurrSession14.md>)

## Session 13 — Attention version 3: multi-head module

### What you should have working at the end

A reusable causal multi-head self-attention `nn.Module` with explicit head splitting and merging, proven equivalent to Session 12 for one head, proven to keep heads independent, and covered by causality and dropout tests, plus a first timed from-memory rebuild.

Plan for 3–4 hours. Natural stopping checkpoint: after Part 6; the timed rebuild in Part 10 can happen on a later day.

### Module contract

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout): ...
    def forward(self, x): ...
```

Require:

```text
x:   [B, T, C]
C % H == 0
D = C / H
q,k,v after projection/split: [B, H, T, D]
scores:                       [B, H, T, T]
weights:                      [B, H, T, T]
head outputs:                 [B, H, T, D]
merged output:                [B, T, C]
projected output:             [B, T, C]
```

PyTorch's `MultiheadAttention` similarly divides the total embedding dimension across heads, but the required implementation here must perform the split, transpose, mask, merge, and output projection directly.[cite:232]

### Part 1 — Derive the ledger from `[B, T, C]`

Write the full path in `notes/session-13.md` before coding:

```text
x               [B, T, C]
→ projection    [B, T, 3C]   (or three [B, T, C] projections)
→ split         3 × [B, T, C]
→ view          [B, T, H, D]
→ transpose     [B, H, T, D]
→ scores        [B, H, T, T]
→ weights       [B, H, T, T]
→ @ v           [B, H, T, D]
→ transpose     [B, T, H, D]
→ merge         [B, T, C]
→ c_proj        [B, T, C]
```

Explain why matrix multiplication needs `T` and `D` as the last two axes. `@` treats all leading axes as batch axes, so `[B, T, H, D]` would multiply the wrong dimensions.

### Part 2 — Stride and contiguity experiment

On a small tensor:

1. Print `.shape`, `.stride()`, and `.is_contiguous()` before and after `transpose(1, 2)`.
2. Confirm `transpose` changes only the stride metadata, not the stored data.
3. Call `.view(B, T, C)` on the transposed tensor and record the error.
4. Compare `.contiguous().view(B, T, C)` with `.reshape(B, T, C)`. `reshape` copies only when it must; `contiguous().view` makes the copy explicit.

Record the trade-off. Neither form is mandatory; you must be able to explain whichever one you use.

### Part 3 — Choose packed or separate projections

Write one paragraph documenting your choice:

- **Packed `Linear(C, 3C)`:** one matrix multiply, the layout used by many GPT implementations.
- **Three `Linear(C, C)` layers:** easier to read and easier to align with Session 12 weights.

Both are correct. The choice does not change the mathematics.

### Part 4 — Implement `attention/multihead_attention.py`

Follow this checklist in order; it is also the checklist for the timed rebuild in Part 10:

1. Assert `n_embd % n_head == 0`; compute `D = n_embd // n_head`.
2. Create the Q/K/V projection(s), the output projection `c_proj: Linear(C, C)`, `attn_dropout`, and `resid_dropout`.
3. Register the causal mask as a non-parameter buffer: `tril(ones(block_size, block_size, dtype=bool))`. Decide and document whether it should be saved in checkpoints (`persistent=`).
4. In `forward`, read `B, T, C` from `x.shape` and assert `T <= block_size`.
5. Project, split, `view(B, T, H, D)`, and `transpose(1, 2)`.
6. Compute `scores = q @ k.transpose(-2, -1) / sqrt(D)`, scaling by the head dimension, not by `C`.
7. Mask with `self.mask[:T, :T]`, then softmax over the key axis (`dim=-1`).
8. Keep the pre-dropout weights available for tests, for example through a `return_weights` flag or a `self.last_weights` attribute, then apply `attn_dropout`.
9. Compute `weights @ v`, transpose back, merge to `[B, T, C]`, and keep this pre-projection `y` available for the head test.
10. Return `resid_dropout(c_proj(y))`.

### Part 5 — One-head equivalence test

Create `tests/test_multihead_attention.py`:

1. Construct the module with `n_head=1` (so `D = C`), `dropout=0.0`, in `eval()` mode.
2. Remove the extra terms: zero all biases or construct linears with `bias=False`, and set `c_proj.weight` to the identity.
3. Copy weights from Session 12 parameters. `nn.Linear` stores `weight` as `[out_features, in_features]`, so `Wq` corresponds to the query weight transposed.
4. Assert output parity with `vector_attention`.
5. Assert gradient parity for `x` and for each projection weight, mapping `module_weight.grad.T` to `Wq.grad`, `Wk.grad`, and `Wv.grad`.

### Part 6 — Independent-head test

1. Construct `n_head=2` with intentionally different projection parameters for each head.
2. Assert `weights[:, 0]` and `weights[:, 1]` differ.
3. Assert `y[..., :D]` equals `vector_attention` run on head 0's slices of the Q/K/V weights, and `y[..., D:]` equals the same for head 1. This proves concatenation preserves both head feature subspaces before the output projection.

### Part 7 — Causality and dropout tests

Add to `tests/test_causality.py`, using all-different sizes such as `B=2, T=5, H=3, D=4`:

- Future-input perturbation does not change prefix output (eval mode).
- Every head has zero probability above the diagonal.
- Attention rows sum to 1 before dropout.
- Evaluation mode disables dropout and is deterministic: two calls return identical outputs.
- Training mode with `dropout=0.5` varies between calls, yet remains causal. Call `torch.manual_seed` before each run of the prefix-perturbation test so both runs draw identical dropout masks.
- `T > block_size` raises an error.
- The mask buffer appears in `named_buffers()` and not in `parameters()`.

### Part 8 — Break it deliberately

Introduce each bug separately, record what catches it, then revert:

1. Scale by `sqrt(C)` instead of `sqrt(D)`. The one-head test cannot catch this, because `D = C` when `H = 1`; the Part 6 head-slice test should. Write down why.
2. Remove `.contiguous()` before `.view`. Record the error.
3. Use `view(B, H, T, D)` directly instead of `view(B, T, H, D).transpose(1, 2)`. Shapes still fit; record which test detects the scrambled heads.

### Part 9 — Answer the first-principles questions

1. Why must `C` be divisible by `H` in this implementation?
2. Why scale by `sqrt(D)` rather than `sqrt(C)`?
3. What does using multiple heads add beyond one larger head?
4. Why concatenate head outputs before an output projection?
5. Why is the causal mask a buffer rather than a parameter?
6. Why does dropout require different train and evaluation behavior?

### Part 10 — Timed memory implementation (Assignment M4.13)

1. Open a blank `attention/attention_from_memory.py` with no reference material visible.
2. Time only the coding portion.
3. Point the existing test suite at it; parametrizing the tests over implementations makes this a one-line change.
4. Record the time, every mistake, and the date in `notes/session-13.md`.
5. Repeat after at least two days.

The Month 4 target is a correct single-head or multi-head implementation in **15 minutes or less** after tests. The final Phase B target remains unaided multi-head attention in under 10 minutes.

Do not optimize the timer by omitting assertions or pretending shapes are understood.

### Part 11 — Compare with Raschka's multi-head and cross-attention

Return to the Raschka post from Session 11 and read the remaining sections: "Multi-Head Attention," "Cross-Attention," and the conclusion. Budget about 30 minutes. Read them as a comparison against the module you just built, not as instructions.

Note these points while reading:

- Its multi-head design uses `h` independent full-size heads built with `repeat` + `bmm`, rather than splitting `C` into `H × D`, and it has no output projection.
- Its comment says it focuses on "the 3rd element, index position 2," but the code reuses `x_2 = embedded_sentence[1]`. The code uses index 1.

In `notes/session-13.md`, write a short comparison:

1. Parameter count of the article's design (`h=3`, `d_q=d_k=24`, `d_v=28`, `d=16`) versus your module at the same `C` and `H`.
2. Output shape per position in each design, and why yours returns `[B, T, C]` while the article's returns `h × d_v`.
3. Why your design requires `C % H == 0` and an output projection, while the article's does not.
4. Whether three heads of size 24 are equivalent to one head of size 72. Use your answer to sharpen first-principles question 3.

Then sketch, without implementing, how your module would change for cross-attention:

- Queries come from `x1: [B, T1, C]`; keys and values come from `x2: [B, T2, C]`.
- Scores and weights become `[B, H, T1, T2]`.
- The causal mask is usually removed, because no autoregressive order exists between the two sequences.

### Deliverable

- `attention/multihead_attention.py` with the documented projection choice.
- Stride/contiguity experiment notes.
- One-head equivalence test.
- Independent-head test.
- Multi-head causality, dropout, and shape tests.
- Break-it log.
- First timed memory implementation and reflection.
- Raschka multi-head comparison and cross-attention sketch.
- `notes/session-13.md`.

### Exit criteria

- Multi-head attention passes all tests.
- Every reshape and transpose can be derived from `[B, T, C]`.
- You can distinguish head count, model dimension, and head dimension.
- You can explain how the article's multi-head design differs from yours and what changes for cross-attention.
