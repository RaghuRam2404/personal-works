[← Month 4 overview](<../Month4.md>) · [← Session 12](<Month4CurrSession12.md>) · [Session 14 →](<Month4CurrSession14.md>)

## Session 13 — Attention version 3: multi-head module

### What you should have working at the end

A reusable causal multi-head self-attention `nn.Module` with explicit head splitting and merging. You reach it in small steps from Session 12's `vector_attention`: first as a loop over heads, then with heads moved onto a tensor axis, and finally wrapped in a module. Tests prove it matches Session 12 when there is one head, keeps heads independent, and stays causal with dropout. The session ends with a first timed rebuild from memory.

Plan for 4–5 hours across two sittings:

- **Sitting 1 (Parts 1–5):** go from one head to many heads, still as plain functions, and try out each new module ingredient on its own. Stopping checkpoint: after Part 5.
- **Sitting 2 (Parts 6–14):** build the module, test it, break it, and compare it with the article.
- The timed rebuild in Part 13 can happen on a later day.

### What carries over and what is new

Session 12 gave you one function: `vector_attention(x, Wq, Wk, Wv)`, with one set of weights producing one `[B, T, T]` weight matrix. This session keeps that core unchanged. The only new things are:

| New idea | Where it is introduced |
|---|---|
| Several heads, each with its own `Wq`, `Wk`, `Wv` slice | Parts 1–2 |
| A head axis `H` inside one tensor: `[B, H, T, D]` | Parts 3–4 |
| `nn.Linear` weight layout, output projection, dropout, mask buffer | Part 5 |
| Wrapping everything in an `nn.Module` | Parts 6–7 |

Session 12 allowed `D != Dv`. To keep head merging simple, this session uses `Dv = D = C / H` for every head, the convention GPT uses.

### Part 1 — Why more than one head?

Write in `notes/session-13.md` before coding:

1. For one query position `i`, a single head produces **one** probability distribution over the earlier positions. Out of all the earlier positions, there is a single answer to "where should I look?"
2. Think of a character model at position `i` that wants to look at **both** the previous character and the start of the current word. Explain why one softmax row has to compromise between the two (a softmax row sums to 1), and why two heads with separate weights can each make one of those lookups sharply.
3. Predict: if `C = 32` and we want `H = 4` heads without making the model wider, how large is each head's `D`? What shape is each head's output, and what shape do you get when you place the four outputs side by side?

You will test question 3 in Part 2.

### Part 2 — Multi-head as a loop over heads

Create `attention/multihead_functional.py` and reuse `vector_attention` without changing it.

```python
def multihead_loop(x, Wq, Wk, Wv, n_head):
    """x: [B, T, C]; Wq, Wk, Wv: [C, C]. Returns (y [B, T, C], weights [B, H, T, T])."""
```

1. Compute `D = C // n_head` and assert `C % n_head == 0`.
2. For head `h`, take the column slices `Wq[:, h*D:(h+1)*D]`, and the same slices of `Wk` and `Wv`. Each slice is `[C, D]`: an ordinary Session 12 weight matrix.
3. Call `vector_attention(x, Wq_h, Wk_h, Wv_h)` → `out_h: [B, T, D]`, `weights_h: [B, T, T]`.
4. Join the head outputs with `torch.cat(outs, dim=-1)` → `[B, T, C]`, and stack the weights with `torch.stack(ws, dim=1)` → `[B, H, T, T]`.

Write the shape ledger for this version in your notes:

```text
x            [B, T, C]
Wq_h         [C, D]          one column block of Wq
q_h          [B, T, D]
scores_h     [B, T, T]
out_h        [B, T, D]
cat over h   [B, T, H·D] = [B, T, C]
```

Quick checks, using all-different sizes such as `B=2, T=5, C=12, H=3`:

- With `n_head=1`, `multihead_loop` equals `vector_attention(x, Wq, Wk, Wv)` exactly.
- `y[..., h*D:(h+1)*D]` depends only on head `h`'s weight columns. Perturb the columns of head 0 and confirm that the slices for heads 1 and 2 do not change.

This loop version is slow but easy to read. It is the **reference** that the faster versions in Part 3 and Part 7 must match.

### Part 3 — Put the heads on a tensor axis

The Python loop over `h` has the same problem as Session 11's loop over `(b, i, j)`: each iteration launches small operations. As in Session 12, the fix is to make heads a tensor axis, so a single matmul processes every head at once.

**Step 1: the key observation.** Compute `q = x @ Wq` once → `[B, T, C]`. Column `c` of `q` belongs to head `c // D`, at position `c % D` inside that head. Show on paper that `q.view(B, T, H, D)[:, :, h, :]` is exactly `q[..., h*D:(h+1)*D]`, which is the slice Part 2 used. `view` splits the last axis into `(H, D)` without moving any data.

**Step 2: move `H` next to `B`.** `@` treats every axis except the last two as a batch axis. For each head to compute its own `[T, D] @ [D, T]`, the layout has to be `[B, H, T, D]`, so call `transpose(1, 2)`. Explain why `[B, T, H, D] @ [B, T, D, H]` would multiply the wrong axes (it would mix heads at one position rather than positions within one head).

**Step 3: write the full ledger** in `notes/session-13.md`:

```text
x               [B, T, C]
→ projection    3 × [B, T, C]    (q, k, v)
→ view          [B, T, H, D]
→ transpose     [B, H, T, D]
→ scores        [B, H, T, T]     q @ k.transpose(-2, -1) / sqrt(D)
→ mask          [T, T] broadcasts over B and H
→ weights       [B, H, T, T]     softmax over the last axis
→ @ v           [B, H, T, D]
→ transpose     [B, T, H, D]
→ merge         [B, T, C]        same layout as torch.cat in Part 2
```

**Step 4: implement `multihead_batched(x, Wq, Wk, Wv, n_head)`** in the same file. Leave out the Python loop over heads.

**Step 5: parity.** Over seeds 0–19 and at least the shapes `(B, T, C, H) ∈ {(1, 1, 4, 1), (2, 5, 12, 3), (3, 8, 16, 4), (2, 13, 12, 2)}`, assert that `multihead_batched` matches `multihead_loop` on both `y` and `weights`. Add this to `tests/test_multihead_attention.py`.

### Part 4 — Stride and contiguity experiment

Step 3 used `view` and `transpose` without explaining what they do to memory. Run this on a small tensor:

1. Print `.shape`, `.stride()`, and `.is_contiguous()` before and after `transpose(1, 2)`.
2. Confirm that `transpose` changes only the stride metadata, not the stored data.
3. Call `.view(B, T, C)` on the transposed tensor and record the error.
4. Compare `.contiguous().view(B, T, C)` with `.reshape(B, T, C)`. `reshape` copies only when it has to; `contiguous().view` makes the copy explicit.

Record the trade-off. Neither form is required, but you must be able to explain the one you use, including in your `multihead_batched` merge step.

### Part 5 — Four ingredients a real module needs

`multihead_batched` is mathematically complete. A GPT block also needs four engineering pieces. Try each one on its own in a scratch script before Part 7 combines them, and write a short note on each.

**5a. `nn.Linear` stores its weight transposed.**

```python
lin = nn.Linear(C, C, bias=False)
torch.testing.assert_close(lin(x), x @ lin.weight.T)
```

`lin.weight` has shape `[out_features, in_features]`, so a Session 12 style `Wq: [C, C]` equals `lin.weight.T`. You will need this mapping in Part 8 to copy weights between the function and the module.

**5b. Output projection `c_proj: Linear(C, C)`.** After the merge, the features of head 0 sit in columns `0..D-1`, head 1 in `D..2D-1`, and so on. No output feature combines information from more than one head yet. `c_proj` gives each output feature a learned mixture of all heads. Write down why Part 2's perturbation test still holds for `y` before `c_proj` but no longer holds after it.

**5c. Dropout.** Run:

```python
drop = nn.Dropout(p=0.5)
t = torch.ones(10)
drop.train(); print(drop(t), drop(t))
drop.eval();  print(drop(t))
```

Record three things: in training mode, entries become either `0` or `1 / (1 - p) = 2`; the expected value is unchanged; in eval mode dropout is the identity. Attention applies it in two places: on the weights (`attn_dropout`) and on the final output (`resid_dropout`). Explain why rows of the weights no longer sum to 1 after dropout, and why tests should therefore check the weights from **before** dropout.

**5d. The mask as a buffer.** Up to now you built `tril(ones(T, T))` on every call. A module builds it once, for the largest context length `block_size`, and slices `[:T, :T]` on each call. Try:

```python
class M(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)))
```

Check that `mask` appears in `named_buffers()` but not in `parameters()`, so the optimizer will never update it. Check that it follows `.to(device)`, and whether it appears in `state_dict()` with `persistent=True` compared with `persistent=False`.

### Part 6 — Choose packed or separate projections

Write one paragraph documenting your choice:

- **Packed `Linear(C, 3C)`:** one matrix multiply, the layout used by many GPT implementations. Split the result with `.split(C, dim=-1)`.
- **Three `Linear(C, C)` layers:** easier to read, and map one-to-one onto the `Wq`, `Wk`, `Wv` arguments of Parts 2–3.

Both are correct. The choice does not change the mathematics.

### Part 7 — Implement `attention/multihead_attention.py`

Module contract:

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout): ...
    def forward(self, x): ...
```

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

This is the `multihead_batched` ledger from Part 3 plus the Part 5 ingredients. PyTorch's `MultiheadAttention` also divides the total embedding dimension across heads, but your implementation must do the split, transpose, mask, merge, and output projection itself.[cite:232]

Work through this checklist in order. It is also the checklist for the timed rebuild in Part 13:

1. Assert `n_embd % n_head == 0`; compute `D = n_embd // n_head`.
2. Create the Q/K/V projection(s), the output projection `c_proj: Linear(C, C)`, `attn_dropout`, and `resid_dropout`.
3. Register the causal mask as a buffer (Part 5d), and document your `persistent=` choice.
4. In `forward`, read `B, T, C` from `x.shape` and assert `T <= block_size`.
5. Project, split, `view(B, T, H, D)`, and `transpose(1, 2)`.
6. Compute `scores = q @ k.transpose(-2, -1) / sqrt(D)`. Scale by the head dimension, not by `C`.
7. Mask with `self.mask[:T, :T]`, then softmax over the key axis (`dim=-1`).
8. Keep the pre-dropout weights available for tests, for example through a `return_weights` flag or a `self.last_weights` attribute, then apply `attn_dropout`.
9. Compute `weights @ v`, transpose back, merge to `[B, T, C]`, and keep this pre-projection `y` available for the head test.
10. Return `resid_dropout(c_proj(y))`.

### Part 8 — One-head equivalence test

In `tests/test_multihead_attention.py`:

1. Construct the module with `n_head=1` (so `D = C`), `dropout=0.0`, in `eval()` mode.
2. Remove the extra terms: zero all biases or construct the linears with `bias=False`, and set `c_proj.weight` to the identity.
3. Copy in weights from Session 12 parameters using the Part 5a mapping: `Wq = query_weight.T`.
4. Assert output parity with `vector_attention`.
5. Assert gradient parity for `x` and for each projection weight, mapping `module_weight.grad.T` to `Wq.grad`, `Wk.grad`, and `Wv.grad`.

### Part 9 — Module versus loop-over-heads test

1. Construct `n_head=2` (and also `n_head=3`) with random projection weights and the same simplifications as Part 8.
2. Assert that `weights[:, 0]` and `weights[:, 1]` differ.
3. Copy the module's weights into `Wq`, `Wk`, `Wv` and assert that the module's pre-projection `y` and pre-dropout weights match `multihead_loop` from Part 2. Since `multihead_loop` runs `vector_attention` once per head slice, this proves the module keeps every head's feature subspace separate before the output projection.

### Part 10 — Causality and dropout tests

Add to `tests/test_causality.py`, using all-different sizes such as `B=2, T=5, H=3, D=4`:

- Perturbing future inputs does not change the prefix output (eval mode).
- Every head has zero probability above the diagonal.
- Attention rows sum to 1 before dropout.
- Eval mode disables dropout and is deterministic: two calls return identical outputs.
- Training mode with `dropout=0.5` varies between calls but stays causal. Call `torch.manual_seed` before each run of the prefix-perturbation test so that both runs draw the same dropout masks.
- `T > block_size` raises an error.
- The mask buffer appears in `named_buffers()` and not in `parameters()`.

### Part 11 — Break it deliberately

Introduce each bug separately, record what catches it, then revert:

1. Scale by `sqrt(C)` instead of `sqrt(D)`. The one-head test cannot catch this, because `D = C` when `H = 1`; the Part 9 test should. Write down why.
2. Remove `.contiguous()` before `.view`. Record the error.
3. Use `view(B, H, T, D)` directly instead of `view(B, T, H, D).transpose(1, 2)`. The shapes still fit. Use Part 3 Step 1 to explain why the heads are now scrambled, and record which test catches it.

### Part 12 — Answer the first-principles questions

1. Why must `C` be divisible by `H` in this implementation?
2. Why scale by `sqrt(D)` rather than `sqrt(C)`?
3. What do multiple heads add beyond one larger head? (Start from your Part 1 answer.)
4. Why concatenate head outputs before an output projection?
5. Why is the causal mask a buffer rather than a parameter?
6. Why does dropout need different behavior in training and evaluation?

### Part 13 — Timed memory implementation (Assignment M4.13)

1. Open a blank `attention/attention_from_memory.py` with no reference material visible.
2. Time only the coding portion.
3. Point the existing test suite at it; if the tests are parametrized over implementations, this is a one-line change.
4. Record the time, every mistake, and the date in `notes/session-13.md`.
5. Repeat after at least two days.

The Month 4 target is a correct single-head or multi-head implementation in **15 minutes or less**, after tests. The final Phase B target remains unaided multi-head attention in under 10 minutes.

Do not improve the time by leaving out assertions or pretending to understand shapes.

### Part 14 — Compare with Raschka's multi-head and cross-attention

Return to the Raschka post from Session 11 and read the remaining sections: "Multi-Head Attention," "Cross-Attention," and the conclusion. Budget about 30 minutes. Read them as a comparison with the module you just built, not as instructions.

Note these points while reading:

- Its multi-head design uses `h` independent full-size heads built with `repeat` + `bmm`, rather than splitting `C` into `H × D`, and it has no output projection. This is closer to your Part 2 loop with full-width heads than to your Part 7 module.
- Its comment says it focuses on "the 3rd element, index position 2," but the code uses `x_2 = embedded_sentence[1]`, which is index 1.

In `notes/session-13.md`, write a short comparison:

1. Parameter count of the article's design (`h=3`, `d_q=d_k=24`, `d_v=28`, `d=16`) versus your module at the same `C` and `H`.
2. Output shape per position in each design, and why yours returns `[B, T, C]` while the article's returns `h × d_v`.
3. Why your design requires `C % H == 0` and an output projection, while the article's does not.
4. Whether three heads of size 24 are equivalent to one head of size 72. Use your answer to sharpen first-principles question 3.

Then sketch, without implementing, how your module would change for cross-attention:

- Queries come from `x1: [B, T1, C]`; keys and values come from `x2: [B, T2, C]`.
- Scores and weights become `[B, H, T1, T2]`.
- The causal mask is usually removed, because there is no autoregressive order between the two sequences.

### Deliverable

- `attention/multihead_functional.py` with `multihead_loop` and `multihead_batched`, and their parity test.
- Part 1 motivation notes and Part 5 ingredient notes.
- Stride/contiguity experiment notes.
- `attention/multihead_attention.py` with the documented projection choice.
- One-head equivalence test.
- Module versus loop-over-heads test.
- Multi-head causality, dropout, and shape tests.
- Break-it log.
- First timed memory implementation and reflection.
- Raschka multi-head comparison and cross-attention sketch.
- `notes/session-13.md`.

### Exit criteria

- Multi-head attention passes all tests, including parity with the loop-over-heads reference.
- Every reshape and transpose can be derived from `[B, T, C]`, including why `view(B, T, H, D)` matches column slicing.
- You can distinguish head count, model dimension, and head dimension.
- You can explain what each of `nn.Linear`'s weight layout, `c_proj`, dropout, and the mask buffer adds over Session 12's function.
- You can explain how the article's multi-head design differs from yours and what changes for cross-attention.
