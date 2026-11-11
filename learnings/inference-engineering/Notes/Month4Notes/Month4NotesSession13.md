[← Month 4 notes](<../Month4Notes.md>) · [← Session 12](<Month4NotesSession12.md>)

## Session 13 — Attention version 3: multi-head module

Curriculum: [Month4CurrSession13.md](../../Curriculum/Monthly%20Curriculum/Month%204/Month4CurrSession13.md). Code: [attention.py](../../Month%204/gpt/attention.py) (`multihead_loop`, `multihead_vectorized`, `CausalSelfAttention`, plus the stride experiment under `__main__`), [10 session13.py](../../Month%204/10%20session13.py) (scratch experiments for Part 5), [test_multihead_attention.py](../../Month%204/gpt/tests/test_multihead_attention.py) (Parts 8-10).

**Scope.** Done: Part 2 (loop over heads), Part 3 (heads on a tensor axis), Part 4 (stride/contiguity), Part 5a, 5c, 5d (scratch experiments), Part 7 (module), Parts 8, 9, 10 (tests). Not done yet: Part 1 motivation notes, Part 5b written note, Part 6 written paragraph, the Part 3 parity test of `multihead_vectorized` against `multihead_loop`, Part 11 (break it), Part 12 (first-principles answers), Part 13 (timed rebuild), Part 14 (Raschka comparison and cross-attention sketch). The tests were written with AI help (the file's first line says so). `pytest tests` currently gives 11 passed.

### Convention differences from the curriculum

- My weights are stored as `[d, C]` (the `nn.Linear.weight` layout) and the code does `x @ W.T`. The curriculum's functions take `[C, C]` and slice columns. So my head slice is by **rows**: `Wq[h*DQ:(h+1)*DQ, :]`. Because the layout already matches `nn.Linear`, the Part 5a transpose is not needed when copying weights into the module.
- `multihead_loop` and `multihead_vectorized` allow `dq = dk != dv` (the asserts only need each of `C, dq, dk, dv` to divide by `H`). The module uses `D = C // H` for q, k and v, as the curriculum says.
- The module has three separate `Linear(C, C)` layers (`Wq`, `Wk`, `Wv`), not a packed `Linear(C, 3C)`. They map one-to-one onto the function arguments. They are created with `bias=False` and `dtype=torch.float64`, so tests can compare with float64 reference code.

### Part 2 — Loop over heads

```python
for h in range(n_head):
    Wqh = Wq[h*DQ : DQ*(h+1), :]      # [DQ, C]
    Wkh = Wk[h*DK : DK*(h+1), :]
    Wvh = Wv[h*DV : DV*(h+1), :]
    out_h, weights_h = vector_attention(x, Wqh, Wkh, Wvh)   # [B,T,DV], [B,T,T]
    tweights.append(weights_h.unsqueeze(1))                  # [B,1,T,T]
    tout.append(out_h)
weights = torch.cat(tweights, dim=1)    # [B, H, T, T]
out     = torch.cat(tout, dim=2)        # [B, T, H*DV]
```

- `vector_attention` is reused unchanged. Each head's slice is an ordinary Session 12 weight matrix.
- The weights are joined by `unsqueeze` + `cat(dim=1)`, which gives the same result as `torch.stack(ws, dim=1)`.
- This is the reference the batched version and the module are compared with. A `debug=True` flag prints every shape (`Wqh`, `out_h`, `weights_h`) as a running shape ledger.

### Part 3 — Heads on a tensor axis (`multihead_vectorized`)

```python
Q  = x @ Wq.T                                  # [B, T, dq]
Qh = Q.view(B, T, H, Dq).transpose(1, 2)       # [B, H, T, Dq]
# same for K, V
scores = Qh @ Kh.transpose(2, 3)               # [B, H, T, T]
scaled = scores / Dk ** 0.5
masked = scaled.masked_fill(~tril(ones(T, T), bool), -inf)   # [T, T] broadcasts over B, H
weights = softmax(masked, dim=3)
pre_out = weights @ Vh                         # [B, H, T, Dv]
out = pre_out.transpose(1, 2).contiguous().view(B, T, dv)    # merge
```

- `view(B, T, H, D)` splits the last axis `c` into `(c // D, c % D)`, so `[:, :, h, :]` is exactly the block of columns (rows of `W`) that the loop version gave to head `h`. No data moves.
- `transpose(1, 2)` puts `H` next to `B`, so `@` treats `B` and `H` as batch axes and does `[T, D] @ [D, T]` per head. Without it, `[B, T, H, D]` would be matmul'd over its last two axes `(H, D)`, which mixes heads at one position instead of positions within a head.
- The scale uses `Dk` (the per-head dimension), not `C`.
- Shape ledger: `[B,T,C]` → `[B,T,dq]` → `[B,H,T,Dq]` → scores `[B,H,T,T]` → weights `[B,H,T,T]` → `[B,H,T,Dv]` → `[B,T,H,Dv]` → `[B,T,dv]`.
- Status: both versions run, and the module is tested against `multihead_loop` (Part 9). A direct `multihead_vectorized` vs `multihead_loop` parity test over seeds and shapes is **not written yet**.

### Part 4 — Stride and contiguity experiment

Run in the `__main__` block of `attention.py` with `B=2, T=3, H=2, Dv=4`:

| Step | shape | stride | contiguous |
|---|---|---|---|
| `weights @ Vh` | `[2, 2, 3, 4]` | `(24, 12, 4, 1)` | True |
| after `transpose_(1, 2)` | `[2, 3, 2, 4]` | `(24, 4, 12, 1)` | False |
| after `.contiguous()` | `[2, 3, 2, 4]` | `(24, 8, 4, 1)` | True |

- `transpose` only swaps the stride entries `12 <-> 4` (metadata change, same storage). `contiguous()` copies the data into the new layout, so the strides become the standard row-major `(24, 8, 4, 1)`.
- `.transpose_(1,2).view(B, T, dv)` without `.contiguous()` fails (noted in a comment in the code), because the merged axis `(H, Dv)` is no longer laid out with uniform stride in memory.
- I use `.transpose(1, 2).contiguous().view(B, T, C)` in both `multihead_vectorized` and the module, so the copy is explicit.
- Not done: the explicit `.view` error message record and the `reshape` comparison.

### Part 5 — Ingredients (scratch script `10 session13.py`)

**5a. `nn.Linear` weight layout.** `nn.Linear(5, 10).weight.shape` is `[10, 5]`, i.e. `[out_features, in_features]`, and `linear(a)` is `a @ weight.T`. The code in this repo already uses `[d, C]` everywhere, so a `Wq` for the functions is the same as `module.Wq.weight` with no transpose.

**5c. Dropout.** Experiment on `a = randint(0, 100, (20,))` and `p=0.5`:

- Train mode: some entries become 0 and survivors are scaled by `1 / (1 - p) = 2`, so a single `ad.sum()` is noisy.
- Eval mode: identity, `ad.sum() == a.sum()`.
- I averaged `drop(a).sum()` over `r = 10, 100, 1000, 10000` runs. The mean approaches `a.sum()`, and the noise shrinks as `r` grows. This is the "expected value is unchanged" claim, shown numerically.
- Rows of the weights stop summing to 1 after dropout (zeros plus 2x survivors). So the module keeps the **pre-dropout** weights in `self.last_weights` and the tests check those.

**5d. Mask as a buffer.** Class `M(block_size, persist)` registers `mask = tril(ones(block_size, block_size, bool))` with `persistent=persist`. I called `m.state_dict()` with `persist=False`. The module test confirms the persistence difference:

- `"mask"` appears in `named_buffers()` and not in `named_parameters()`, so an optimizer never sees it.
- With `persist=False` (my default in `CausalSelfAttention`) `"mask"` is absent from `state_dict()`. With `persist=True` it is present. Since the mask is a fixed function of `block_size`, it can be rebuilt on load and does not need saving.
- Not checked: the `.to(device)` behavior.

### Part 7 — `CausalSelfAttention`

```python
Q  = self.Wq(x);  Qh = Q.view(B, T, H, D).transpose(1, 2)    # same for K, V
scores = Qh @ Kh.transpose(2, 3) / D ** 0.5
mask = ~self.get_buffer("mask")[:T, :T]
weights = softmax(scores.masked_fill(mask, -inf), dim=3)
self.last_weights = weights                       # pre-dropout, for tests
attention_weights = self.attn_dropout(weights)
concat = (attention_weights @ Vh).transpose(1, 2).contiguous().view(B, T, C)
self.last_y = concat                              # pre-c_proj, for the head test
return self.residual_dropout(self.c_proj(concat))
```

- `assert n_embd % n_head == 0` in `__init__`, `assert T <= self.block_size` in `forward`.
- The mask is built once for `block_size` and sliced `[:T, :T]` each call. The buffer is stored as "True = allowed", so the fill uses `~mask`.
- `last_weights` and `last_y` are plain attributes, which cover the "keep pre-dropout weights / pre-projection y available" requirement (the module's `return_weights` flag is not used).
- Caveats seen while reading my code: `last_weights` holds a reference to the autograd graph, so it should be `.detach()`ed outside tests. `self.dropout` is stored as a float and the `nn.Dropout` modules are named `attn_dropout` and `residual_dropout`.

### Part 8 — One-head equivalence

`n_head=1`, `dropout=0.0`, `eval()`, `c_proj.weight = eye(C)`, with `B, T, C = 3, 5, 6`. Weights are cloned from the module into plain tensors for `vector_attention`. Asserts:

- the output matches, `last_weights[:, 0]` matches the reference weights, and `last_y` equals the output (identity projection);
- gradients match for `x`, `Wq`, `Wk` and `Wv`, using a random upstream gradient `R` (`(out * R).sum().backward()`) so the check is not an all-ones gradient.

### Part 9 — Module versus `multihead_loop`

For `n_head` in `{2, 3}` with `C=12`, `B=2, T=5`, identity `c_proj`:

- `last_weights[:, 0]` differs from `last_weights[:, h]` for the other heads, so the heads are really different.
- `last_weights` equals the `multihead_loop` weights and `last_y` equals the loop output, with the module's `Wq.weight`, `Wk.weight`, `Wv.weight` passed directly (row slicing matches the `nn.Linear` layout).
- Since `multihead_loop` runs `vector_attention` per head slice, this shows the module keeps each head's feature subspace separate before `c_proj`.

### Part 10 — Causality and dropout tests

Sizes: `B=2, T=5, H=3, D=4` (so `C=12`), `block_size=8`. Tests:

| Test | Checks |
|---|---|
| `test_prefix_unchanged_by_future_eval` | Perturb positions `>= 3`; prefix output is unchanged; future outputs do change (the test can see a difference). |
| `test_zero_above_diagonal_every_head` | `last_weights` is exactly 0 above the diagonal for all 3 heads. |
| `test_rows_sum_to_one_before_dropout` | Even in train mode with `p=0.5`, the pre-dropout rows sum to 1. |
| `test_eval_is_deterministic` | With `p=0.5` in `eval()`, two calls are identical (`rtol=0, atol=0`). |
| `test_train_varies_but_stays_causal` | Two train-mode calls differ. With the same `manual_seed` before each run, the prefix is unchanged, and `last_weights` stays causal. |
| `test_T_greater_than_block_size_raises` | `T = block_size + 1` raises `AssertionError`. |
| `test_shorter_than_block_size_works` (extra) | `T < block_size` works, which checks the `[:T, :T]` slice. |
| `test_mask_is_buffer_not_parameter` | `mask` is in `named_buffers()`, not in `named_parameters()`, not in `state_dict()` by default, and is in `state_dict()` with `persist=True`. |

The seed before each run in the train-mode causality test matters because both runs must draw the same dropout masks. Otherwise a prefix difference could come from the mask rather than from leakage.

### Still open (not done in this session yet)

- Part 1 motivation notes, Part 5b (why the perturbation test holds before `c_proj` but not after), Part 6 paragraph on packed vs separate projections.
- A parity test of `multihead_vectorized` against `multihead_loop` (seeds 0-19, four shapes), and the `n_head=1` check of `multihead_loop` against `vector_attention`.
- Part 11 break-it log, Part 12 answers, Part 13 timed rebuild (`attention_from_memory.py`), Part 14 Raschka comparison and cross-attention sketch.
