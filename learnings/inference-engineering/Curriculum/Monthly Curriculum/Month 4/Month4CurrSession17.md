[← Month 4 overview](<../Month4.md>) · [← Session 16](<Month4CurrSession16.md>)

## Session 17 (optional) — Rotary position embeddings (RoPE)

> **Optional experimentation session.** It is not part of the Month 4 acceptance checklist. Do it only after Session 15's GPT passes its tests, and preferably after Session 16 has produced a learned-position baseline checkpoint to compare against. Budget 3–4 hours.

### What you should have working at the end

- `gpt/rope.py` with a precomputed cos/sin cache and an `apply_rope(x, cos, sin)` function.
- A `pos_encoding` switch in the GPT config: `"learned"` (Session 15), `"rope"`, or `"none"`.
- Tests proving RoPE's defining properties: norm preservation, the relative-position property, the position-0 identity, and equality between the interleaved and half-split layouts.
- A small training comparison of learned, RoPE, and no-position models with the same budget.
- A length-extrapolation experiment that evaluates beyond the training `block_size`.
- Notes connecting RoPE to the KV cache you will build in Month 5.

### What carries over and what is new

| From Session 13–15 | New in this session |
|---|---|
| Learned `wpe` added once to the residual stream at the input | No position vector is added to the input |
| Position is information *inside* `x` that every layer can read | Position is applied *inside every attention layer*, to Q and K only |
| Scores depend on absolute positions (through `x`) | Scores depend only on the *relative distance* `n - m` |
| Cannot run at `T > block_size` (no `wpe` row exists) | Can be computed for any position; quality beyond the training length must be measured |

### Part 1 — Why another positional scheme?

Before writing code, answer these in `notes/session-17.md`:

1. Without any position information, why does unmasked attention behave like a bag of tokens? (Permuting the input rows permutes the output rows the same way.)
2. With the causal mask, can a model still recover *some* position information? How? (Hint: how many keys can position `t` see?)
3. Learned `wpe` adds a vector to `x`. After the first layer it is mixed with content in the residual stream. Why might it help to inject position directly where it is used, in the Q·K dot product?
4. For language, which is usually more important: "this token is at position 417" or "this token is 3 positions before me"?

### Part 2 — The 2-D rotation idea

Take a 2-D query `q = (q0, q1)` at position `m` and a 2-D key `k = (k0, k1)` at position `n`. Rotate each by an angle proportional to its position:

```text
R(a) = [[cos a, -sin a],
        [sin a,  cos a]]

q' = R(m·θ) q
k' = R(n·θ) k
```

Derive by hand, using `R(a)ᵀ = R(-a)` and `R(a) R(b) = R(a + b)`:

```text
q' · k' = qᵀ R(m·θ)ᵀ R(n·θ) k = qᵀ R((n - m)·θ) k
```

The score depends only on the **offset** `n - m`, not on `m` or `n` separately. That is the whole idea of RoPE.

Also verify on paper:

- Rotation preserves length: `‖q'‖ = ‖q‖`.
- At position 0 the rotation is the identity: `R(0) = I`.

### Part 3 — From 2 dimensions to D dimensions

A head has `D` dimensions (D even). Split them into `D/2` pairs and rotate each pair at its own frequency:

```text
θ_i = base^(-2i / D),   i = 0, 1, ..., D/2 - 1,   base = 10000

angle for pair i at position m:  m · θ_i
```

- Pair 0 rotates fastest, one radian per position.
- The last pair rotates very slowly.
- Each pair is a clock hand at a different speed: the same idea as the sinusoidal encoding from the paper, now used as a rotation instead of an added vector.

Shape ledger, per attention layer:

```text
positions m:          [T]
θ:                    [D/2]
angles = outer(m, θ): [T, D/2]
cos, sin:             [T, D/2]   (precompute for block_size, slice [:T])
q, k after split:     [B, H, T, D]
q_rot, k_rot:         [B, H, T, D]
v:                    unchanged
```

RoPE is applied **after** the head split and **before** `q @ kᵀ`. It is never applied to V, because V carries content, not addressing.

### Part 4 — Two memory layouts (know both)

Both layouts are correct. They differ only in which dimensions form a pair.

**Interleaved (RoFormer paper, original Llama/Meta code):** pairs are `(0,1), (2,3), (4,5), ...`

```text
x_even = x[..., 0::2]
x_odd  = x[..., 1::2]
out_even = x_even·cos - x_odd·sin
out_odd  = x_even·sin + x_odd·cos
interleave(out_even, out_odd) back to [..., D]
```

**Half-split, also called `rotate_half` (GPT-NeoX, Hugging Face Llama):** pairs are `(0, D/2), (1, D/2+1), ...`

```text
x1 = x[..., :D/2]
x2 = x[..., D/2:]
out = cat(x1·cos - x2·sin,  x1·sin + x2·cos)
```

Question for your notes: why do these two layouts give **different** Q and K tensors but **identical** attention scores, provided the same layout is used for both Q and K? (Hint: applying the same permutation of feature dimensions to both vectors does not change their dot product.)

Practical consequence: when loading pretrained weights, using the wrong layout raises no error. It silently produces garbage. Remember this for Month 5+ when you load real checkpoints.

### Part 5 — Implement `gpt/rope.py`

Contract:

```python
def build_rope_cache(head_dim, max_pos, base=10000.0, dtype=torch.float32, device=None):
    """Return cos, sin of shape [max_pos, head_dim // 2]."""

def apply_rope(x, cos, sin):
    """x: [B, H, T, D]; cos/sin: [T, D/2] → returns [B, H, T, D]. Half-split layout."""
```

Checklist:

1. Assert `head_dim % 2 == 0`.
2. Compute `θ` and the angle table in **float32 or float64**, even if the model later runs in bf16/fp16. Large positions multiplied by small angles lose precision in low-precision floats.
3. Register `cos` and `sin` as buffers on the attention module with `persistent=False`, the same choice as the causal mask, because they can always be recomputed.
4. Broadcast `cos[:T]`, `sin[:T]` from `[T, D/2]` to `[B, H, T, D/2]`.
5. Write the interleaved variant as `apply_rope_interleaved` as well, for the Part 7 test.

Then modify your `CausalSelfAttention`:

```text
q, k, v = project and split     → [B, H, T, D]
if use_rope:
    q = apply_rope(q, cos[:T], sin[:T])
    k = apply_rope(k, cos[:T], sin[:T])
scores = q @ kᵀ / sqrt(D)       → unchanged from here on
```

And in `GPTLanguageModel`: when `pos_encoding == "rope"` or `"none"`, do not create `wpe` and do not add it.

### Part 6 — A position offset argument (preview of the KV cache)

Add an optional `start_pos=0` argument to `apply_rope` (or to the attention forward) that rotates using positions `start_pos, ..., start_pos + T - 1`, by slicing `cos[start_pos : start_pos + T]`.

Why this matters: during cached generation in Month 5, you will feed **one** new token at position `t`. It must be rotated by `t`, not by `0`. Forgetting the offset is one of the most common KV-cache bugs: the code runs and the output is subtly wrong.

### Part 7 — Tests (`tests/test_rope.py`)

1. **Shape and dtype:** output shape equals input shape, for several `B, H, T, D`.
2. **Norm preservation:** `‖apply_rope(x)‖` equals `‖x‖` per vector, to tolerance.
3. **Position 0 is identity:** the rotation at position 0 leaves the vector unchanged.
4. **Relative-position property:** this is the defining test. Pick random `q`, `k`. Compute the score between `q` at position `m` and `k` at position `n`. Shift both by the same `s` (`m + s`, `n + s`) using `start_pos`. The score must be unchanged for several `s`.
5. **Absolute positions do matter for the pair:** changing only `n` (keeping `m`) must change the score. This guards against a no-op implementation that would pass test 4.
6. **Layout equivalence:** the interleaved scores from `(q, k)` equal the half-split scores from `(P q, P k)`, where `P` permutes dimensions from the interleaved pairing to the half-split pairing.
7. **Complex-number reference:** treat each pair as a complex number `z = x_a + i·x_b` and multiply by `exp(i·m·θ)` using `torch.polar` / `torch.view_as_complex`. It must match `apply_rope`.
8. **Causality still holds:** rerun the Session 13/15 future-token tests with `pos_encoding="rope"`.
9. **Tiny-batch overfit:** the RoPE GPT still overfits one batch, as in Session 15.
10. **Offset consistency:** running the full sequence of length `T` in one call gives the same rotated `k` at position `t` as rotating a single token with `start_pos=t`.

### Part 8 — Break it deliberately

Make each mistake, record which test fails (or that **none** does), then revert:

- Apply RoPE to V as well.
- Apply RoPE to Q only.
- Use half-split for Q and interleaved for K.
- Use `θ_i = base^(-i / D)` instead of `base^(-2i / D)`.
- Forget `start_pos` (always rotate from 0).
- Apply RoPE before the head split, on `[B, T, C]` with `C` in place of `D`.

For every mistake that no test catches, write the test that would.

### Part 9 — Training comparison

Use the same config, seed, data, number of updates, and evaluation protocol as your Session 16 baseline (shorten everything if compute is tight, but keep it identical across runs):

| Run | `pos_encoding` | Parameters | Best val loss | Notes |
|---|---|---|---|---|
| A | `learned` | | | |
| B | `rope` | | | |
| C | `none` | | | |

Record:

- The parameter difference between A and B (it should equal `block_size × C`) and why.
- Train/val curves for all three on one plot.
- A 200-character sample from each best checkpoint, same prompt, temperature and seed.

Questions:

- Does C learn anything at all? How much worse is it, and what does that tell you about Part 1 question 2?
- Is B better, worse, or indistinguishable from A at this scale? At character level with a small `block_size`, a large gap is not guaranteed, so report what you measure, not what you expected.

### Part 10 — Length extrapolation experiment

Let `T_train = block_size` used during training.

1. Build the mask and the RoPE cache for `T_eval = 2 × T_train` and `4 × T_train`, without retraining.
2. Evaluate validation loss at `T_train`, `2×`, and `4×`, and report loss **per position bucket** (`[0, T_train)`, `[T_train, 2T_train)`, ...), not only the average.
3. Repeat with **position interpolation**: scale positions by `T_train / T_eval` before computing the angles, so every position falls inside the trained range.
4. Note what happens with run A (learned): why is it impossible rather than merely bad?

| Model | Loss @ T_train | Loss @ 2× | Loss @ 4× |
|---|---|---|---|
| Learned | | n/a | n/a |
| RoPE (plain) | | | |
| RoPE + position interpolation | | | |

Write two or three sentences interpreting the per-bucket losses. "RoPE can compute any position" is not the same claim as "RoPE generalizes to any position."

### Part 11 — First-principles questions

Answer without notes:

1. Why is RoPE applied to Q and K but not V?
2. Derive why the RoPE score depends only on `n - m`.
3. Why does each dimension pair need a different frequency? What goes wrong if every pair uses the same `θ`?
4. Why does RoPE add no learned parameters?
5. Why can two correct RoPE implementations disagree on the Q tensor yet agree on every attention score?
6. In a KV cache, should you store K before or after rotation? What would it cost to choose the other option?
7. Why must a newly decoded token be rotated with its true position, not position 0?
8. What do position interpolation and changing `base` (NTK-aware scaling) each change, in terms of the angle table `m · θ_i`?

Suggested answers to check yourself against afterwards: (1) V is the content being retrieved; position should affect *who attends to whom*, which is only decided by the scores. (6) After rotation. A cached rotated K is final and never needs touching again, while storing it unrotated would require re-rotating every cached key at every step.

### Part 12 — Compare with references

Only after your implementation and tests pass:

- RoFormer paper, *RoFormer: Enhanced Transformer with Rotary Position Embedding*: <https://arxiv.org/abs/2104.09864>
- EleutherAI blog, *Rotary Embeddings: A Relative Revolution*: <https://blog.eleuther.ai/rotary-embeddings/>
- Position interpolation, *Extending Context Window of Large Language Models via Positional Interpolation*: <https://arxiv.org/abs/2306.15595>
- YaRN, *Efficient Context Window Extension of Large Language Models*: <https://arxiv.org/abs/2309.00071>
- Hugging Face Llama `rotate_half` / `apply_rotary_pos_emb` in `transformers/models/llama/modeling_llama.py`

For the Hugging Face code, identify: which layout it uses, where the cos/sin cache lives, how it handles positions during cached generation (`position_ids`), and in what dtype it computes the angles.

### Deliverable

- `gpt/rope.py` with `build_rope_cache`, `apply_rope` (half-split), `apply_rope_interleaved`, and `start_pos` support.
- `CausalSelfAttention` and `GPTLanguageModel` supporting `pos_encoding ∈ {learned, rope, none}`.
- `tests/test_rope.py` with the ten tests from Part 7.
- Break-it log.
- `outputs/rope_experiments.md` with the training table, curves, samples, and extrapolation table.
- `notes/session-17.md` with the Part 1 and Part 11 answers.

### Exit criteria

- All RoPE tests pass, and the existing attention and GPT tests still pass in all three `pos_encoding` modes.
- You can derive the relative-position property on paper in under five minutes.
- You can explain why the interleaved and half-split layouts are both correct and why mixing them is a silent bug.
- You have a measured, not assumed, comparison of learned vs RoPE vs none, and an extrapolation result reported per position bucket.
- You can state exactly what `start_pos` must be during cached decoding, ready for Month 5.
