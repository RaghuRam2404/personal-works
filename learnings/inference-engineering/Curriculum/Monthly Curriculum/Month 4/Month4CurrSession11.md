[← Month 4 overview](<../Month4.md>) · [← Session 10](<Month4CurrSession10.md>) · [Session 12 →](<Month4CurrSession12.md>)

## Session 11 — Attention version 1: explicit loops

### What you should have working at the end

A single-head causal self-attention function written with explicit Python loops, a hand-worked example that its output reproduces, and a test file proving six invariants. Speed is irrelevant here; every index must carry meaning.

Plan for roughly 3 hours. Natural stopping checkpoint: after Part 5.

### Input and parameters

```text
x:  [B, T, C]
Wq: [C, D]
Wk: [C, D]
Wv: [C, Dv]
q:  [B, T, D]
k:  [B, T, D]
v:  [B, T, Dv]
```

For each batch `b`, query position `i`, and permitted key position `j <= i`:

$$
s_{b,i,j}=\frac{q_{b,i}\cdot k_{b,j}}{\sqrt{D}}
$$

Normalize scores only over permitted keys and calculate:

$$
y_{b,i}=\sum_{j=0}^{i}\operatorname{softmax}(s_{b,i,:})_jv_{b,j}
$$

The Transformer paper defines scaled dot-product attention using query-key similarities followed by a weighted sum of values; causal masking prevents an autoregressive position from using future tokens.[cite:233]

### Part 0 — Setup

1. Add `pytest` as a development dependency if it is not installed (`uv add --dev pytest`).
2. Create `attention/__init__.py`, `tests/`, `outputs/`, and `notes/session-11.md`.
3. Use `torch.float64` on CPU for every test in Sessions 11–13. Double precision allows tolerances such as `atol=1e-10`, so a real bug cannot hide behind "floating-point noise."

### Part 1 — Read Raschka's single-head walkthrough

Read **Understanding and Coding the Self-Attention Mechanism of Large Language Models From Scratch** (Sebastian Raschka, 2023): <https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html>

Read **only** these sections now: the introduction, "Embedding an Input Sentence," "Defining the Weight Matrices," "Computing the Unnormalized Attention Weights," and "Computing the Attention Scores." Also look at the single-head dimension-summary figure inside the "Cross-Attention" section. Leave "Multi-Head Attention" and the rest of "Cross-Attention" for Session 13, Part 11; reading its multi-head design first would anchor the wrong head structure right before you build the split-head module.

Run the code cells yourself in a scratch notebook. Budget about 40 minutes.

Read it for:

- How one query is scored against every key, normalized, and used to mix values.
- The worked numbers, which show each intermediate concretely and preview your Part 5 hand calculation.
- The `sqrt(d_k)` explanation: a dot product of `d_k` roughly unit-variance terms has variance about `d_k`. Part 3 turns this into a derivation and an experiment.
- `d_v != d_k`, which reappears as the `D != Dv` parity case in Session 12.

Be aware of these differences from this curriculum before you copy any habit from it:

| Article | This curriculum | Consequence |
|---|---|---|
| No causal mask; every query sees every key | Causal attention is mandatory | Treat the article as the non-causal special case |
| Unbatched `[T, d]`, one query at a time | `[B, T, C]`, all queries | You build the batched version yourself |
| Column-vector convention, `q = W_q x`, `W_q: [d_q, d]` | Row-vector convention, `q = x @ Wq`, `Wq: [C, D]` | Rewrite each article equation in `x @ W` form in your notes |
| `torch.rand` (uniform, all positive) initialization | Zero-mean random initialization | Its large raw scores (e.g. 11.1) are a demo artifact |

Done when: you can reproduce the article's `omega_2` and `attention_weights_2` printouts yourself and have rewritten its projection equations in the `x @ W` convention.

### Part 2 — Concepts before code

In `notes/session-11.md`, write short answers to first-principles questions 1–3 below before implementing anything. Use the search-engine analogy as a starting point and then go beyond it:

- **Query:** what this position is looking for.
- **Key:** what each position advertises about itself so it can be found.
- **Value:** the payload delivered once a position is selected.

Also explain why the raw input `x` is not used directly for all three roles: one vector would have to be simultaneously a good search term, a good index entry, and a good payload, and separate learned projections remove that conflict.

The analogy is intuition, not a guarantee of what any trained head learns.

### Part 3 — Derive and observe the `sqrt(D)` scaling

On paper: if the components `q_x` and `k_x` are independent with mean 0 and variance 1, then

$$
\operatorname{Var}(q\cdot k)=\sum_{x=1}^{D}\operatorname{Var}(q_xk_x)=D
$$

Then verify empirically:

1. Draw `q` and `k` with `torch.randn(10000, 64)`.
2. Compute the variance of `(q * k).sum(-1)`; expect about 64.
3. Divide the dot products by `8`; expect variance about 1.
4. Compare `softmax(torch.randn(10) * 8)` with `softmax(torch.randn(10))`. The first is close to one-hot.

Connect this to Sessions 1–2: a saturated softmax has near-zero local gradients for most entries, so unscaled scores starve `Wq` and `Wk` of learning signal. Describe the scaled regime as "not saturated," not as "linear."

Done when: your notes contain the derivation and the four measured numbers.

### Part 4 — Write the shape ledger before coding

In your notes, write every tensor's shape from the table above, plus:

```text
weights: [B, T, T]
out:     [B, T, Dv]
```

For each of the three loops, write which tensor axis its index walks and what one iteration means. For example: "`i` — which position is asking."

### Part 5 — Work a three-token example by hand

Use `B=1`, `T=3`, `C=D=Dv=2`, `Wq=Wk=Wv=I`, and

```text
x[0, 0] = [1, 0]
x[0, 1] = [0, 1]
x[0, 2] = [1, 1]
```

so `q = k = v = x`. On paper, compute every allowed score, every probability, and every output row. Only then compare against:

| Query position | Allowed keys | Scaled scores | Weights | Output |
|---:|---|---|---|---|
| 0 | 0 | `[0.7071]` | `[1]` | `[1, 0]` |
| 1 | 0, 1 | `[0, 0.7071]` | `[0.3302, 0.6698]` | `[0.3302, 0.6698]` |
| 2 | 0, 1, 2 | `[0.7071, 0.7071, 1.4142]` | `[0.2483, 0.2483, 0.5035]` | `[0.7517, 0.7517]` |

Confirm explicitly:

- Position 0 can use only position 0.
- Position 1 can use positions 0 and 1.
- Position 2 can use positions 0, 1, and 2.

Done when: your hand calculation matches the table to four decimal places.

### Part 6 — Implement `attention/loop_attention.py`

Use this interface:

```python
def loop_attention(
    x: torch.Tensor,
    Wq: torch.Tensor,
    Wk: torch.Tensor,
    Wv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (out [B, T, Dv], weights [B, T, T])."""
```

Implementation rules:

1. Linear projections may be vectorized: `q = x @ Wq`, `k = x @ Wk`, `v = x @ Wv`.
2. Use explicit Python loops over the batch index `b`, query position `i`, and key position `j in range(i + 1)`.
3. Compute each score explicitly as `(q[b, i] * k[b, j]).sum() / sqrt(D)`.
4. Softmax only over the permitted scores, stabilized by subtracting their maximum before exponentiation.
5. Aggregate values with an explicit loop over `j`: accumulate `w_j * v[b, j]`.
6. Positions above the diagonal must hold exactly 0 in the returned `weights`.
7. The function must remain differentiable, because Session 12 compares its gradients against the vectorized version. Build rows as Python lists and `torch.stack` them, or assign into preallocated tensors. Confirm that `out.sum().backward()` runs and populates `Wq.grad`.

Done when: `loop_attention` reproduces the Part 5 table.

### Part 7 — Test the invariants (Assignment M4.11)

Create `tests/test_attention_forward.py`. Use shapes with all-different sizes, such as `B=2, T=5, C=6, D=4, Dv=3`, so an axis mix-up raises an error instead of passing silently.

| # | Invariant | How to test it |
|---|---|---|
| 1 | Every attention row sums to 1 over allowed keys | `weights.sum(-1)` is close to 1 everywhere |
| 2 | Every probability above the causal diagonal is 0 | `torch.triu(weights, diagonal=1)` is exactly zero |
| 3 | Changing `x[:, i+1:]` cannot change output positions `<= i` | Clone `x`, overwrite the future with new random values, compare `out[:, :i+1]` |
| 4 | With all permitted scores equal, attention is uniform over the prefix | Set `Wq = 0` so every score is 0; row `i` must equal `1/(i+1)` on its prefix |
| 5 | With `T=1`, the output equals the single projected value | Compare `out` with `x @ Wv` |
| 6 | Output remains finite for deliberately large scores | Multiply `Wq` by `1e4`; assert `torch.isfinite(out).all()` |

Also add the Part 5 example as a fixed-value regression test.

Done when: all tests pass across several seeds.

### Part 8 — Break it deliberately

Introduce each bug separately, run the tests, record which test fails first, then revert:

1. Remove the max subtraction from the softmax. Expected: invariant 6 produces `inf` or `nan`.
2. Let `j` run to `T` instead of `i + 1`. Expected: invariants 2 and 3 fail.
3. Drop the `sqrt(D)` division. Note which tests, if any, catch it, and why scaling is a training-dynamics property that most invariants cannot see.

### First-principles questions

1. What question does a query represent computationally?
2. What does a key do differently from a value?
3. Why are query-key dot products scores rather than outputs?
4. Why does softmax operate over the key-position axis?
5. Why is the causal mask applied before softmax?
6. Why does dividing by `sqrt(D)` help keep score scale controlled?
7. What is the time and memory complexity in sequence length? State time as `O(B·T²·D)` and the stored weights as `O(B·T²)`, and explain each factor.

### Deliverable

- Raschka single-head read-through notes, including the convention rewrite.
- `attention/loop_attention.py`.
- Hand-worked example.
- `tests/test_attention_forward.py` with all six invariants.
- Full shape ledger.
- Break-it log.
- `notes/session-11.md`.

### Exit criteria

- Loop attention is correct and causal.
- You can explain every loop and tensor index without saying “because the code does it.”
- You can state what Raschka's single-head walkthrough covers and what it omits relative to causal, batched attention.
