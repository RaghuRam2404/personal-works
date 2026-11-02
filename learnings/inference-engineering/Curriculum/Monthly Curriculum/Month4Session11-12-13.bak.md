
## Session 11 — Attention version 1: explicit loops

### Objective

Implement causal single-head self-attention in the most explicit form, prioritizing meaning over speed.

### Input and parameters

Let:

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

### Loop restriction

The educational implementation must use explicit Python loops over:

- Batch index.
- Query position.
- Key position.

Linear projections may be vectorized, but score construction and value aggregation must remain explicit. Implement a numerically stable softmax by subtracting the maximum permitted score before exponentiation.

### Required outputs

Return:

- Attention output `[B, T, Dv]`.
- Attention weights `[B, T, T]` for testing and inspection.

Future positions should have exactly zero probability after masking.

### Hand-worked example

Use `B=1`, `T=3`, and very small vectors. Calculate all allowed scores, probabilities, and outputs by hand. Confirm:

- Position 0 can use only position 0.
- Position 1 can use positions 0 and 1.
- Position 2 can use positions 0, 1, and 2.

### Assignment M4.11

Implement and test these invariants:

1. Every attention row sums to 1 over allowed keys.
2. Every probability above the causal diagonal is 0.
3. Changing `x[:, i+1:]` cannot change output positions `<= i`.
4. With all permitted scores equal, attention is uniform over the prefix.
5. With `T=1`, the output equals the single projected value.
6. Output remains finite for deliberately large scores because softmax is stabilized.

### First-principles questions

1. What question does a query represent computationally?
2. What does a key do differently from a value?
3. Why are query-key dot products scores rather than outputs?
4. Why does softmax operate over the key-position axis?
5. Why is the causal mask applied before softmax?
6. Why does dividing by `sqrt(D)` help keep score scale controlled?
7. What is the time and memory complexity in sequence length?

### Deliverable

- `attention/loop_attention.py`.
- Hand-worked example.
- Invariant tests.
- Full shape ledger.

### Exit criteria

- Loop attention is correct and causal.
- You can explain every loop and tensor index without saying “because the code does it.”

---

## Session 12 — Attention version 2: vectorized

### Objective

Replace explicit score and aggregation loops with batched matrix multiplication while preserving identical mathematics.

### Vectorized derivation

Given `q`, `k`, and `v`:

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(D)
scores = scores.masked_fill(~causal_mask, float("-inf"))
weights = F.softmax(scores, dim=-1)
out = weights @ v
```

Required shapes:

```text
q:                [B, T, D]
k.transpose:      [B, D, T]
scores:           [B, T, T]
causal_mask:       [T, T] or broadcastable equivalent
weights:           [B, T, T]
v:                 [B, T, Dv]
out:               [B, T, Dv]
```

### Parity harness

Use the same input and projection parameters for loop and vector implementations. Compare:

- Projected `q`, `k`, and `v`.
- Pre-mask allowed scores.
- Attention weights.
- Outputs.
- Input gradients.
- `Wq`, `Wk`, and `Wv` gradients.

Use a shared scalar test loss such as a weighted sum of outputs so that all output elements contribute non-symmetrically.

Run across at least 20 random seeds and multiple shapes, including:

- `B=1, T=1`.
- `B>1`.
- `D != Dv`.
- A repeated-token input.
- A non-power-of-two sequence length.

### Causal leakage test

For a random cut position `i`:

1. Clone the input.
2. Replace all future positions after `i` with unrelated values.
3. Run both inputs.
4. Assert outputs through position `i` are unchanged.
5. Confirm at least one later output usually changes.

### Assignment M4.12

Benchmark loop and vector implementations for small increasing sequence lengths. Measure only after warm-up and synchronize CUDA if using it. Record the trend rather than expecting a fixed speedup.

Then answer:

1. Did vectorization change asymptotic `T^2` work?
2. Why can it still be dramatically faster?
3. Which intermediate tensor causes quadratic memory growth?
4. Why does a lower-triangular mask preserve parallel training?
5. Why are boolean-mask semantics easy to reverse across APIs?

PyTorch's current scaled-dot-product-attention API treats `True` in its boolean attention mask as allowed to participate, while other attention APIs may use `True` to mean masked; always verify the API rather than assuming mask polarity.[cite:221]

### Optional reference check

After your implementations pass, compare the vectorized output with `F.scaled_dot_product_attention` using aligned mask and dropout settings. This is validation only; do not replace your core implementation.

### Deliverable

- `attention/vector_attention.py`.
- 20-seed forward and gradient parity results.
- Causal leakage test.
- Small timing table.
- `outputs/attention_parity.md`.

### Exit criteria

- Loop and vector outputs and gradients match.
- Causality is tested behaviorally, not only by inspecting the mask.
- All axes of `[B, T, T]` attention weights are explainable.

---

## Session 13 — Attention version 3: multi-head module

### Objective

Build reusable causal multi-head self-attention with explicit head splitting and recombination.

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

### Required implementation choices

- One packed `qkv` projection or three explicit projections; document the choice.
- Causal mask registered as a non-parameter buffer.
- Scaling by head dimension, not total embedding dimension.
- Softmax over the key axis.
- Attention dropout.
- Output projection and residual dropout.
- Sequence-length assertion against `block_size`.
- `reshape`/`transpose`/`contiguous` use explained by shape and memory layout.

### Equivalence test

Create a multi-head module with one head and align its parameters with the Session 12 single-head implementation. Assert output and gradient parity with dropout disabled.

### Independent-head test

Construct two heads with intentionally different projection parameters. Confirm their attention matrices differ and that concatenation preserves both head feature subspaces before the output projection.

### Required causality tests

- Future-input perturbation does not change prefix output.
- Every head has zero probability above the diagonal.
- Attention rows sum to 1 before dropout.
- Evaluation mode disables dropout and is deterministic.
- Training mode can vary under dropout while remaining causal.

### Assignment M4.13

Write `attention_from_memory.py` in a blank file without reference material. Time only the coding portion. The Month 4 target is a correct single-head or multi-head implementation in **15 minutes or less** after tests. The final Phase B target remains unaided multi-head attention in under 10 minutes.

Do not optimize the timer by omitting assertions or pretending shapes are understood. Record mistakes and repeat after at least two days.

### First-principles questions

1. Why must `C` be divisible by `H` in this implementation?
2. Why scale by `sqrt(D)` rather than `sqrt(C)`?
3. What does using multiple heads add beyond one larger head?
4. Why concatenate head outputs before an output projection?
5. Why is the causal mask a buffer rather than a parameter?
6. Why does dropout require different train and evaluation behavior?

### Deliverable

- `attention/multihead_attention.py`.
- One-head equivalence test.
- Multi-head causality and shape tests.
- First timed memory implementation and reflection.

### Exit criteria

- Multi-head attention passes all tests.
- Every reshape and transpose can be derived from `[B, T, C]`.
- You can distinguish head count, model dimension, and head dimension.
