[← Month 4 overview](<../Month4.md>) · [← Session 4](<Month4CurrSession4.md>) · [Session 6 →](<Month4CurrSession6.md>)

## Session 5 — Becoming a Backprop Ninja

### Objective

Move from scalar micrograd intuition to vectorized tensor calculus over a full MLP with BatchNorm.

### Material

Watch **Building makemore Part 4: Becoming a Backprop Ninja**.

- Video: <https://youtu.be/q8SA3rM6ckI>
- Official notebook: <https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb>

The lecture manually backpropagates through cross-entropy, the second linear layer, tanh, BatchNorm, the first linear layer, and the embedding table without calling `loss.backward()`.[cite:132]

### Watching rules

- Do not type the gradient expressions while watching.
- Pause before each derivative and identify the output gradient, local Jacobian action, and desired input shape.
- Write reduction and broadcasting axes explicitly.
- Treat shape restoration as part of differentiation.

### Build the dependency map

Expand the forward pass into elementary operations:

```text
Xb
→ emb
→ embcat
→ hprebn
→ bnmeani
→ bndiff
→ bndiff2
→ bnvar
→ bnvar_inv
→ bnraw
→ hpreact
→ h
→ logits
→ logit_maxes
→ norm_logits
→ counts
→ counts_sum
→ counts_sum_inv
→ probs
→ logprobs
→ loss
```

Your exact names may differ, but every elementary operation needed for the expanded backward pass must be visible.

### Gradient ledger

Create a table with columns:

| Node | Forward shape | Parents | Incoming gradient shape | Reduction/broadcast reversal | Manual-gradient status |
|---|---|---|---|---|---|

Populate it before implementing gradients.

### Assignment M4.5

For these operations, derive the vector-Jacobian product and shape behavior:

- Addition with broadcasting.
- Matrix multiplication.
- Elementwise multiplication.
- Mean.
- Sum with `keepdim=True`.
- Exponentiation.
- Reciprocal.
- Logarithm.
- Tanh.
- Index selection.
- Embedding lookup with repeated indices.

Then answer:

1. Why is writing a full Jacobian usually unnecessary?
2. Why must gradients be summed over broadcast dimensions?
3. Why does an embedding gradient require accumulation for repeated IDs?
4. Why can two algebraically equivalent forward expressions lead to differently convenient backward derivations?

### Deliverable

- Complete forward dependency graph.
- Gradient ledger.
- Operation derivative notes.

### Exit criteria

- Every forward node has a known shape and parent set.
- You can state the reverse traversal order.
- Broadcasting and repeated-index accumulation are understood before coding.
