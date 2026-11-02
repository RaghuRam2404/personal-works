[← Month 4 overview](<../Month4.md>) · [← Session 14](<Month4CurrSession14.md>) · [Session 16 →](<Month4CurrSession16.md>)

## Session 15 — Assemble GPT

### Objective

Implement a tested decoder-only Transformer from your own attention module.

### Required modules

Implement:

```python
class CausalSelfAttention(nn.Module)
class FeedForward(nn.Module)
class Block(nn.Module)
class GPTLanguageModel(nn.Module)
```

### GPT forward contract

```python
def forward(self, idx, targets=None):
    # idx: [B, T]
    # logits: [B, T, V]
    # loss: scalar or None
```

Required model path:

1. Token embedding lookup.
2. Learned position embedding lookup.
3. Addition into the residual stream.
4. Repeated pre-LayerNorm Transformer blocks.
5. Final LayerNorm.
6. Vocabulary projection.
7. Optional flattened cross-entropy.

### Feed-forward network

Use the lecture-style two-layer position-wise MLP:

```text
C → 4C → C
```

with nonlinearity and dropout. Explain why this network processes each position independently even though all positions are stored in the same tensor.

### Required tests

#### Shape tests

- Input `[B, T]` returns logits `[B, T, V]`.
- With targets, loss is scalar and finite.
- Every block preserves `[B, T, C]`.
- Parameter count is reported.

#### Causality test

Change future token IDs after position `i`. In evaluation mode, logits through position `i` must remain unchanged.

#### Gradient test

After one backward pass:

- Every expected trainable parameter has a finite gradient.
- No gradient exists for registered masks or integer token IDs.
- At least one gradient in each block is nonzero.

#### Tiny-batch overfit test

Train repeatedly on one tiny batch. The model should drive its loss substantially downward. This tests that the architecture, targets, gradients, and optimizer are connected before expensive full training.

#### Generation test

Implement naive generation:

```python
for _ in range(max_new_tokens):
    idx_cond = idx[:, -block_size:]
    logits, _ = model(idx_cond)
    next_logits = logits[:, -1, :]
    probs = softmax(next_logits)
    idx_next = multinomial(probs)
    idx = cat([idx, idx_next], dim=1)
```

Do not implement a KV cache this month. This intentionally recomputes the context and becomes Month 5's baseline.

### Parameter-count exercise

Estimate the parameter count symbolically before calling `sum(p.numel())`. Include:

- Token embedding.
- Position embedding.
- Per-block attention projections.
- Per-block feed-forward layers.
- LayerNorm affine parameters.
- Final vocabulary projection.

Explain why the `4C` feed-forward expansion usually contributes a large share of each block's parameters.

### Required configurations

#### Smoke configuration

Use a small model that can rapidly pass tests and overfit one batch, for example:

```text
block_size = 64 or 128
n_embd = 96 or 128
n_head = 4
n_layer = 2–4
dropout = 0.0 for deterministic tests
```

#### Project-shaped configuration

Instantiate a configuration near the Phase B project's approximately 10M parameters. The lecture reference uses `n_embd=384`, `n_head=6`, `n_layer=6`, and `block_size=256` on TinyShakespeare.[cite:189] Compute the actual count for your vocabulary and implementation rather than assuming the number.

### Assignment M4.15

Run this debugging ladder in order:

1. Shape-only forward pass.
2. Finite scalar loss.
3. Backward and gradient inspection.
4. One optimizer step changes parameters.
5. One-batch overfit.
6. Short training run.
7. Generation.

Record the first failure at each stage and its cause. Do not launch the long training run until all earlier stages pass.

### Deliverable

- `gpt/model.py`.
- Full test suite.
- Parameter estimate versus actual count.
- One-batch overfit curve.
- Naive generation implementation.

### Exit criteria

- GPT passes shape, causality, gradient, overfit, and generation tests.
- The approximately 10M-parameter configuration can perform forward and backward passes on the selected device.
- Every model component can be mapped to its function and shape.
