# Week 10 Assignment Solutions

## Task 1 — Scaled Dot-Product Attention

```python
import math
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    # scores: [batch, heads, seq_q, seq_k]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
```

**Expected output (unit test):**
- Output shape: `[batch, heads, seq_q, d_v]` — confirmed.
- `attn_weights.sum(dim=-1)` — all 1.0 (within 1e-5).

**Common gotchas:**
- Transposing the wrong dimensions: must be `K.transpose(-2, -1)`, not `K.transpose(0, 1)`.
- Applying mask after softmax — must be before; post-softmax masking does not zero out the scores properly.
- Scaling by `d_k` instead of `sqrt(d_k)` — extremely common, causes training to work but suboptimally.
- Not using `dim=-1` in softmax — wrong normalization direction.

---

## Task 2 — Multi-Head Attention Key Snippet

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.h = num_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, Q, K, V, mask=None):
        B, T, _ = Q.shape
        # project + split heads
        q = self.W_Q(Q).view(B, T, self.h, self.d_k).transpose(1, 2)
        k = self.W_K(K).view(B, -1, self.h, self.d_k).transpose(1, 2)
        v = self.W_V(V).view(B, -1, self.h, self.d_k).transpose(1, 2)
        out, _ = scaled_dot_product_attention(q, k, v, mask)
        # concat heads + project
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_O(out)
```

The `.contiguous()` call before `.view()` is necessary because `transpose` creates a non-contiguous tensor in memory.

**Common gotchas:**
- Forgetting `.contiguous()` before `.view()` — raises RuntimeError about non-contiguous tensors.
- Using `d_model` instead of `self.d_k` in the `view` — breaks head splitting.
- Passing a single boolean to `masked_fill` vs. a proper mask tensor — need a `[B, 1, T, T]` shaped mask for broadcasting.

---

## Task 4 — Scaling Experiment (Key Numbers)

Run this and record output:

```python
import torch, math
for d_k in [4, 16, 64, 256]:
    Q = torch.randn(1000, d_k)
    K = torch.randn(1000, d_k)
    raw   = (Q @ K.T)
    scaled = raw / math.sqrt(d_k)
    print(f"d_k={d_k}: raw std={raw.std():.2f}, scaled std={scaled.std():.2f}")
```

Expected output (approximate):
```
d_k=4:   raw std=2.01, scaled std=1.00
d_k=16:  raw std=4.03, scaled std=1.01
d_k=64:  raw std=8.04, scaled std=1.01
d_k=256: raw std=16.08, scaled std=1.01
```

The std of unscaled scores grows as `sqrt(d_k)`. When scores have std ~16, softmax saturates: the top value becomes close to 1.0 and all gradients through softmax vanish (the Jacobian of softmax is near-zero). Scaling by `1/sqrt(d_k)` keeps std near 1.0 regardless of dimension.

---

## How to Verify You Did It Right

1. Copy task: after 2000 steps, loss should be < 0.01 and the model should reproduce any 10-token sequence exactly.
2. Causal mask test: in decoder self-attention, `attn_weights[0, :, 0, 1:]` should be all zeros (position 0 cannot see position 1+).
3. Scaling experiment: std of unscaled scores grows as `sqrt(d_k)`; scaled is ~1.0 for all d_k values.
4. W&B shows smooth loss decrease with the warmup LR schedule (loss may spike briefly before warmup ends, then decrease steadily).
