# Week 10 TakeAway — Attention Is All You Need

**This week in 15 words:** Self-attention replaces recurrence; parallelism unlocks scale; every architectural choice has a mathematical reason.

---

## Key Formulas

```
# Scaled dot-product attention
Attention(Q,K,V) = softmax( Q @ K.T / sqrt(d_k) ) @ V

# Multi-head attention
head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
MHA = Concat(head_1,...,head_h) @ W_O

# FFN
FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2

# Warmup LR schedule
lr = d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))

# Sinusoidal PE
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

---

## Key Code Pattern — Causal Mask

```python
# Build causal mask for decoder self-attention
T = seq_len
causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
# Shape: [T, T], True where future positions should be masked
# Broadcast to [B, 1, T, T] for multi-head attention
causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
```

---

## Key Code Pattern — Residual + LayerNorm

```python
# Post-LN (original paper):
output = layer_norm(x + sublayer(x))

# Pre-LN (modern, more stable):
output = x + sublayer(layer_norm(x))
```

---

## Numbers to Remember

| Parameter | Original Paper | Your toy run |
|---|---|---|
| d_model | 512 | 128 |
| num_heads | 8 | 4 |
| d_k = d_v | 64 | 32 |
| d_ff | 2048 | 256 |
| num_layers | 6 | 2 |
| warmup_steps | 4000 | 400 |

Scaling: `1/sqrt(d_k)` keeps pre-softmax score std ≈ 1.0 regardless of d_k.

---

## Decision Rules

- If loss stuck at log(vocab_size): causal mask is missing, LR is wrong, or positional encoding is not added.
- If gradients vanish: check residual connections are present. Check LR warmup is working.
- If attention weights are all equal: softmax on wrong dim, or scores not flowing gradients.
- Encoder self-attention: NO mask. Decoder self-attention: causal mask. Decoder cross-attention: source padding mask only.

---

## Red Flags During Training

- Loss never drops below 2.3 on a 10-token vocab copy task → causal mask or positional encoding broken.
- Loss oscillates wildly after warmup → LR too high post-warmup; reduce peak LR.
- All attention heads learn identical patterns → d_k too small or no random initialization of projections.
