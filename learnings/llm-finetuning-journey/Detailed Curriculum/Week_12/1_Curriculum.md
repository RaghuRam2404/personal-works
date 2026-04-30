# Week 12 — Modern Architectural Improvements (RMSNorm, SwiGLU, RoPE, GQA, SWA)

## Learning Objectives

By end of this week, you will be able to:

- Implement RMSNorm and explain why it drops the re-centering step of LayerNorm
- Implement SwiGLU and explain why gated activations improve FFN expressiveness
- Implement Rotary Position Embeddings (RoPE) using complex number rotation
- Explain Grouped-Query Attention (GQA) and compute its KV head memory savings
- Modify your nanoGPT to replace all four components and compare val loss
- Describe Sliding Window Attention and when it is appropriate

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Read papers (Pre-LN/RMSNorm/SwiGLU — ~15p total) | 1 hr |
| Read RoPE paper (2104.09864) — sections 1–3 | 1 hr |
| Read GQA paper (2305.13245) | 0.5 hrs |
| Watch Umar Jamil LLaMA video (1h10m) | 1.25 hrs |
| Implement and integrate all 4 improvements into nanoGPT | 3.5 hrs |
| Compare val loss, commit | 0.75 hrs |

---

## Concepts

### Why These Improvements Exist

The original Transformer from Week 10 uses Post-LN, learned absolute positional embeddings, ReLU activations, and full multi-head attention where every head reads/writes the full KV cache. Between 2019 and 2024, each of these choices was improved. These improvements are not cosmetic — they are what separate a research prototype from a production LLM. LLaMA, Mistral, Qwen, and Gemma all use exactly this combination.

### RMSNorm (Root Mean Square Layer Normalization)

Standard LayerNorm:
```
y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
```

RMSNorm drops the mean-subtraction (re-centering) and the bias (beta):
```
y = x / RMS(x) * gamma
  where RMS(x) = sqrt( mean(x^2) )
```

Why? The re-centering step (`x - mean(x)`) accounts for roughly 7–8% of LayerNorm's compute. Dropping it is essentially free if it doesn't hurt performance — and empirically it doesn't. The scaling by `1/RMS(x)` is what actually stabilizes training. The bias `beta` is also unnecessary because the following linear layer can absorb any constant offset. Zhang & Sennrich (2019) showed RMSNorm achieves equivalent or better performance than LayerNorm with less compute.

### SwiGLU

Standard FFN in GPT-2:
```
FFN(x) = max(0, x W_1 + b_1) W_2 + b_2    # ReLU
```

SwiGLU (from "GLU Variants Improve Transformer", Noam Shazeer 2020):
```
FFN_SwiGLU(x) = (Swish(x W_1) * (x W_gate)) W_2
  where Swish(z) = z * sigmoid(z)
```

There are now three projection matrices (`W_1`, `W_gate`, `W_2`) instead of two. The gate allows the network to modulate what information passes through at each position. The intuition: `W_gate` learns to "open" or "close" channels in the hidden representation based on the input. This gating mechanism dramatically improves expressiveness per parameter.

The standard practice: shrink the intermediate dimension from 4x to approximately 2.67x (8/3) to keep total FLOPs constant with three matrices. LLaMA uses `4 * (2/3) * d_model` as the intermediate dimension, rounded to a multiple of 256.

### Rotary Position Embeddings (RoPE)

Learned absolute positional embeddings (Week 11) have a key weakness: they cannot generalize beyond the training context length. Sinusoidal encodings (Week 10) generalize better but are fixed and additive — they conflate content and position information.

RoPE (Su et al., 2021) encodes position by rotating the query and key vectors in a position-dependent way before computing attention. For a 2D subspace of the query vector, the rotation by angle `m * theta` (where `m` is the position and `theta` is a frequency parameter) gives:

```
rotate(x, m) = x * cos(m*theta) + x_perp * sin(m*theta)
```

This is applied independently to each pair of dimensions using frequency bases `theta_i = 10000^(-2i/d_k)` (same structure as sinusoidal PE).

The key property: after rotating both Q and K, the dot product `q_m · k_n` depends only on the relative position `m - n`, not on the absolute positions. This is exactly what we want — attention should care about relative distance, not absolute position. RoPE naturally encodes relative position in the attention scores without modifying the values.

In practice, this means the model can generalize to longer sequences than it was trained on (with some degradation), and extended context techniques (RoPE scaling) can push this further.

```python
def apply_rotary_emb(x, cos, sin):
    # x: [B, H, T, D]
    # Rotate pairs of dimensions
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
```

### Grouped-Query Attention (GQA)

Multi-head attention (MHA) has `h` query heads, `h` key heads, `h` value heads. For inference, you must store the KV cache for every layer — that's `2 * h * d_k` floats per token per layer.

For LLaMA-2 70B: `h=64`, `d_k=128`, 80 layers → `2 * 64 * 128 * 80` = 1.3M floats per token. At FP16, generating 4096 tokens requires ~10GB just for the KV cache on top of the model weights.

Multi-Query Attention (MQA, Shazeer 2019) uses a single KV head shared by all query heads: `h` query heads, `1` KV head. This reduces KV cache by `h` times but slightly hurts quality.

GQA (Ainslie et al., 2023) is the compromise: `G` KV groups where `1 ≤ G ≤ h`. Each group of `h/G` query heads shares one KV head. LLaMA-3 8B uses `h=32` query heads and `8` KV heads (G=8). This gives an 4x KV cache reduction with minimal quality loss.

```python
# In GQA forward pass:
# queries: [B, n_q_heads, T, d_k]
# keys:    [B, n_kv_heads, T, d_k]  -- fewer heads
# Repeat KV heads to match query heads
n_rep = n_q_heads // n_kv_heads
keys = keys.repeat_interleave(n_rep, dim=1)   # broadcast KV heads
vals = vals.repeat_interleave(n_rep, dim=1)
```

### Sliding Window Attention (SWA)

Standard self-attention has O(T^2) complexity — every token attends to every other token. For long contexts (T=32k, 128k), this is prohibitively expensive.

Sliding Window Attention (Beltagy et al., 2020, used in Mistral) restricts each token to attend to only the W nearest tokens in both directions (for a window of size 2W+1). This gives O(T*W) complexity. Information from distant tokens still propagates through multiple layers — after L layers, each token has an effective receptive field of L*W tokens.

Mistral 7B uses W=4096 (window of ~4k tokens) with a context of 8k, giving 2x computational savings per layer with minimal quality degradation for most tasks.

## Connections

**Building on:** Week 11's nanoGPT (which you will modify), Week 10's attention mechanism.

**Used in:** Week 14 (LLaMA paper — all four improvements are in LLaMA), Week 15 (GPT-2 repro — you'll use Flash Attention, related), Week 13 (GQA directly reduces KV cache size).

## Common Misconceptions / Pitfalls

- **RoPE is applied to Q and K only, not V.** The values don't need position encoding because they are selected, not scored.
- **GQA: repeating KV heads vs. averaging.** `repeat_interleave` is correct. Don't average the KV heads — that changes the semantics.
- **SwiGLU intermediate dimension.** The standard is `8/3 * d_model` rounded to a multiple of 256. If you use `4 * d_model`, you get 1.5x more FLOPs than intended.
- **RMSNorm gamma is still a learned parameter.** You drop the beta (bias) but keep the per-element scaling gamma.
- **RoPE frequency computation.** Use `theta_i = 10000^(-2i/d_k)` for `i = 0, 1, ..., d_k/2 - 1`. A common bug is computing frequencies over the full d_k range instead of d_k/2.
