# Week 12 TakeAway — Modern Architectural Improvements

**This week in 15 words:** RMSNorm, SwiGLU, RoPE, GQA — the four changes that turn GPT into LLaMA.

---

## Key Formulas

```
# RMSNorm
RMS(x) = sqrt( mean(x^2) + eps )
y = (x / RMS(x)) * gamma    # no beta, no mean subtraction

# SwiGLU
FFN(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )
intermediate_dim = round_to_256( int(8/3 * d_model) )

# RoPE frequencies
theta_i = 10000 ^ (-2i / d_k)   for i = 0..d_k/2-1
freqs[pos, i] = pos * theta_i
q_rot = apply_rotation(q, cos(freqs), sin(freqs))  # to Q and K only

# GQA KV cache savings
kv_cache_bytes = 2 * n_kv_heads * d_k * T * n_layers * 2  # FP16
```

---

## Key Code Patterns

```python
# RMSNorm
rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
return x / rms * self.weight

# SwiGLU forward
return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# RoPE apply (LLaMA convention)
q1, q2 = q[..., :D//2], q[..., D//2:]
q_rot = torch.cat([q1*cos - q2*sin, q1*sin + q2*cos], dim=-1)

# GQA: repeat KV heads
n_rep = n_heads // n_kv_heads
k = k.repeat_interleave(n_rep, dim=1)  # [B, n_heads, T, d_k]
v = v.repeat_interleave(n_rep, dim=1)
```

---

## Comparison Table

| Component | Original GPT | Modern (LLaMA-style) |
|---|---|---|
| Normalization | LayerNorm (w/ beta) | RMSNorm (no beta) |
| FFN activation | ReLU or GELU, 2 proj | SwiGLU, 3 proj, 8/3x dim |
| Position encoding | Learned abs. or sinusoidal | RoPE (relative, no wpe table) |
| Attention heads | MHA (n_kv=n_q) | GQA (n_kv < n_q) |

---

## Numbers to Remember

- RMSNorm: ~8% faster than LayerNorm (no mean computation)
- SwiGLU intermediate dim: `8/3 * d_model`, rounded to 256 multiple
- RoPE base: 10000 (LLaMA 1/2), 500000 (LLaMA 3)
- GQA in LLaMA-3 8B: n_q=32 heads, n_kv=8 heads → 4x KV cache reduction

---

## Decision Rules

- Apply RoPE to Q and K only. Never to V.
- RMSNorm weight = gamma (ones init), no beta parameter.
- SwiGLU: SiLU goes on the gate branch, not the up branch.
- GQA: `n_heads % n_kv_heads == 0` must hold.
- Register RoPE cos/sin as buffers, not parameters.

---

## Red Flags During Training

- NaN at step 1: RMSNorm weights are zero, or RoPE freqs computed with integer dtype.
- Loss plateaus immediately: SwiGLU `down_proj` input shape doesn't match `hidden_dim` × 2.
- OOM on long sequences: forgot GQA; KV cache growing with n_heads instead of n_kv_heads.
