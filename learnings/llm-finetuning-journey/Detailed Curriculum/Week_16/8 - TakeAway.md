# Week 16 TakeAway — Phase 2 Gate

**This week in 15 words:** Prove you can implement a modern LLM from memory. No reference. No shortcuts.

---

## Phase 2 Knowledge Map

```
Week 9:  Bahdanau attention         → score function, context vector, heatmap
Week 10: Attention Is All You Need  → scaled dot-product, MHA, sinusoidal PE
Week 11: Decoder-only (nanoGPT)     → causal mask, weight tying, CLM loss
Week 12: Modern improvements        → RMSNorm, SwiGLU, RoPE, GQA
Week 13: KV cache + sampling        → cached inference, top-k, top-p, temperature
Week 14: LLaMA papers               → production code reading, LLaMA 1/2/3 diffs
Week 15: GPT-2 reproduction         → mixed precision, grad accum, Flash Attention
Week 16: Gate                       → prove it all from memory
```

---

## Everything from Memory (Must Know)

```python
# RMSNorm
x / x.pow(2).mean(-1,keepdim=True).add(eps).sqrt() * gamma

# SwiGLU
down( silu(gate(x)) * up(x) )  # h = 8/3 * d, rounded to 256

# RoPE (LLaMA convention)
q1, q2 = q[...,:D//2], q[...,D//2:]
q_rot = cat([q1*cos - q2*sin, q1*sin + q2*cos], dim=-1)

# GQA repeat
k.repeat_interleave(n_heads // n_kv_heads, dim=1)

# Scaled dot-product attention
softmax( Q @ K.T / sqrt(d_k) ) @ V

# Causal mask
triu(ones(T,T), diagonal=1).bool()  # True = mask

# KV cache concat (dim=2 is seq dim)
k = cat([k_past, k_new], dim=2)

# Top-p
sort desc → cumsum → remove where cumsum - prob > p → renorm → sample

# Training loss
F.cross_entropy(logits[:,:-1,:].reshape(-1,V), x[:,1:].reshape(-1))

# Gradient accumulation
(loss / G).backward()  # always divide before backward
```

---

## Gate Pass Criteria (All Must Be True)

| Criteria | Target |
|---|---|
| Can implement MHA from memory | < 15 min, no reference |
| Can implement modern Block from memory | < 15 min, no reference |
| GPT-2 repro val loss | ≤ 3.27 |
| Gate project val loss | ≤ 1.5 |
| Gate project SQL looks valid | Yes |
| All 4 modern components in gate project | RMSNorm, SwiGLU, RoPE, GQA |
| Self-assessment score | ≥ 10/12 |

---

## If You Are Not Ready

Go back to:
- **Attention shapes / MHA**: Week 10
- **nanoGPT / CLM / causal mask**: Week 11
- **RMSNorm / SwiGLU / RoPE / GQA**: Week 12
- **KV cache / sampling**: Week 13
- **LLaMA details**: Week 14
- **Mixed precision / grad accum**: Week 15

Do not proceed to Phase 3 with unresolved gaps. Phase 3 will expose them.
