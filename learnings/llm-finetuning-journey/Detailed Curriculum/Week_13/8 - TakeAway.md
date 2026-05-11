# Week 13 TakeAway — KV Cache, Inference, Sampling

**This week in 15 words:** Cache K and V from past tokens; sample smartly — temperature, top-k, top-p; never recompute.

---

## Key Code Patterns

```python
# KV cache: concat past K/V with current K/V
if kv_cache is not None:
    k_past, v_past = kv_cache
    k = torch.cat([k_past, k], dim=2)  # dim=2 is seq dim after head split
    v = torch.cat([v_past, v], dim=2)
# Return new cache for next step
return output, (k, v)

# Top-k filtering
def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k)
    logits[logits < v[..., -1, None]] = float('-inf')
    return logits

# Top-p (nucleus) filtering
def top_p_logits(logits, p):
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)
    remove = (cumprobs - probs) > p
    sorted_logits[remove] = float('-inf')
    return logits.scatter(-1, sorted_idx, sorted_logits)

# Full sampling pipeline
def sample(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / temperature
    if top_k: logits = top_k_logits(logits, top_k)
    if top_p: logits = top_p_logits(logits, top_p)
    return torch.multinomial(F.softmax(logits, dim=-1), 1)
```

---

## KV Cache Memory Formula

```
bytes = 2 × n_kv_heads × d_k × T × n_layers × bytes_per_param
# FP16: bytes_per_param = 2
# GQA reduces n_kv_heads; directly reduces this number
```

---

## Decision Rules

- Causal mask: NOT needed in cached inference (past tokens are past by construction).
- Cache device: keep cache on the same device as the model (GPU/MPS/CPU).
- SQL generation temperature: 0.1–0.5 (deterministic enough for valid SQL).
- Creative generation: temperature 0.8–1.1 + top_p 0.9.
- Beam search: avoid for LLMs. Use for constrained structured output only if necessary.

---

## Speedup Reference

| Setting | Expected speedup (KV vs no-KV) |
|---|---|
| 4-layer nanoGPT, CPU | 5–10x |
| 7B model, GPU | 20–100x |
| 7B + GQA 4x fewer KV heads | 20–100x, +4x less memory |

---

## Red Flags During Inference

- Non-cached and cached paths produce different tokens (not just numerically close): cache is concatenated along wrong dim (use `dim=2` not `dim=1`).
- KV cache grows without bound: add `block_size` truncation or sliding window eviction.
- `top_p` sometimes produces zero valid tokens: forgot to ensure at least 1 token survives; add `remove[0] = False` or clamp.
- `temperature=0.0` causes division by zero: use `max(temperature, 1e-8)`.
