# Week 13 Assignment Solutions

## Task 1 — KV Cache in CausalSelfAttention

```python
def forward(self, x, kv_cache=None):
    B, T, C = x.shape
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    nh, hs = self.n_head, C // self.n_head

    q = q.view(B, T, nh, hs).transpose(1, 2)   # [B, nh, T, hs]
    k = k.view(B, T, nh, hs).transpose(1, 2)
    v = v.view(B, T, nh, hs).transpose(1, 2)

    if kv_cache is not None:
        k_past, v_past = kv_cache
        k = torch.cat([k_past, k], dim=2)        # [B, nh, T_past+T, hs]
        v = torch.cat([v_past, v], dim=2)
        # No causal mask needed: Q is for new tokens only,
        # K/V are all past tokens — causality is implicit.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = F.softmax(att, dim=-1)
    else:
        # Training mode: apply causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

    y = att @ v                                  # [B, nh, T, hs]
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return self.resid_dropout(self.c_proj(y)), (k, v)
```

**Common gotchas:**
- Concatenating along dim=1 instead of dim=2 — concatenates along head dim, not sequence dim.
- Returning `(k, v)` before concatenation — next step won't have the new token's K/V.
- Applying causal mask in cache mode — incorrectly masks past tokens for the current query.

---

## Task 3 — Sampling Strategies

```python
def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    # Step 1: temperature
    logits = logits / max(temperature, 1e-8)

    # Step 2: top-k
    if top_k is not None and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[..., -1, None]] = float('-inf')

    # Step 3: top-p (nucleus)
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative prob above threshold
        # Shift right to include the token that crosses the threshold
        sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
        sorted_logits[sorted_indices_to_remove] = float('-inf')
        logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()
```

**Expected test results:**
- `top_k=1`: always returns `logits.argmax()` (greedy).
- `top_p=1.0`: probs matches `softmax(logits)` exactly — all tokens eligible.
- `temperature=0.01, top_k=None, top_p=None`: effectively greedy (near-deterministic).

---

## Task 4 — Benchmark Expected Numbers

On a 4-layer, 256-dim nanoGPT generating 200 tokens from a 50-token prompt on CPU:

```
Without KV cache: 200 tokens in ~8.5s  = ~23 tok/s
With KV cache:    200 tokens in ~1.2s  = ~167 tok/s
Speedup: ~7.3x
```

Numbers vary significantly by hardware and model size. On Colab T4 GPU, expect 15–30x speedup. If speedup is less than 3x, check that you're actually using single-token inference in the cached path (T=1 per step).

**Common gotchas:**
- Not calling `model.eval()` and `torch.no_grad()` — dropout and gradient tracking slow down inference significantly.
- Processing the full sequence at every step in the "cached" path — this means you didn't implement the cache correctly.
- Timing includes model loading — make sure both benchmarks start with the model already loaded.

---

## How to Verify You Did It Right

1. Non-cached and cached inference must produce identical logits: `torch.allclose(logits_no_cache, logits_cached, atol=1e-4)` → True.
2. `sample_next_token(logits, top_k=1)` == `logits.argmax()` → always.
3. After 200 tokens, `kv_cache[0][0].shape[2]` == 250 (50 prompt + 200 generated).
4. Speedup ≥ 5x confirmed by timing printout.
