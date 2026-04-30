# Week 13 — KV Cache, Inference Optimization, and Sampling

## Learning Objectives

By end of this week, you will be able to:

- Explain exactly what the KV cache stores and why it eliminates redundant computation at inference time
- Implement a working KV cache from scratch in your nanoGPT
- Implement temperature, top-k, and top-p (nucleus) sampling from scratch
- Benchmark inference speed with and without KV cache and explain the speedup
- Describe the trade-offs between greedy, top-k, top-p, and temperature sampling
- Explain why beam search is rarely used in modern LLM inference

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Read The Illustrated GPT-2 (KV cache section) | 0.75 hrs |
| Read HuggingFace "How to generate text" blog post | 0.5 hrs |
| Read HF Generation strategies docs | 0.5 hrs |
| Implement KV cache in nanoGPT | 2.5 hrs |
| Implement all sampling strategies, benchmark | 2.5 hrs |
| Write commit + notes | 0.75 hrs |

---

## Concepts

### Why Inference is Different from Training

During training, you feed a full sequence of length T, compute all T positions in parallel (enabled by the causal mask), and backpropagate. The entire sequence is processed in one GPU kernel call.

During inference (autoregressive generation), you generate one token at a time. After generating token `t`, you append it to the context and run another forward pass to generate token `t+1`. Naively, this means recomputing the attention for all `t` previous tokens every time you generate a new one — an O(T^2) operation overall.

The KV cache eliminates this redundancy.

### The KV Cache

In the attention computation `softmax(Q K^T / sqrt(d_k)) V`, when generating token `t`:
- The query Q is computed only for the new token (position `t`)
- The keys K and values V are computed for ALL positions 1..t

But K and V for positions 1..t-1 were already computed when generating tokens 1..t-1. There's no need to recompute them — they haven't changed (the weights are fixed, and past inputs are fixed). So you cache them.

The KV cache stores, for each layer: a tensor of shape `[batch, n_kv_heads, T_cached, d_k]` for K and the same for V. When generating token `t+1`:
1. Compute K and V for position `t+1` only (shape `[batch, n_kv_heads, 1, d_k]`)
2. Concatenate with cached K, V: new cache shape is `[batch, n_kv_heads, t+1, d_k]`
3. Compute attention: Q shape `[batch, n_heads, 1, d_k]` × K shape `[batch, n_kv_heads, t+1, d_k]`
4. Output: a single new token's representation

This reduces per-step computation from O(T^2) to O(T) — you only compute one new key/value pair and attend over the full cache once.

**Memory cost:** The KV cache grows linearly with generated tokens: `2 × n_kv_heads × d_k × T_generated × n_layers × bytes`. For a 7B model generating 4096 tokens, this is several GBs (see Week 12 GQA discussion). This is why GQA is important — smaller `n_kv_heads` directly reduces KV cache memory.

### Implementing KV Cache in nanoGPT

The key change is that during inference, `CausalSelfAttention.forward` receives the current token(s) only, but needs access to past K, V tensors:

```python
def forward(self, x, kv_cache=None):
    B, T, C = x.shape
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    # reshape to [B, n_head, T, d_k]
    ...
    if kv_cache is not None:
        k_past, v_past = kv_cache
        k = torch.cat([k_past, k], dim=2)   # concat along seq dim
        v = torch.cat([v_past, v], dim=2)
    new_kv = (k, v)  # return for next step
    ...
    return y, new_kv
```

During inference, T=1 (you process one new token at a time). The full attention is computed over k (which has grown to the full context length), but only one new row of Q is projected.

### Sampling Strategies

After the forward pass, the model produces logits of shape `[vocab_size]` for the next token position. How you sample from these logits determines the character of the generated text.

**Greedy (argmax):**
```
next_token = argmax(logits)
```
Always picks the single highest-probability token. Fast and deterministic. Produces repetitive, safe text. For SQL generation, this often works well since SQL is highly constrained.

**Temperature:**
```
probs = softmax(logits / T)
next_token = multinomial(probs, 1)
```
Divides logits by temperature `T` before softmax. `T < 1` makes the distribution sharper (more greedy). `T > 1` makes it flatter (more random). `T = 1` is default. For SQL generation, use `T = 0.1`–`0.5`. For creative writing, use `T = 0.8`–`1.2`.

**Top-k:**
```
logits_topk, _ = torch.topk(logits, k)
logits[logits < logits_topk[-1]] = -inf
probs = softmax(logits)
next_token = multinomial(probs, 1)
```
Keep only the top k logits, set the rest to -inf. Prevents sampling from the long tail of unlikely tokens. `k=50` is a common default. Combined with temperature: apply temperature first, then top-k.

**Top-p (Nucleus Sampling, Holtzman et al. 2019):**
```
sorted_probs, sorted_idx = torch.sort(softmax(logits), descending=True)
cumulative = torch.cumsum(sorted_probs, dim=-1)
# Remove tokens where cumulative probability exceeds p
remove = cumulative > p  # shift right to keep the token that crosses p
remove[..., 1:] = remove[..., :-1].clone()
remove[..., 0] = False
sorted_probs[remove] = 0
# Renormalize and sample
...
```
Instead of a fixed top-k, keep the smallest set of tokens whose cumulative probability exceeds `p` (commonly `p=0.9` or `p=0.95`). The size of this set adapts to the distribution: for highly confident predictions, only 2–3 tokens are kept; for uncertain positions, 50+ tokens are kept. This is more principled than top-k for variable-confidence distributions.

**Why Not Beam Search for LLMs?**
Beam search was dominant in neural MT (2014–2018). It maintains B hypotheses in parallel and keeps the top-B at each step. But in modern LLMs:
1. Beam search is expensive: B forward passes per step vs. 1 for greedy/sampling.
2. Beam search with large LLMs produces degenerate, repetitive text (the "beam search curse" — high-probability sequences are often boring).
3. Sampling with top-p/temperature produces more diverse and natural output.
4. For constrained tasks like SQL, greedy or very low temperature works fine without the expense of beam search.

HuggingFace's `generate()` supports beam search but recommends sampling for creative tasks and greedy/constrained decoding for structured output.

### Speculative Decoding (Preview)

A powerful inference optimization (not implemented this week): run a small "draft" model to generate K tokens quickly, then verify all K at once with the large model using a single parallel forward pass. If all K are accepted, you've generated K tokens in one step of the large model. Speedups of 2–3x are common. This is the technique used in production systems like ChatGPT.

## Connections

**Building on:** Week 11 nanoGPT (your base implementation), Week 12 GQA (which directly reduces KV cache size), Week 9 (the "key/value" abstraction in Bahdanau attention is the ancestor of KV cache).

**Used in:** Week 14 (LLaMA uses GQA to reduce KV cache), Week 15 (GPT-2 repro — KV cache is needed for fast generation), Phase 5 (SQL generation quality depends critically on sampling strategy).

## Common Misconceptions / Pitfalls

- **KV cache is not the same as model weights.** Model weights are fixed after training. KV cache is the accumulated K, V tensors for the current generation context; it changes at every step.
- **The causal mask is not needed at inference time with KV cache.** Since you only compute Q for the new token (position T), and K/V for positions 1..T, the attention naturally only sees past tokens — no mask needed. If you apply the causal mask, you'd incorrectly mask out some of the cached K/V values.
- **Top-p vs top-k order.** Apply temperature first, then top-p or top-k. Applying them in the wrong order changes the distribution.
- **KV cache grows indefinitely for long contexts.** You need to handle the case where the context exceeds `block_size`. Truncate the oldest tokens from the cache (sliding window inference).
