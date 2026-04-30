# Week 16 Assignment Solutions — Phase 2 Gate Reference

This file provides reference answers for the self-assessment and a reference implementation skeleton for the gate project.

---

## Task 1 — Reference Answers

**1. Bahdanau attention:** Solved the fixed-size context vector bottleneck in seq2seq models. Score: `e_{i,j} = v_a^T tanh(W_a s_{i-1} + U_a h_j)`. Context vector: `c_i = sum_j(alpha_{i,j} * h_j)` — weighted sum of encoder hidden states where weights are softmax-normalized scores.

**2. Scaled dot-product:** `Attn(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`. Divide by sqrt(d_k) because for d_k-dimensional unit vectors, the dot product variance is d_k. Without scaling, large d_k → large scores → softmax saturation → near-zero gradients. Using d_k instead of sqrt(d_k) would over-scale and make the distribution too flat.

**3. MHA:** 4 matrices: W_Q, W_K, W_V (each [d_model, d_k]), W_O ([h*d_v, d_model]). d_k = d_v = d_model / num_heads. Multiple heads let each head specialize in different relationship types (local, syntactic, coreference, etc.).

**4. Causal mask:** Upper-triangular bool matrix, True for future positions. Applied as `masked_fill(mask, -inf)` before softmax. Without it, position i sees future tokens during training; at inference, future tokens don't exist → model outputs garbage.

**5. RMSNorm:** `y = x / sqrt(mean(x^2) + eps) * gamma`. Drops mean subtraction and beta. OK because: mean centering is redundant (next linear layer absorbs bias), and the scaling by 1/RMS is what actually stabilizes training.

**6. SwiGLU:** `FFN(x) = down(SiLU(gate(x)) * up(x))`. Three projections. Intermediate dim = `int(8/3 * d_model)` rounded to nearest 256 multiple.

**7. RoPE:** Key property: `q_m · k_n` depends only on `(m-n)` because RoPE uses rotation matrices with the property `R(m)^T R(n) = R(n-m)`. LLaMA 3 theta = 500,000.

**8. GQA:** `num_key_value_heads` = number of KV heads (< n_heads). `repeat_kv`: `k.repeat_interleave(n_heads // n_kv_heads, dim=1)`. KV savings: `n_kv / n_heads = 8/32 = 25%` of MHA cache.

**9. KV cache:** Stores all past K, V tensors for each layer. No causal mask needed because we only compute Q for the new token — past tokens are past by construction. GQA reduces `n_kv_heads`, and cache size scales linearly with `n_kv_heads`.

**10. Top-p:** Sort probs descending. Compute cumulative sum. Remove all tokens where cumulative sum (before that token) > p. Renormalize. Better than top-k because it adapts: when model is confident, only 2–3 tokens needed to reach 0.9; when uncertain, 100+ tokens included.

**11. Gradient accumulation:** Run G micro-batches, accumulate gradients, then take one optimizer step. Divide loss by G before backward because accumulated gradient = `sum(grad_i/G) = mean(grad_i)` = gradient of the full batch mean loss. BF16 vs FP16: BF16 has the same exponent range as FP32 (avoids overflow), just lower mantissa precision. FP16 can overflow for large activations.

**12. LLaMA 1 → 3 differences:** (1) Context: 2048 → 8192. (2) Vocab: 32k → 128k. (3) GQA: no → yes for all sizes. (4) RoPE theta: 10k → 500k. (5) Training tokens: 1T → 15T+.

---

## Gate Project — Reference Implementation Skeleton

```python
# gate_model.py — skeleton with correct structure

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.weight

def precompute_freqs(dim, max_len, base=10000):
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    pos = torch.arange(max_len).float()
    freqs = torch.outer(pos, theta)
    return freqs.cos(), freqs.sin()

def apply_rope(q, k, cos, sin):
    q1, q2 = q[..., :q.size(-1)//2], q[..., q.size(-1)//2:]
    k1, k2 = k[..., :k.size(-1)//2], k[..., k.size(-1)//2:]
    q_rot = torch.cat([q1*cos - q2*sin, q1*sin + q2*cos], dim=-1)
    k_rot = torch.cat([k1*cos - k2*sin, k1*sin + k2*cos], dim=-1)
    return q_rot, k_rot

def repeat_kv(x, n_rep):
    if n_rep == 1: return x
    return x.repeat_interleave(n_rep, dim=1)

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        h = int(8/3 * dim)
        h = ((h + 255) // 256) * 256
        self.gate = nn.Linear(dim, h, bias=False)
        self.up   = nn.Linear(dim, h, bias=False)
        self.down = nn.Linear(h, dim, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

The gate project passes if you can assemble these components into a working model, training loop, and generation function without help.

---

## How to Verify You Passed the Gate

1. Loss at step 0 is around `log(vocab_size)`. For SQL with ~70 character vocab: ~4.25.
2. Loss at step 3000 is below 1.5.
3. `gate_samples.txt` shows at least 5/10 samples that are syntactically recognizable SQL (SELECT...FROM, WHERE, etc.).
4. `gate_model.py` has no `nn.LayerNorm` calls (only your RMSNorm), no `wpe` lookup table (only RoPE), and uses `SwiGLU` not a vanilla FFN.
5. `gate_decision.md` is honest.
