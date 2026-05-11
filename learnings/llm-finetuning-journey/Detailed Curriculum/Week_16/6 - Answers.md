# Week 16 Quiz Answers — Phase 2 Comprehensive Review

---

**Q1. Answer: C**

**Why:** If softmax is applied along the wrong dimension — for example, `dim=0` or `dim=-2` instead of `dim=-1` — the attention weights don't sum to 1 across source positions. The weighted sum of values then doesn't produce a meaningful combination. A common symptom is the model collapsing to predict the same token at every position. This can also happen if the softmax weights are all equal (uniform), causing the context vector to be a flat average that always maps to the same output token via the LM head. C is more directly diagnostic than the other options.

---

**Q2. Answer: B (or D — both are correct)**

**Why:** `repeat_interleave(8, dim=1)` takes each of the 4 KV heads and repeats it 8 times consecutively, producing `[B, 32, T, d_k]`. This is the semantically correct operation for GQA: each group of 8 query heads shares one KV head, and `repeat_interleave` produces the correctly interleaved layout.

Option D (`unsqueeze + expand + reshape`) is also correct and is effectively what HuggingFace's `repeat_kv` uses internally (with `expand` for zero-copy, then `reshape`).

Option A (`expand`) only works if you add a dimension first — `K.expand(B, 32, T, d_k)` would fail because K has 4 heads, not 1.

---

**Q3. Answer: B**

**Why:** `optimizer.zero_grad()` zeros all `.grad` attributes of the model parameters. If you don't call it between logical batches, the gradient from batch 2's micro-steps is added to the gradient still sitting in `.grad` from batch 1. After the batch 2 optimizer step, the gradient contains 64 micro-steps worth of accumulated gradient instead of 32. This is equivalent to training with double the intended learning rate for batch 2. After more batches, the error compounds further. Always call `optimizer.zero_grad()` at the start of every logical batch (before the micro-step loop).

---

**Q4. Answer: B**

**Why:** The residual stream is the key conceptual model for understanding transformer computation, articulated clearly by Anthropic's interpretability research. At each layer, the sublayer (attention or FFN) reads from the current state of x, computes a residual delta, and adds it back: `x = x + delta`. The final x at the last layer is the sum of all deltas contributed by every sublayer across all layers, plus the initial embedding. Each layer is writing information to a shared channel. This is why residual connections are essential: without them, there is no persistent channel — each layer would have to learn the full function from scratch.

---

**Q5. Answer: C**

**Why:** Val loss of 1.45 on a character-level SQL model is reasonable — the model has learned character statistics. But incoherent SQL syntax during generation usually indicates temperature is too high. At higher temperatures (e.g., 1.0), tokens with moderate probability (like space or wrong keywords) are sampled more often. For SQL, which has very rigid token sequences (SELECT must be followed by column names, not FROM), high temperature causes syntax violation. Reducing temperature to 0.1–0.3 will make the model stick to its most confident predictions, which for a reasonably trained SQL model will be syntactically correct keyword placement. Verify by generating with temp=0.1 — if SQL is valid, temperature was the issue.

---

**Q6 (short answer).**

```python
# RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight

# SwiGLU
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

If you could write both of these from memory without hesitation, you pass this dimension of the gate.

---

**Q7 (short answer).**

KV cache: `2 × n_kv_heads × d_k × T × n_layers × bytes_per_param (FP16=2)`

= `2 × 8 × 128 × 4096 × 32 × 2`

= 2 × 8 = 16
× 128 = 2048
× 4096 = 8,388,608
× 32 = 268,435,456
× 2 = 536,870,912 bytes = **512 MB**

Total VRAM: model weights (14 GB) + KV cache (0.5 GB) = **14.5 GB** — fits in 24 GB with 9.5 GB to spare.

With full MHA (n_kv=32 instead of 8): KV cache = 512 MB × 4 = **2 GB** → total 16 GB. Still fits, but GQA gives 1.5 GB back. At batch size 8 with full MHA: 8 × 2 GB = 16 GB KV cache + 14 GB model = 30 GB → OOM. With GQA: 8 × 0.5 GB + 14 GB = 18 GB → fits. This is why GQA matters for serving.

---

**Q8 (scenario).**

**Hypothesis 1 (most likely): RoPE not sliced correctly for position.** If the cos/sin tables are not indexed by the correct positions during KV cache inference (e.g., always using `pos[:T_new]` instead of `pos[T_past:T_past+T_new]`), positions beyond 256 are encoded with wrong rotations. For prompts ≤ 256, positions 0-255 are correct; beyond that, wrong. Diagnose: print the position indices being passed to RoPE during generation at step 256 — should be 256, not 0.

**Hypothesis 2: KV cache concatenation bug.** If new K/V are concatenated in the wrong dimension (e.g., dim=1 instead of dim=2), the cache has wrong shape beyond a certain length. Diagnose: print `kv_cache[0][0].shape` at step 250 (should be `[B, n_heads, 250, d_k]`) and at step 260.

**Hypothesis 3: block_size=256 being applied as a hard limit.** If the attention mask is built for `min(T, 256)` and doesn't correctly handle T > 256, attention over the full context is corrupted. Check if there's a `[:block_size]` slice applied incorrectly to the position indices.

**Hypothesis 4: Causal mask not correctly sized for cache mode.** If the causal mask is precomputed as `[block_size, block_size]` and not correctly sliced during inference, tokens at position > 256 may see wrong masked positions. Diagnose: run inference with a 300-token prompt and print the attention weights for the token at position 257 — should attend to all positions 0-256, not a subset.
