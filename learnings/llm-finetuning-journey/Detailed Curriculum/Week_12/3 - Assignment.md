# Week 12 Assignment — Modernizing nanoGPT

## Setup Checklist

- [ ] Your nanoGPT from Week 11 (`model.py`) as the base
- [ ] Create a new file `model_v2.py` — do not overwrite your Week 11 model
- [ ] GitHub branch `week-12-modern-arch`
- [ ] W&B project `week-12-modern-arch` with two runs: `baseline` and `modernized`
- [ ] Colab Free (or Mac MPS — this is small enough)

---

## Task 1 — Replace LayerNorm with RMSNorm

**Goal:** Implement `RMSNorm` from scratch and substitute it for every `nn.LayerNorm` in your nanoGPT.

**Requirements:**
- Implement `RMSNorm(nn.Module)`:
  - Constructor: `RMSNorm(dim, eps=1e-6)`
  - Learnable parameter: `weight` (gamma), shape `[dim]`, initialized to ones
  - No beta parameter, no mean subtraction
  - Forward: `x * (1 / sqrt(mean(x^2) + eps)) * self.weight`
- Replace all `nn.LayerNorm` calls in `Block` and in the final norm before the LM head
- Run a unit test: compare RMSNorm output shape and confirm it matches LayerNorm's output shape

**Deliverable:** `rmsnorm.py` (standalone module) + integrated into `model_v2.py`

---

## Task 2 — Replace ReLU/GELU MLP with SwiGLU

**Goal:** Replace the two-projection MLP with a three-projection SwiGLU FFN.

**Requirements:**
- New `SwiGLUMLP(nn.Module)` with three projections: `gate_proj`, `up_proj`, `down_proj`
- Intermediate dim: `int(8/3 * n_embd)` rounded up to nearest multiple of 256
- Forward: `down_proj( F.silu(gate_proj(x)) * up_proj(x) )`
  - Note: `F.silu(x) = x * sigmoid(x)` = Swish
- Replace the `MLP` class in `Block` with `SwiGLUMLP`
- Count parameters before and after: the 8/3 ratio should keep total params approximately the same as the 4x MLP

**Deliverable:** `swiglu.py` + integrated into `model_v2.py`

---

## Task 3 — Replace Learned Position Embeddings with RoPE

**Goal:** Implement Rotary Position Embeddings and remove the `wpe` learned embedding table.

**Requirements:**
- Implement `precompute_rope_freqs(dim, max_seq_len, base=10000)`:
  - Compute `theta_i = base^(-2i/dim)` for `i = 0, ..., dim//2 - 1`
  - Compute `freqs = outer(positions, theta)` — shape `[max_seq_len, dim//2]`
  - Return `(cos(freqs), sin(freqs))` — each shape `[max_seq_len, dim//2]`
- Implement `apply_rotary_emb(q, k, cos, sin)`:
  - For each of Q and K: split into pairs `(x_even, x_odd)`, apply rotation
  - Return rotated Q and K
- Remove `self.transformer.wpe` from `GPT`
- In `CausalSelfAttention.forward`, apply RoPE to Q and K before computing attention
- Precompute cos/sin tables at model init and register as buffers

**Hints:**
- Interleave vs. halve-and-rotate: there are two conventions. The LLaMA convention splits the vector in half (`x[..., :d_k//2]` and `x[..., d_k//2:]`) and rotates. The original RoPE paper interleaves pairs. Use the LLaMA convention.
- Only the first `T` rows of cos/sin are needed per forward pass: `cos[:T], sin[:T]`.

**Deliverable:** `rope.py` + integrated into `model_v2.py`

---

## Task 4 — Add GQA Support

**Goal:** Parameterize your attention to support `n_kv_heads < n_heads`.

**Requirements:**
- Add `n_kv_heads` to config (default: same as `n_heads` for backward compat)
- Assert `n_heads % n_kv_heads == 0`
- Project K and V to `[B, T, n_kv_heads * d_k]` instead of `[B, T, n_heads * d_k]`
- Before attention computation, repeat KV heads: `keys.repeat_interleave(n_heads // n_kv_heads, dim=1)`
- Test configuration: `n_heads=6, n_kv_heads=2` (3 query heads per KV head)
- Measure KV cache memory: print `n_kv_heads * d_k * block_size * n_layers * 2 * 2` bytes (for FP16)

**Deliverable:** Updated `model_v2.py` with GQA support.

---

## Task 5 — Benchmark: Baseline vs. Modernized

**Goal:** Compare training on the SQL corpus from Week 11.

**Requirements:**
- Train both `model.py` (Week 11 baseline) and `model_v2.py` (modernized) with identical hyperparameters: same architecture size, same data, same number of steps (3000), same random seed
- Log both to W&B as runs `baseline` and `modernized` within project `week-12-modern-arch`
- Report: final val loss for each, parameter counts, and any training speed difference

**Deliverable:** W&B comparison screenshot or table in `comparison.md`. GitHub commit `week-12-modern-arch`.

---

## Stretch Goals

- Implement Sliding Window Attention: restrict each token to attend to only the previous W tokens (W=32 for this small model). Compare quality and speed vs. full attention.
- Visualize RoPE: plot the cosine similarity between pairs of position embeddings. Does it decay with distance?
- Implement Flash Attention 2 signature (torch.nn.functional.scaled_dot_product_attention with `is_causal=True`) as a drop-in replacement. Measure speedup vs. manual attention.
