# Week 12 Quiz Answers

---

**Q1. Answer: B**

**Why:** The re-centering step (`x - mean(x)`) was originally motivated by stabilizing training by preventing hidden state means from drifting. However, in practice, the subsequent linear/projection layer has its own bias term that can absorb any constant shift. The mean-centering computation — computing a mean over the feature dimension and subtracting it — accounts for roughly 7–8% of LayerNorm's compute. Zhang & Sennrich (2019) showed empirically that removing it has no negative effect on convergence or final quality. RMSNorm thus gets approximately 8% faster normalization for free.

**Why others are wrong:**
- A: NaN instability comes from overflow/underflow, not from mean subtraction specifically.
- C: RMSNorm uses root-mean-square, not L2 norm per se (RMS = L2/sqrt(n)).
- D: Residual connections add to the stream; mean-centering doesn't interfere with this.

---

**Q2. Answer: C**

**Why:** A standard 4x MLP uses two projections: `d_model → 4d → d_model`, totaling `2 × d_model × 4d_model = 8d² ` FLOPs. SwiGLU uses three projections: two of `d_model → h` and one of `h → d_model`, totaling `3 × d_model × h` FLOPs. Setting `3h = 8d_model` gives `h = 8/3 × d_model ≈ 2.67 × d_model`. In practice this is rounded to a multiple of 256 for hardware efficiency.

---

**Q3. Answer: B**

**Why:** RoPE encodes position information in the attention score computation — specifically in the `q · k` dot product. The mathematical property of RoPE (that `q_m · k_n` depends only on `m-n`) applies only to the Q and K vectors. The value V is retrieved after attention weights are computed; it doesn't participate in the position-dependent dot product. Applying RoPE to V would just rotate the value vectors by position, which doesn't encode relative position in any meaningful way and introduces a bias. Your lower val loss is likely coincidental noise at small scale.

---

**Q4. Answer: B**

**Why:** Standard MHA has 32 KV heads. GQA with 8 KV heads uses 32/8 = 4x fewer KV heads. Since KV cache size = `2 × n_kv_heads × d_k × T × num_layers` bytes, the memory scales linearly with `n_kv_heads`. Fewer KV heads = proportionally smaller KV cache.

---

**Q5. Answer: B**

**Why:** Sliding Window Attention limits each layer's attention to a window of W tokens. But in a deep network, information propagates across layers. In layer 1, token at position 1000 can see tokens 744–1000 (window 256). In layer 2, each of those tokens already carries information from their own windows. After L layers, the effective receptive field is approximately L × W. For a 32-layer model with W=256, the effective receptive field is ~8192 tokens, covering the full context.

---

**Q6 (short answer).**

RoPE multiplies each query vector by a rotation matrix `R(m)` and each key vector by `R(n)`, where `m` and `n` are positions. The rotation matrix for angle `theta` has the property: `R(m)^T R(n) = R(n-m)`. Therefore:

```
q_m · k_n = (R(m) q)^T (R(n) k) = q^T R(m)^T R(n) k = q^T R(n-m) k
```

The result depends only on the relative position `n-m`, not on `m` or `n` separately. This is desirable because language meaning is relative-position-dependent: "the cat sat on the mat" — "sat" is always two positions after "cat" regardless of where the sentence starts in a longer document. Absolute positions are arbitrary; relative positions carry the structural information.

---

**Q7 (short answer).**

KV cache formula: `2 × n_heads × d_k × T × n_layers × bytes_per_param`

For full attention (LLaMA-2 7B): `2 × 32 × 128 × 8192 × 32 × 2 bytes = 4.3 GB` per sequence in the batch.

For batch size 4: `4 × 4.3 GB = 17.2 GB` — leaves only 6.8 GB for model weights (7B at FP16 ≈ 14 GB). This doesn't fit.

For SWA (window=4096): effectively only W=4096 tokens are cached at once (older tokens can be evicted). `2 × 32 × 128 × 4096 × 32 × 2 = 2.15 GB` per sequence. For batch 4: `8.6 GB`. Total with model weights: `8.6 + 14 ≈ 22.6 GB` — just fits in 24 GB.

SWA makes the 8K context batch=4 case feasible. Full attention does not.

---

**Q8 (scenario).**

**Cause 1: RMSNorm weight initialization.** If `self.weight` was not initialized to ones (or is accidentally zeros), the normalization output is all zeros, causing NaN in subsequent layers. Diagnose: print `rmsnorm.weight` at init. Fix: `nn.Parameter(torch.ones(dim))`.

**Cause 2: SwiGLU with wrong intermediate dimension.** If `hidden_dim` is computed incorrectly (e.g., as 0 due to integer division), the linear layers have zero output dim, causing NaN. Diagnose: print `model.mlp.gate_proj.weight.shape`. Fix: assert `hidden > 0` in `__init__`.

**Cause 3: RoPE producing NaN from extreme values.** If `freqs` computation uses wrong dtype (e.g., `torch.int`), cos/sin will fail. Or if `base` is 0 or negative. Diagnose: print `cos.isnan().any()` immediately after precomputing RoPE freqs. Fix: ensure `torch.arange(0, dim, 2).float()` and verify base=10000.
