# Week 14 Quiz Answers

---

**Q1. Answer: B**

**Why:** Chinchilla-compute-optimal means training a given model on a given compute budget achieves the best loss per FLOP during training. But the resulting model may not be the most efficient at inference time. A smaller model trained on more tokens may achieve the same loss as a larger model trained less — but the smaller model is cheaper to run in production (fewer parameters, less memory, faster inference). LLaMA explicitly optimized for inference cost, not training cost. This is the central insight of the paper: "We should be patient during training to get cheaper inference."

---

**Q2. Answer: B**

**Why:** GQA has `num_key_value_heads` KV heads (e.g., 8) but `num_attention_heads` query heads (e.g., 32). Before computing the attention dot product `Q @ K^T`, K and V must have the same number of heads as Q (for the batched matrix multiply to work). `repeat_kv` duplicates each KV head `n_rep = num_attention_heads / num_key_value_heads = 4` times, producing a tensor of shape `[batch, 32, seq, head_dim]`. No new parameters are introduced — it's a pure broadcasting operation.

---

**Q3. Answer: B**

**Why:** RoPE frequencies are `theta_i = base^(-2i/d)`. With `base=10000`, frequencies range from `10000^0 = 1` (fastest oscillation) to `10000^(-1) ≈ 0.0001` (slowest). At position 2048 and a fast frequency, the rotation may have cycled multiple times, causing positional aliasing. With `base=500000`, all frequencies are 50x slower — position 2048 looks similar to position 0 or 500 but different enough at position 100,000. This extends the range over which positions are distinguishable.

---

**Q4. Answer: B**

**Why:** KV cache memory = `2 × n_kv × d_k × T × n_layers × 2 bytes`. For 7B (32 heads, 128 d_k, 32 layers): KV cache for 4096 tokens = `2 × 32 × 128 × 4096 × 32 × 2 = 2.1 GB`. The model weights are 7B × 2 bytes = 14 GB. KV cache is 13% of the total. For 70B (64 heads, 128 d_k, 80 layers): KV cache = `2 × 64 × 128 × 4096 × 80 × 2 = 10.7 GB` on top of 140 GB weights. At 70B scale, KV cache is non-trivial to fit on any single GPU. GQA reduces it from 10.7 GB to ~1.3 GB — the difference between fitting on one GPU and not.

---

**Q5. Answer: B**

**Why:** The model checkpoint contains weights trained with `rope_theta=500000` — the attention layers learned to interpret positional information encoded with that theta. If you initialize RoPE with theta=10000, you apply different rotation angles to Q and K, producing attention scores that don't match what the model learned during training. For positions up to ~2048, the mismatch is small. Beyond 2048, the frequencies cycle differently and the model produces coherent content but with degraded long-range dependencies. This is a silent failure — no error, just bad generation.

---

**Q6 (short answer).**

**Argument for tying:** The embedding matrix maps tokens to vectors in the model's semantic space. The LM head maps from that space back to token probabilities. If the spaces are the same (tied), then tokens close in embedding space will have similar probability distributions — semantically similar tokens are interchangeable. This enforces a useful consistency and saves parameters.

**Counter-argument at LLaMA scale:** LLaMA has a 32k vocabulary (LLaMA 1/2) and d_model=4096. Tied weights: 32k × 4096 = 131M parameters. As a fraction of 7B total parameters, this is 1.9% — negligible. At this scale, the model has enough capacity that the embedding and logit spaces may benefit from independent learning. The model can represent subtle distinctions that tied weights might constrain. In practice, the quality difference is marginal; LLaMA's choice to separate them is an engineering preference, not a fundamental necessity.

---

**Q7 (short answer).**

1. Project input `hidden_states` into Q, K, V using three linear layers: `self.q_proj`, `self.k_proj`, `self.v_proj`.
2. Reshape Q from `[batch, seq, num_heads * head_dim]` to `[batch, num_heads, seq, head_dim]` (same for K, V with `num_key_value_heads`).
3. Apply RoPE to Q and K using precomputed cos/sin tables sliced to the current sequence length.
4. If a past KV cache exists, concatenate past K/V with current K/V along the sequence dimension.
5. Call `repeat_kv` on K and V to expand from `num_key_value_heads` to `num_attention_heads`.
6. Compute scaled dot-product attention: `softmax(Q @ K^T / sqrt(head_dim)) @ V`.
7. Reshape output from `[batch, num_heads, seq, head_dim]` back to `[batch, seq, num_heads * head_dim]`.
8. Project output through `self.o_proj` and return.

---

**Q8 (scenario).**

The primary cause of OOM with LLaMA 7B at batch=4, max_length=1024 is not the KV cache (which is small at 1024 tokens) but the activation memory during the forward and backward passes. With batch=4 and seq=1024, the intermediate activations in each attention layer and FFN layer occupy: `batch × seq × d_model × n_layers × bytes`. For LLaMA 7B: `4 × 1024 × 4096 × 32 × 2 ≈ 1GB` just for activations, plus gradients (another ~1GB), plus optimizer states (Adam uses 2× model size = 28GB for FP32). Total easily exceeds 24GB.

**What to check first:** Optimizer state memory. AdamW stores two momentum tensors per parameter in FP32: 7B × 2 × 4 bytes = 56GB — this alone doesn't fit. Solution: use gradient checkpointing (`model.gradient_checkpointing_enable()`), reduce batch size to 1 with gradient accumulation of 4 steps, and use 8-bit Adam (`bitsandbytes`). This will be the standard approach in Phase 4.
