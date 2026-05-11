# Week 12 Quiz — Modern Architectural Improvements

Calibration: mid-junior ML interview level. Focus on the "why" behind each architectural choice.

---

**Q1.** RMSNorm omits the mean-subtraction step of LayerNorm. What is the theoretical justification for this, and what is the practical benefit?

A) Mean subtraction causes numerical instability on GPUs; removing it prevents NaN  
B) The mean-centering step is redundant because the subsequent linear layer can absorb any offset; removing it saves compute  
C) RMSNorm uses the L2 norm instead of variance, which produces more stable gradients  
D) Mean subtraction interferes with residual connections by centering the residual stream at zero  

---

**Q2.** SwiGLU uses three projection matrices (`gate_proj`, `up_proj`, `down_proj`) instead of two. To keep total FLOPs approximately equal to a standard 4x MLP, what should the intermediate dimension be?

A) 4 × d_model (same as standard FFN, extra matrix is free)  
B) 2 × d_model  
C) (8/3) × d_model ≈ 2.67 × d_model  
D) d_model / 2  

---

**Q3.** You implement RoPE and apply it to Q, K, and V. Your model trains to a lower val loss than the baseline. A colleague says applying RoPE to V is wrong. Who is right?

A) You are right — applying RoPE to V encodes position in the retrieved values, improving quality  
B) Your colleague is right — RoPE should only be applied to Q and K; applying to V changes what is retrieved, not how attention scores are computed  
C) Both are valid; it's an empirical choice with no theoretical preference  
D) Your colleague is right, but only for causal (decoder) models — encoder models can apply RoPE to V  

---

**Q4.** A LLaMA-3 model has `n_heads=32` query heads and `n_kv_heads=8` key-value heads. What is the ratio of KV cache memory compared to standard MHA with 32 KV heads?

A) 8x smaller  
B) 4x smaller  
C) 2x smaller  
D) The same — KV cache size depends on sequence length, not number of heads  

---

**Q5.** During inference on a 1000-token context with Sliding Window Attention (window W=256), how many tokens can the model in the final layer "see" from the first position?

A) 256 (the window W)  
B) W × num_layers (effective receptive field grows linearly with depth)  
C) 0 — SWA cannot reach position 1 from the final layer because the window doesn't extend that far  
D) All 1000 — SWA only limits what is cached, not what is attended to  

---

**Q6 (short answer).** Explain what property of RoPE makes the attention score `q_m · k_n` depend only on the relative position `m - n` rather than absolute positions `m` and `n`. Why is this property desirable for language modeling?

---

**Q7 (short answer).** You are building a text-to-SQL model that needs to handle 8,192-token inputs (long database schemas). Your GPU has 24GB VRAM. You're considering using Mistral 7B (with SWA, window=4096) vs. LLaMA-2 7B (full attention). For a batch size of 4, estimate the KV cache memory savings from SWA at context length 8192 (in GB, assuming FP16, 32 layers, 32 heads, d_k=128).

---

**Q8 (scenario).** You replace LayerNorm with RMSNorm and GELU-MLP with SwiGLU in your nanoGPT. Training loss is now NaN after step 10. You revert to the baseline and NaN disappears. List 3 likely causes and how to debug each.
