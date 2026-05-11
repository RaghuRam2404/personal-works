# Week 16 Quiz — Phase 2 Comprehensive Review

This quiz covers all of Phase 2 (Weeks 9–15). Treat it as a final exam for the phase. You should be able to answer all questions without notes. If you cannot, identify which week to revisit.

---

**Q1.** A colleague's transformer is generating text where every token is the same character repeated. Which of the following is the most likely cause?

A) The top-p sampling threshold is too low  
B) The causal mask is missing, allowing the model to attend to future tokens and collapse to a fixed point  
C) The softmax in the attention module is applied along the wrong dimension  
D) The KV cache is not being updated correctly during generation  

---

**Q2.** You are implementing GQA. Your config has `n_heads=32, n_kv_heads=4`. Your K tensor after projection has shape `[B, 4, T, d_k]`. Before computing the attention dot product with Q of shape `[B, 32, T, d_k]`, you must expand K. What is the correct operation?

A) `K.expand(B, 32, T, d_k)` — zero-copy broadcast  
B) `K.repeat_interleave(8, dim=1)` — produces `[B, 32, T, d_k]`  
C) `K.reshape(B, 32, T//8, d_k)` — redistribute sequence positions  
D) `K.unsqueeze(2).expand(B, 4, 8, T, d_k).reshape(B, 32, T, d_k)` — equivalent to repeat_interleave  

---

**Q3.** During GPT-2 124M training with `grad_accum_steps=32`, you forget to call `optimizer.zero_grad()` between logical batches (but you do call it once at the start of training). What happens after the second logical batch?

A) Nothing — gradients from batch 2 overwrite batch 1 gradients  
B) Gradients from batches 1 and 2 accumulate, causing a gradient that is 2x too large for batch 2's optimizer step  
C) Training proceeds normally because `loss.backward()` accumulates gradually  
D) PyTorch raises an error when gradients are not zeroed  

---

**Q4.** Which of the following best describes the residual stream view of transformer computation?

A) Each layer transforms the input entirely; the residual is added for numerical stability only  
B) Each sublayer reads from a persistent vector x, computes a delta, and writes it back via addition; the final x is a superposition of all layer contributions  
C) The residual stream carries gradient information backward; it doesn't affect forward pass semantics  
D) Residual connections are only present in the encoder, not the decoder  

---

**Q5.** You train your Phase 2 gate project model for 3000 steps and val loss reaches 1.45. Generated SQL looks like: `SELECT users WHERE FROM id name GROUP`. The syntax is wrong. What is the most likely issue — model quality or training?

A) The model is undertrained; increase steps to 10,000  
B) The character-level model has learned SQL characters but not SQL grammar; val loss of 1.45 is reasonable for character-level — the model needs more data  
C) Temperature is too high during generation; reduce to 0.1  
D) The KV cache is corrupted, causing generation to produce tokens from wrong positions  

---

**Q6 (short answer).** Write, from memory, the complete PyTorch forward method for `RMSNorm` and `SwiGLU`. Include all tensor operations.

---

**Q7 (short answer).** You have a 7B model running inference on a server with 24GB VRAM. The model weights take 14GB (FP16). You want to generate 4096 tokens with batch size 1. Calculate whether the KV cache will fit, assuming 32 layers, 32 Q heads, 8 KV heads, d_k=128. Show your calculation.

---

**Q8 (scenario — Phase 2 synthesis).** You are given a `modeling_llama.py` implementation and told it has a bug: generated text is incoherent for prompts longer than 256 tokens but fine for shorter prompts. Based on everything you learned in Phase 2, list 4 hypotheses in order of likelihood and describe how you would diagnose each.
