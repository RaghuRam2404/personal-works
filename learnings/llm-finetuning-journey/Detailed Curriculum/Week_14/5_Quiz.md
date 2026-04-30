# Week 14 Quiz — LLaMA Papers and Production Code

Calibration: mid-junior ML interview level. Questions focus on "why" decisions were made.

---

**Q1.** LLaMA 1 was trained on approximately 1.4 trillion tokens, far more than the Chinchilla-compute-optimal amount for a 7B model. What is the practical motivation for over-training (training beyond the compute-optimal token count)?

A) Over-training improves performance on benchmark tasks because the model sees more diverse examples  
B) Over-training produces a model that is more efficient at inference time, even if training was suboptimal  
C) Touvron et al. were unaware of the Chinchilla paper when designing LLaMA  
D) Over-training is required to achieve weight convergence in pre-RMSNorm architectures  

---

**Q2.** In `modeling_llama.py`, the `repeat_kv` function is called on K and V tensors before computing attention. What does it do, and why is it necessary for GQA?

A) It compresses the KV heads from `num_attention_heads` to `num_key_value_heads` to save memory  
B) It expands the `num_key_value_heads` KV tensors to match `num_attention_heads` by repeating each KV head multiple times  
C) It pads the K/V sequences to a multiple of `num_key_value_groups` for CUDA efficiency  
D) It transposes K and V from [batch, seq, heads, head_dim] to [batch, heads, seq, head_dim]  

---

**Q3.** LLaMA 3 8B uses `rope_theta=500000` while LLaMA 1 used `rope_theta=10000`. What is the effect of increasing theta on the RoPE frequencies?

A) Larger theta makes all frequencies higher, helping the model attend to more distant tokens  
B) Larger theta makes the base frequencies lower (slower oscillation), so position encodings stay distinct across longer sequences  
C) Theta controls the number of dimensions over which RoPE is applied; larger theta covers more dimensions  
D) Theta affects the initialization scale of the RoPE parameters; larger theta prevents NaN  

---

**Q4.** LLaMA 2 introduced GQA only for the 34B and 70B models, not for the 7B and 13B. What is the most likely reason?

A) The 7B and 13B models are too small to benefit from GQA  
B) For small models, the KV cache is a smaller fraction of total GPU memory; GQA's benefit is larger for big models where KV cache dominates  
C) GQA requires more parameters and would make 7B/13B models too large  
D) The 7B GQA implementation had bugs that were not fixed by publication time  

---

**Q5.** You load a LLaMA 3 model checkpoint but accidentally initialize the RoPE embeddings with `rope_theta=10000` instead of `500000`. What will you observe?

A) Training will fail immediately with a shape mismatch error  
B) The model will load fine but generation quality will degrade, especially for tokens at positions > 2048, because the positional encoding doesn't match what the model learned  
C) Generation will be slightly slower but functionally equivalent  
D) The model will automatically correct theta based on the checkpoint's config  

---

**Q6 (short answer).** LLaMA does not tie embedding and LM head weights (unlike GPT-2). What is the theoretical argument for tying weights? What is the counter-argument that justifies keeping them separate at LLaMA's scale?

---

**Q7 (short answer).** You are examining `LlamaAttention.forward()` in `modeling_llama.py`. You see that the KV cache is implemented as two separate tensors (past_key, past_value) stored in a `Cache` object. Describe the full sequence of operations in `LlamaAttention.forward()` from the input hidden states to the output, in 6–8 steps (one sentence each).

---

**Q8 (scenario).** You want to fine-tune LLaMA-3 8B on your PostgreSQL/TimescaleDB text-to-SQL dataset. You notice the HuggingFace config has `num_key_value_heads=8` and `num_attention_heads=32`. During a training run, you are OOM (out of memory) on a 24GB GPU with batch_size=4, max_length=1024. What parameter should you check first, and what is causing the OOM?
