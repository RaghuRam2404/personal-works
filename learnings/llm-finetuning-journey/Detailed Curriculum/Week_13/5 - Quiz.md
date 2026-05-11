# Week 13 Quiz — KV Cache, Inference Optimization, and Sampling

Calibration: mid-junior ML interview level.

---

**Q1.** During autoregressive inference without a KV cache, what is the computational complexity of generating T tokens from a context of length N?

A) O(N + T) — linear in total tokens  
B) O(T^2) — quadratic in generated tokens only  
C) O((N + T)^2) — quadratic in total sequence length  
D) O(N * T) — proportional to prompt times generated length  

---

**Q2.** You implement KV cache and notice that the output logits for a given prompt differ slightly between the cached and non-cached inference paths (difference of ~1e-3). What is the most likely cause?

A) The KV cache is not properly initialized and contains random values  
B) Floating point non-associativity: different operation orderings in the cached vs non-cached paths produce slightly different results  
C) The causal mask is being applied in the cached path, masking some past tokens  
D) The KV cache concatenation is along the wrong dimension  

---

**Q3.** Top-p (nucleus) sampling with `p=0.95` is preferred over top-k with `k=50` for most language generation tasks. Why?

A) Top-p is computationally cheaper than top-k  
B) Top-k always includes too many tokens; top-p always includes too few  
C) Top-p adapts the number of sampled tokens to the model's confidence: fewer candidates when the model is certain, more when it is uncertain  
D) Top-p produces lower perplexity on held-out data than top-k  

---

**Q4.** You apply `temperature=2.0` to your SQL generation model. The model was producing mostly valid SQL at `temperature=1.0`. What do you expect at `temperature=2.0`?

A) More valid SQL, because higher temperature increases exploration  
B) Less valid SQL, because the distribution is flatter and unlikely tokens (wrong SQL keywords) are sampled more often  
C) Identical output, because temperature only affects generation diversity, not token probabilities  
D) NaN in the logits, because dividing by 2.0 causes underflow  

---

**Q5.** A LLaMA-3 8B model (32 layers, 32 Q heads, 8 KV heads, d_k=128) generates 2048 tokens in FP16. How many bytes does the KV cache occupy?

A) 32 × 2 × 8 × 128 × 2048 × 2 = 2.1 GB  
B) 32 × 2 × 32 × 128 × 2048 × 2 = 8.6 GB  
C) 2 × 8 × 128 × 2048 × 2 = 4.3 MB  
D) 32 × 2 × 8 × 128 × 2048 = 1.1 GB  

---

**Q6 (short answer).** Explain why a causal mask is not needed when using a KV cache during single-token inference. Draw a simple diagram or describe the shapes involved.

---

**Q7 (short answer).** You are building a SQL generation system. Your model sometimes generates `SELECT SELECT * FROM` (repeating keywords). List two specific decoding strategies that would reduce this repetition, explain the mechanism behind each, and state the trade-off.

---

**Q8 (scenario).** You benchmark your KV-cached nanoGPT and find it is only 1.3x faster than non-cached inference (generating 200 tokens from a 50-token prompt). You expected ≥5x. What are 3 things you should check in your implementation?
