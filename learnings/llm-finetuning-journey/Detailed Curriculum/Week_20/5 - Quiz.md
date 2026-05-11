# Week 20 Quiz — Pretraining Setup

## Multiple Choice

**Q1.** You are designing a 50M-parameter GPT model with d_model=768. Each transformer block has approximately how many parameters (ignoring layer norm)?

A) 2.4M (768² × 4)  
B) 7.1M (768² × 12)  
C) 9.4M (768² × 16)  
D) 3.5M (768² × 6)  

---

**Q2.** You train a BPE tokenizer with vocab_size=32000. What is the approximate initial cross-entropy loss of your model (random weights) on a next-token prediction task?

A) 1.0  
B) 3.5  
C) 10.4  
D) 32.0  

---

**Q3.** In the pre-LayerNorm (pre-LN) formulation used in your GPT implementation, Layer Normalization is applied:

A) After the attention output, before adding the residual  
B) Before the attention computation, to the input of each sub-layer  
C) Only at the final layer, before the LM head  
D) After each position-wise feed-forward layer  

---

**Q4.** Weight tying between the input embedding and the LM head projection:

A) Increases model performance but also increases parameter count  
B) Reduces parameter count by the size of the embedding matrix and often improves perplexity  
C) Is required for flash attention to work correctly  
D) Is only possible when vocab_size == d_model  

---

**Q5.** You are storing tokenized data as `uint16` in your `.bin` files. This limits your maximum token ID to:

A) 32,767  
B) 65,535  
C) 131,071  
D) 4,294,967,295  

---

## Short Answer

**Q6.** Explain the difference between context_len=1024 and block_size=1024 in your training pipeline. What happens at inference time if you want to generate text longer than context_len?

---

**Q7.** Your sanity check shows initial loss = 6.2 instead of the expected ~10.4. List 3 possible causes, ranked by likelihood.

---

**Q8.** You decide to use GPT-2's tokenizer (vocab_size=50257) instead of training your own (vocab_size=32000). How does this change your parameter count, and what is the trade-off?

---

## Scenario

**Q9.** It is the end of Week 20. You have:
- A 56M-parameter GPT model that passes all sanity checks
- `train.bin` with 80M tokens (you ran out of Colab session before getting to 100M)
- Initial loss = 10.3 (correct)
- Loss after 200 steps = 6.8

Your Week 21 plan is to train for 2B tokens on Colab Pro A100.

1. Given 80M tokens in train.bin, how many epochs do you need to train to reach 2B tokens? Is this acceptable?
2. Chinchilla says your 56M model should train on ~1.1B tokens (20 × 56M). You are targeting 2B tokens — is this over-training?
3. What is the estimated training time at 50K tokens/sec (typical Colab A100 throughput for a 56M model)?
4. Should you re-generate a larger `train.bin` before starting Week 21? What is your recommendation?
