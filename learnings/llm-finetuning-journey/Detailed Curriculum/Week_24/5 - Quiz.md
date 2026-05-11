# Week 24 Quiz — SOTA Pretraining Recipes

## Multiple Choice

**Q1.** DeepSeek-V3 is described as a 671B parameter model but activates only 37B parameters per token. This is possible because:

A) The model uses sparse self-attention that skips 90% of attention heads  
B) The FFN layers use a Mixture of Experts architecture where only a subset of experts is activated for each token  
C) The model is quantized to 4-bit, reducing effective parameter count  
D) 634B parameters are in the embedding layer, which is not activated during forward pass  

---

**Q2.** Qwen2.5-Coder-7B was trained on approximately 5.5T tokens with a code-first data distribution. Compared to Llama 3-8B (15T general tokens), this suggests that Qwen2.5-Coder:

A) Has lower general language understanding than Llama 3-8B due to fewer total tokens  
B) Will likely outperform Llama 3-8B on code generation tasks but may underperform on factual trivia  
C) Is larger than Llama 3-8B and therefore always outperforms it  
D) Cannot be fine-tuned on domain-specific data because it was already trained on code  

---

**Q3.** Multi-Head Latent Attention (MLA) in DeepSeek-V3 primarily addresses which bottleneck?

A) GPU compute limitations for large batch sizes  
B) KV cache memory consumption during long-context inference  
C) Gradient vanishing in deep transformers during pretraining  
D) Tokenization efficiency for multilingual text  

---

**Q4.** The fill-in-the-middle (FIM) training objective used in DeepSeek-Coder and Qwen2.5-Coder:

A) Trains the model to predict the end of a document given only the beginning  
B) Trains the model to infill a missing middle section given both the prefix and suffix  
C) Is a data augmentation technique that rotates sentence order  
D) Is only used during fine-tuning, not pretraining  

---

**Q5.** Llama 3-8B uses a vocabulary of 128,256 tokens while Qwen2.5-Coder uses 151,936 tokens. A larger vocabulary primarily means:

A) The model requires less training data to converge  
B) The embedding and LM head matrices are larger, and each token carries more information (fewer tokens per word)  
C) The model is slower at inference because there are more logits to compute  
D) The model cannot be fine-tuned with QLoRA  

---

## Short Answer

**Q6.** Explain Grouped Query Attention (GQA) in 3 sentences, and explain why it reduces inference memory without affecting model quality significantly.

---

**Q7.** Llama 3-8B was trained on 15T tokens — far beyond Chinchilla-optimal. You calculated in Week 17 that Chinchilla optimal for 8B params is ~160B tokens. Llama 3 trained on 93× more. Give 2 reasons why this was the right decision for Meta.

---

**Q8.** If you were advising a startup that wants to build a SQL assistant for MySQL (not PostgreSQL), which of the 5 models would you choose as a base, and what data would you include in fine-tuning that you would not need for a PostgreSQL assistant?

---

## Scenario

**Q9.** You are building a PostgreSQL/TimescaleDB text-to-SQL assistant using Qwen2.5-Coder-7B as your base.

1. What SQL-specific knowledge does Qwen2.5-Coder-7B already have that reduces your fine-tuning burden? Be specific.
2. What PostgreSQL/TimescaleDB knowledge is it unlikely to have, and why?
3. You have a budget of $150 for fine-tuning compute (RunPod A100 at $1.50/hr). Approximately how many hours of fine-tuning can you afford, and what fine-tuning approach would you use (full SFT vs. LoRA vs. QLoRA)?
4. What would the first 3 items in your fine-tuning dataset look like (give example question + SQL pairs)?
