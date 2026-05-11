# Week 22 Quiz — Evaluating Language Models

## Multiple Choice

**Q1.** Your 50M model achieves a validation cross-entropy loss of 3.5 on FineWeb-Edu val. The perplexity is:

A) 3.5  
B) 16.4  
C) 33.1  
D) 101.2  

---

**Q2.** You compute perplexity by: (a) computing loss for each of 200 batches, (b) averaging the 200 loss values, (c) taking `exp()` of the average. A colleague says you should instead compute `mean(exp(per_batch_loss))`. Which approach is correct?

A) Your colleague is correct — averaging perplexities gives the geometric mean which is more robust  
B) Your approach is correct — average the log-likelihoods (losses), then exponentiate  
C) Both approaches give identical results by the law of large numbers  
D) Neither is correct — perplexity should be computed on the full dataset in a single forward pass  

---

**Q3.** Setting temperature=0.1 during text generation (very low temperature) will typically cause:

A) More diverse and creative output  
B) Faster generation because fewer candidates are sampled  
C) Repetitive, deterministic output that converges to greedy decoding  
D) Higher perplexity than temperature=1.0  

---

**Q4.** Your 50M model's perplexity on a SQL snippet is 58, while its perplexity on general English web text is 32. This means:

A) The model is better at SQL than at English  
B) The model assigns lower probability (is more surprised) by SQL tokens than by English tokens  
C) SQL is a harder language to model than English, regardless of training data  
D) There is a data contamination issue — SQL was in the validation set  

---

**Q5.** Top-k sampling (k=50) at each generation step:

A) Limits the vocabulary to the 50 most common tokens in the training data  
B) Selects only from the 50 highest-probability tokens at the current position, setting all others to -infinity  
C) Guarantees that the top-1 (most probable) token is always selected  
D) Requires beam search to function correctly  

---

## Short Answer

**Q6.** Your model's validation perplexity is 35 on FineWeb-Edu validation data. List 3 likely causes if this is higher than you expected given your training progress (val loss was 3.2 at the end of training, implying expected PPL ≈ 24.5).

---

**Q7.** Explain why perplexity computed with a 32K-vocabulary tokenizer is not directly comparable to GPT-2's perplexity (computed with its 50K-vocabulary tokenizer) on the same text.

---

**Q8.** After generating 10 samples, you observe: sample lengths are fine (100–150 tokens), but all samples share the same 3 sentence patterns repeated with different nouns. What generation parameter change would most likely fix this, and why?

---

## Scenario

**Q9.** You present your 50M model's evaluation to a team. They ask:

1. Your model achieves perplexity 32 on general English. GPT-2-small (117M params, 40B tokens) achieves perplexity ~18 on similar data. Explain what accounts for this 1.78× gap in perplexity terms.
2. Someone suggests fine-tuning your 50M model on 5K PostgreSQL SQL examples. List 2 reasons this is a poor choice compared to fine-tuning Qwen2.5-Coder-7B on the same data.
3. The team asks you to demonstrate your model's SQL ability. You generate 5 SQL samples using prompt "SELECT * FROM orders WHERE" and all 5 are syntactically invalid. Is this a problem with your model or with your generation settings? How would you diagnose?
4. What would you train differently if you had to redo this 50M pretraining run?
