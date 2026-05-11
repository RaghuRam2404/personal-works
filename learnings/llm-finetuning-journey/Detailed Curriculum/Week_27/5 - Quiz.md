# Week 27 Quiz — Phase 3 Gate Assessment

This quiz covers the full Phase 3 (Weeks 17–27). It is a comprehensive assessment, not a weekly quiz. Treat it as a practice technical interview for a mid-level ML engineer role.

## Multiple Choice

**Q1.** Given a compute budget of $30 at $1.50/hr with A100 (35% MFU), the Chinchilla-optimal model size and token count is approximately:

A) N=500M params, D=10B tokens  
B) N=200M params, D=4B tokens  
C) N=50M params, D=1B tokens  
D) N=7B params, D=140B tokens  

---

**Q2.** Your 50M model's validation perplexity is 35 on FineWeb-Edu validation data. Which of the following is NOT a likely cause?

A) Insufficient training (only 500M tokens instead of 2B)  
B) Tokenizer mismatch between training and evaluation  
C) Your model has too many parameters for this dataset  
D) Validation set includes documents from a domain not well-covered in training  

---

**Q3.** When computing log-likelihood scores in `lm-evaluation-harness` for HellaSwag, the scores are length-normalized. This means:

A) Scores are divided by the number of sentences in the option  
B) Scores are divided by the number of tokens in the option to prevent bias toward shorter options  
C) Scores are normalized to a 0–1 range by dividing by the maximum possible log-likelihood  
D) Scores are divided by the total number of evaluation examples  

---

**Q4.** You want to fine-tune Qwen2.5-Coder-7B on your postgres-sql-v1 dataset. During training, the loss should be computed on:

A) All tokens in the conversation  
B) Only the system prompt tokens  
C) Only the assistant (SQL) turn tokens  
D) The user question tokens only  

---

**Q5.** DeepSeek-V3's Mixture of Experts architecture has 671B total parameters but 37B active parameters per token. For memory purposes during inference, you need GPU VRAM for approximately:

A) 671B × 2 bytes = 1.34TB (full model must be in GPU memory)  
B) 37B × 2 bytes = 74GB (only active parameters during forward pass)  
C) All 671B parameters must be loaded, but only 37B are used per step  
D) Neither — MoE models run on CPU only  

---

## Short Answer

**Q6.** Describe the complete pipeline for building your postgres-sql-v1 dataset in 5 steps. For each step, name the tool or technique used.

---

**Q7.** Your 50M model perplexity is 35 on FineWeb val. List 3 likely causes, ranked by likelihood. For each, describe one diagnostic check.

---

**Q8.** Explain ZeRO Stage 2 in 4 sentences: what does it shard, what remains full, and how does it differ from DDP and ZeRO-3?

---

## Scenario

**Q9 (Comprehensive scenario — 15 min):**

You are a senior engineer reviewing a junior's Phase 3 work. They show you:
- A 50M model trained for 3B tokens (1 epoch on a 3B-token dataset)
- Val loss = 3.1
- Perplexity = 22.2
- HellaSwag score = 32% (GPT-2-small = 31.6%)
- ARC-Easy = 44% (GPT-2-small = 43.2%)
- MMLU = 27% (near random)
- Dataset: 3,800 training examples in ChatML format, 97% SQL validity rate, 65 hand-written examples

Answer the following:
1. Is the training run acceptable for Phase 3 goals? Why or why not?
2. Is the 3,800-example dataset acceptable for Phase 3 goals?
3. The junior says "my HellaSwag and ARC scores are close to GPT-2 but my model is only 50M vs 117M — this proves my training was better." Evaluate this claim critically.
4. The junior wants to immediately fine-tune this 50M model on the SQL dataset. Give your recommendation and explain the reasoning.
5. The junior's MMLU score of 27% is "barely above random." Should they be concerned? Why or why not?
