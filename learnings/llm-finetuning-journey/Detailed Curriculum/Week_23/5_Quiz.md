# Week 23 Quiz — LM Evaluation

## Multiple Choice

**Q1.** lm-evaluation-harness scores most multiple-choice benchmarks (MMLU, HellaSwag, ARC) using:

A) The model generating "A", "B", "C", or "D" as the next token  
B) Log-likelihood scoring: choosing the option with the highest log P(option | question)  
C) Embedding similarity between the question and each option  
D) A separate classifier fine-tuned to pick the best option  

---

**Q2.** Your 50M model scores 27% on MMLU (0-shot). The random baseline is 25%. This result means:

A) Your model has failed and needs to be retrained  
B) Your model is performing near random, which is expected for 50M parameters on a factual knowledge benchmark  
C) Your model is cheating by memorizing the MMLU test set  
D) Your evaluation setup is broken — a model cannot score near random on a multiple-choice task  

---

**Q3.** HellaSwag scores are computed by length-normalizing log-likelihoods. Why?

A) To ensure that all models use the same computational budget  
B) To prevent bias toward options that are shorter (and thus have higher absolute log-probability)  
C) To convert perplexity to bits-per-character for cross-tokenizer comparison  
D) To penalize models that rely on the first token of each option  

---

**Q4.** A model achieves perplexity 12 on Wikipedia but scores 30% on MMLU (near random). This combination suggests:

A) An error in the evaluation setup — low perplexity always implies high MMLU accuracy  
B) The model has learned Wikipedia's surface statistics (word co-occurrence patterns) but not the factual relationships tested by MMLU  
C) The model's tokenizer has a vocabulary mismatch with the MMLU test set  
D) MMLU scores are not meaningful for models with perplexity below 20  

---

**Q5.** 5-shot evaluation (compared to 0-shot) on a benchmark:

A) Trains the model on 5 examples before evaluation  
B) Prepends 5 labeled example pairs to each test question as context, so the model can learn the task format  
C) Averages model scores over 5 random seeds  
D) Uses 5 different prompts per question and averages the scores  

---

## Short Answer

**Q6.** Explain why benchmark contamination is a serious concern when evaluating open-weight LLMs, and describe one method for detecting it.

---

**Q7.** You evaluate your 50M model and GPT-2 (117M) on ARC-Easy. GPT-2 scores 43%, your model scores 30%. List 3 reasons that explain this gap, beyond just parameter count.

---

**Q8.** Design a 3-benchmark evaluation suite specifically for a PostgreSQL text-to-SQL model. For each benchmark, specify: the metric, the dataset source, and what it measures that the others do not.

---

## Scenario

**Q9.** Your company wants to evaluate a fine-tuned Qwen2.5-Coder-7B on your PostgreSQL text-to-SQL task. A colleague says: "Just run MMLU and HellaSwag — if it scores 65% on MMLU, it's smart enough."

1. Explain why MMLU and HellaSwag are insufficient evaluations for your specific task.
2. What is the primary metric you would use for text-to-SQL evaluation, and why?
3. How would you construct a held-out evaluation set that avoids data contamination?
4. You run your model on Spider (a standard text-to-SQL benchmark) and it scores 82%. Your colleague is impressed. List 2 reasons why this might not translate to good performance on your actual PostgreSQL/TimescaleDB benchmark.
