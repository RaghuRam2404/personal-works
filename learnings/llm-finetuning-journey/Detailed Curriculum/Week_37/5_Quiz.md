# Week 37 Quiz — Building a Domain Fine-Tuning Dataset

## Multiple Choice

**Q1.** You want to build a PostgreSQL text-to-SQL dataset from public sources. sql-create-context has 78K examples. You filter out MySQL-specific syntax and invalid SQL, leaving 60K examples. You sample 10K for training. A colleague says: "We should use all 60K — more data is always better." What is the strongest argument for using only 10K?

A. 60K examples would take too long to tokenize and process  
B. Training on 60K examples for 2 epochs would cost more compute and may include lower-quality examples from the long tail of the dataset; marginal quality gains above 10K are likely small for a 7B model  
C. The peft library cannot handle datasets with more than 15K examples  
D. 60K examples would cause catastrophic forgetting of all pretraining knowledge

---

**Q2.** You generate synthetic SQL training examples using Claude. After generation, your dataset has 5K examples. You run `sqlparse.parse(sql)` on all of them. 200 examples return `None` for `get_type()`. What should you do with these 200 examples?

A. Include them — sqlparse is imperfect and these may still be valid SQL  
B. Manually review all 200 and keep valid ones  
C. Remove all 200 automatically — any SQL that sqlparse cannot classify is likely to confuse the model  
D. Replace the SQL with "NULL" and use the example anyway

---

**Q3.** Your dataset has 15K examples with this SQL type distribution: 70% simple SELECT, 15% JOIN, 10% GROUP BY, 5% other. What is the likely impact on your fine-tuned model?

A. The model will be excellent at all SQL types equally  
B. The model will be very strong at simple SELECT queries but weak at JOINs and complex queries, potentially worse than the base model on those specific types  
C. The imbalanced distribution is fine — the model learns from all examples regardless of frequency  
D. The model will refuse to generate JOINs because they are underrepresented

---

**Q4.** You are using a chat template for training examples. For each example, you format: system + user (schema + question) + assistant (SQL answer). Where should the training loss be computed?

A. Over all tokens in the sequence (system + user + assistant)  
B. Over only the assistant tokens (the SQL answer)  
C. Over only the user tokens (the question)  
D. Over a random 50% of all tokens

---

**Q5.** You have 30 hand-crafted TimescaleDB examples. For the training split, you include all 30. For your held-out test set (from Week 32), you have 5 TimescaleDB examples. Which concern is valid?

A. 30 training examples is too many — the model will memorize them  
B. The 5 test examples may be too few to reliably measure TimescaleDB performance — small sample size creates high variance in the evaluation metric  
C. TimescaleDB examples should not be in the training set at all  
D. The hand-crafted examples are likely incorrect — only LLM-generated examples should be used

---

## Short Answer

**Q6.** Explain in 3–4 sentences why using MySQL-syntax SQL examples in a PostgreSQL fine-tuning dataset is harmful, even if the model has already seen MySQL SQL in pretraining. Be specific about what tokens and patterns would be reinforced incorrectly.

---

**Q7.** You discover that 15% of your synthetic LLM-generated examples contain SQL that executes correctly but answers a different question than the one asked (e.g., the question asks for "total revenue by month" but the SQL returns "average revenue by year"). How would you detect this automatically, and what is the maximum acceptable rate of such errors?

---

**Q8.** Your dataset has 14,500 training examples and 500 validation examples. The training data was generated from 3 sources: sql-create-context (8K), synthetic LLM generation (5K), and hand-crafted TimescaleDB (500). Your validation set was randomly sampled from the combined dataset. What is a potential data contamination risk in this split strategy, and how would you prevent it?

---

## Scenario

**Q9.** You are 2 hours into building your 15K dataset. You have successfully filtered 8K examples from sql-create-context but your LLM API calls are failing (rate limits on the Claude API). You cannot generate the planned 5K synthetic examples. You have access to: (1) gretel/gretel-text-to-sql on HuggingFace, (2) wikisql on HuggingFace, (3) your own TimescaleDB schema. Describe your revised data collection strategy to reach 15K examples by end of week.
