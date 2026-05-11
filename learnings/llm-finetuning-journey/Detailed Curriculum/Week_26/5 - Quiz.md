# Week 26 Quiz — Building the Domain Dataset

## Multiple Choice

**Q1.** You are deduplicating your dataset using MinHash with Jaccard threshold=0.7. A question "Find all orders from last week" and "List all orders placed in the previous 7 days" have a Jaccard similarity of 0.65. MinHash will:

A) Mark them as duplicates and remove one  
B) Keep both — they are below the 0.7 threshold  
C) Mark them as duplicates based on semantic similarity, not Jaccard  
D) Automatically merge them into a single example  

---

**Q2.** When publishing your dataset to HuggingFace Hub, setting `private=True` means:

A) The dataset is encrypted and cannot be downloaded  
B) Only you (and anyone you explicitly grant access to) can see and download the dataset  
C) The dataset cannot be used for commercial fine-tuning  
D) The dataset is automatically deleted after 30 days  

---

**Q3.** You want to verify that your hand-written SQL examples are semantically correct (not just syntactically valid). The strongest verification method is:

A) Running `sqlglot.parse()` on each example  
B) Using GPT-4 to review each SQL query for correctness  
C) Executing each query against a real PostgreSQL database with realistic test data and checking the returned rows  
D) Running BLEU score between your query and a reference query  

---

**Q4.** Your Tier 3 self-instruct generation produces many examples with the exact same TimescaleDB `time_bucket('1 hour', ts)` pattern. This is a problem because:

A) The model will generate this pattern for every query, even when a different time interval is appropriate  
B) `time_bucket` is only valid for minute intervals, not hours  
C) MinHash will remove all of these as duplicates regardless of threshold  
D) Repetitive patterns in training data always cause catastrophic forgetting  

---

**Q5.** Your v1 dataset has 5,000 examples but only 20 use `LATERAL` joins. To fine-tune a model that can reliably generate `LATERAL` joins, you should:

A) Increase the weight of the 20 LATERAL examples by repeating them 50× in the training data  
B) Accept that LATERAL joins will be rare in the model's output — it learned from the training distribution  
C) Both A and B are valid approaches depending on how critical LATERAL joins are for your deployment  
D) Remove LATERAL join examples entirely since they are too rare to learn from  

---

## Short Answer

**Q6.** Explain why cross-deduplication (deduplicating Tier 3 against Tier 1+2 combined) is important, and what failure mode occurs if you skip it.

---

**Q7.** Your dataset statistics show: average SQL length = 18 tokens, and 40% of examples are "SELECT col FROM table" (single table, no joins, no WHERE clause). What is the problem, and what 2 approaches would you use to fix it?

---

**Q8.** Write a 3-question evaluation protocol for your v1 dataset that you would run before using it for fine-tuning. For each question, describe what a "pass" looks like.

---

## Scenario

**Q9.** After publishing postgres-sql-v1, you fine-tune Qwen2.5-Coder-7B and evaluate it. The model:
- Achieves 85% execution accuracy on your validation split
- But only 45% execution accuracy on 50 hand-written test questions you created after building the dataset (held out, never in training)

1. What does this gap (85% vs 45%) tell you about your training dataset?
2. You look at the failing cases on the held-out test set. 30% of failures involve TimescaleDB's `time_bucket_gapfill()` function, which appears in only 3 of your 5,000 training examples. What does this tell you, and how do you fix it?
3. Another 20% of failures are correct SQL but wrong: the model uses `SUM(amount)` instead of `SUM(quantity * unit_price)` because the question says "total value" but the training examples all compute totals as single-column aggregations. What type of error is this, and what training data would fix it?
4. Outline your plan for postgres-sql-v2 based on this analysis.
