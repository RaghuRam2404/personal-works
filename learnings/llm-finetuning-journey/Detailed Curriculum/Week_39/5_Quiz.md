# Week 39 Quiz — Execution-Based Evaluation

## Multiple Choice

**Q1.** You run exact match evaluation on your Week 38 model and get 52%. You then run execution-based eval on the same 100 examples and get 71%. Which explanation best accounts for this gap?

A. The model memorized the training set and is overfitting, inflating execution numbers  
B. Multiple valid SQL phrasings return the same result set, so exact match undercounts correct predictions  
C. Execution-based eval is easier because it ignores column names and only checks row counts  
D. Your exact match implementation has a bug that discards whitespace differences

---

**Q2.** Your eval harness creates a fresh schema per test example using `DROP SCHEMA IF EXISTS eval_tmp CASCADE` followed by `CREATE SCHEMA eval_tmp`. A colleague suggests using a single shared schema and only dropping/recreating tables between tests. What is the main risk of the colleague's approach?

A. It is slower because DROP TABLE is more expensive than DROP SCHEMA  
B. Side effects from one test example (leftover rows, sequences, views) can bleed into the next test  
C. PostgreSQL does not support dropping individual tables inside a named schema  
D. It forces you to reconnect to the database on every test example

---

**Q3.** You set `SET statement_timeout = 5000` in your harness before executing model-generated SQL. A test example generates `SELECT * FROM orders JOIN order_items ON orders.id = order_items.order_id` against a table with 500K rows. What is the most likely outcome?

A. The query executes successfully because 500K rows is well within Postgres capacity  
B. The harness rejects it immediately because `SELECT *` is a non-SELECT pattern  
C. The query may exceed the timeout if no index exists on the join column, returning a timeout error  
D. The statement timeout applies only to INSERT and UPDATE statements, not SELECT

---

**Q4.** When comparing result sets in your harness, you normalize NULLs by replacing them with the string `"NULL"`. A colleague argues this is wrong — two rows `(1, NULL)` and `(1, "NULL")` would falsely compare as equal. Which response is most accurate?

A. The colleague is wrong: NULL and "NULL" can never appear in the same column in a typed schema  
B. The colleague is correct; you should use a sentinel object (e.g., a Python `object()`) that is unique per NULL  
C. The colleague is partially right; the issue only matters if the schema has text columns that might legitimately contain the string "NULL" — document this limitation  
D. NULL normalization is unnecessary because Python's `==` handles NULL equality correctly

---

**Q5.** Your model's execution success rate is 94% but execution correctness is only 58%. Your base model's execution success rate is 78% and correctness is 29%. Which intervention is most likely to close the gap between success and correctness in your fine-tuned model?

A. Add more data — execution errors (6%) are the bottleneck, so fix syntax first  
B. Analyze the 36 semantically wrong examples, identify the dominant failure mode (wrong WHERE, wrong aggregation), and add targeted training examples for that category  
C. Increase LoRA rank from 16 to 64 — the model lacks capacity for semantic understanding  
D. Switch from QLoRA to full fine-tuning — quantization degrades semantic precision

---

## Short Answer

**Q6.** Explain in 3–4 sentences why execution correctness is a better metric than exact match for evaluating a text-to-SQL model. Give one concrete example where exact match gives the wrong verdict.

---

**Q7.** Your harness rejects all non-SELECT statements. A test example's expected SQL is `WITH cte AS (SELECT ...) SELECT * FROM cte`. Describe what happens when the model generates this exact query, and whether the `is_safe_sql` function handles it correctly. If not, how do you fix it?

---

**Q8.** You run your harness and find that 12 of 40 failures fall into the "wrong filter" category — the model generates the right table and columns but a subtly wrong WHERE clause. List 3 specific interventions you would make to the training data to address this failure mode specifically.

---

## Scenario

**Q9.** You are presenting Week 39 results to a technical lead. Your numbers:

- Base model exec correctness: 28%
- Week 38 fine-tuned model exec correctness: 67%
- Exact match (Week 38): 49%
- Execution success rate (Week 38): 93%

The tech lead says: "67% is not good enough for production. What are your next 3 steps to get to 85%?"

Write a concrete response covering: what data interventions you would make, whether you would change the training approach, and what role Phase 5 (GRPO) plays in pushing beyond 75%.
