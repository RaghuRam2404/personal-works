# Week 47 Quiz — GRPO Reward Function Design

## Multiple Choice

**Q1.** Your SQL reward function returns 0.2 for "SQL executes and returns rows (unverified)." A reward hacker generates: `SELECT table_name FROM information_schema.tables` — this always executes and returns many rows. What is the correct reward your function should return?

A) 0.2 — it executes correctly per the definition  
B) 0.5 — it returns the correct number of rows for some expected output  
C) 0.0 — your anti-hack guard should block system schema queries  
D) 0.1 — it executes but returns wrong data type  

---

**Q2.** In GRPO with K=8, you observe that for 80% of training prompts, all 8 completions receive reward = 0.0 (no SQL extracted). What is the most likely cause?

A) Your reward function has a bug — it is blocking valid SQL with the information_schema guard  
B) Your model is generating text that does not contain any extractable SQL (has not learned the output format)  
C) The Postgres database is rejecting connections, causing all executions to fail  
D) Your extract_sql() function requires ```sql fences but the model uses ```SQL (uppercase)  

---

**Q3.** You add a "reasoning bonus" of +0.5 to any completion that contains more than 200 characters before the SQL. After 100 GRPO steps, the model starts generating completions like: "This is a really detailed explanation that is very very long and repeats itself many times..." before the SQL. What reward hacking occurred and why?

A) The model learned to optimize the reasoning bonus independently of SQL quality  
B) The model is reducing KL divergence by generating more tokens (spreading probability mass)  
C) The LLM's context window filled up, causing it to repeat tokens  
D) This is expected — longer reasoning chains produce better SQL  

---

**Q4.** Your reward function correctly identifies exact-match SQL (reward=1.0) using `set(rows_actual) == set(rows_expected)`. What is one class of SQL correctness problems this misses?

A) Queries that return duplicate rows (duplicate rows are collapsed by set comparison)  
B) Queries that reference system tables (handled by anti-hack guard)  
C) Queries that time out (handled by timeout)  
D) Queries that return zero rows (handled by empty result check)  

---

## Short Answer

**Q5.** Design a reward function for SQL with these requirements: (1) binary execution success is not enough — the query must return the right number of rows, (2) partial credit for correct table structure even if values are wrong, (3) no credit for information_schema queries, (4) the function must run in under 500ms per query. Write pseudocode or a description of the function with all levels labeled.

---

**Q6.** You discover that 40% of your training prompts always produce all-zero-advantage groups (either all 8 completions succeed or all 8 fail). Explain the statistical consequence for GRPO training and propose two dataset-level fixes.

---

## Deep Scenario

**Q7.** You have deployed your GRPO reward function and after 500 training steps, you observe:
- Average reward has increased from 0.18 to 0.42
- The reward distribution at step 500: 15% at 0.0, 30% at 0.2, 45% at 0.5, 10% at 1.0
- Manual inspection shows the model often generates `SELECT * FROM orders GROUP BY customer_id` for prompts asking for "revenue by customer" — this executes but returns wrong columns
- The model rarely generates JOIN queries anymore — 90% of completions use single-table queries

Diagnose whether this is reward hacking, reward shaping failure, or model capability limitation. Then propose one reward change and one training data change to fix it.
