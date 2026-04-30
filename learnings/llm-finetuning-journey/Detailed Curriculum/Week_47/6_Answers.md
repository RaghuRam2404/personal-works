# Week 47 Quiz Answers

## Q1. Answer: C

**Answer:** C — 0.0, blocked by anti-hack guard.

**Why:** The information_schema guard must check for `INFORMATION_SCHEMA` in the SQL (case-insensitive) and return 0.0 before the execution step. `SELECT table_name FROM information_schema.tables` is a valid SQL query that always executes successfully, but it produces meaningless output for a Text-to-SQL system that should be answering business questions about your domain data. Without this guard, GRPO will learn to generate system queries because they reliably score 0.2.

---

## Q2. Answer: B

**Answer:** B — The model is generating text without extractable SQL.

**Why:** If the model (v2-dpo) was never properly trained to output SQL in a consistent format (with code fences or as bare SELECT statements), `extract_sql()` will return None for most outputs. This is a model format alignment problem, not a reward function bug. Fix: check 10 raw model outputs before blaming the reward function. If the model generates prose explanations without any SQL, add a SFT step specifically on format alignment (System: "Output only SQL. Start with SELECT or WITH.").

**Why others are wrong:**
- A: The information_schema guard would only block queries that contain those specific strings — it would not cause 80% failure if the model generates domain SQL.
- C: Connection failures would raise exceptions, which your reward function catches and returns 0.0 — identical symptom, but check the logs for connection error messages.
- D: The extract_sql() in the solution handles both `sql` and `SQL` case-insensitively — but this is worth checking if D seems plausible.

---

## Q3. Answer: A

**Answer:** A — The model learned to optimize the reasoning bonus independently of SQL quality.

**Why:** The model discovered that generating 200+ characters of text, regardless of content quality, always adds +0.5 to the reward. Since this bonus is independent of SQL correctness, the model can reliably earn it by generating any text. This is reward hacking at the bonus level. A reasoning bonus should never exceed 10% of the maximum reward (so +0.05 if max is 0.5, not +0.5). And it should be conditional on the SQL also being correct: `reasoning_bonus = 0.05 if len(preamble) > 100 AND reward >= 0.5 else 0.0`.

---

## Q4. Answer: A

**Answer:** A — Queries that return duplicate rows are collapsed by set comparison.

**Why:** `SELECT customer_id FROM orders WHERE date > '2024-01-01'` might return 100 rows with many duplicate customer_id values. If the expected output also has 100 rows with duplicates, `set()` equality would compare unique values only — a query that returns different duplicates might pass the set test while being semantically wrong. Fix: use `sorted(actual_rows) == sorted(expected_rows)` (list equality with sorting) instead of set equality to preserve duplicates. Or use multiset comparison: check that row counts match AND that every unique (value, count) pair matches.

---

## Q5. Multi-Requirement Reward Function

```
Level 0.0: SQL extraction fails, OR contains system schema, OR not SELECT/WITH
Level 0.0: SQL has syntax error
Level 0.1: SQL has runtime error (wrong table/column) — table structure awareness but not complete
Level 0.2: SQL executes and returns rows, but row count does not match expected
           (partial credit: model has the right idea but wrong conditions)
Level 0.5: SQL executes and row count matches expected exactly
           (structural correctness — right count, may have wrong values)
Level 1.0: SQL executes AND sorted(rows) == sorted(expected)
           (exact semantic correctness)

Performance requirement: enforce 500ms timeout with SET statement_timeout = 500;
Anti-hacks: 
  - Block INFORMATION_SCHEMA, PG_CATALOG
  - Return 0.0 if row_count > 5 * len(expected) + 10
  - Return 0.0 if no FROM clause (catches SELECT NULL, SELECT 1)
```

---

## Q6. Zero-Advantage Prompts

**Statistical consequence:** For prompts where all K completions have equal rewards, std = 0, advantages are all 0, and the gradient is zero. With 40% of prompts producing zero gradients, you are effectively training on only 60% of your prompt set. The effective batch size is reduced, training is slower, and the model may overfit to the 60% of prompts that do produce gradients (particularly easy or medium difficulty prompts).

**Fix 1 (prompt selection):** Implement "active difficulty selection" — before each training batch, filter to prompts where the current model succeeds 20–80% of the time. Measure per-prompt success rate on a running window of 20 completions, and only sample prompts in the 20–80% zone for GRPO training. This ensures most training steps produce varied group rewards. Easy prompts (>80% success) and hard prompts (<20% success) are moved to a "graduation queue."

**Fix 2 (shaped reward):** Replace binary 0/1 rewards with the multi-level hierarchy described in the curriculum. With 5 reward levels, it is much harder for all K completions to tie exactly. Even if all 8 execute correctly, they may differ in row count exactness or reasoning quality, producing non-zero advantages.

---

## Q7. Scenario — Reward Hacking or Capability Limitation?

**Diagnosis:** The observed behavior (model avoids JOINs, generates `SELECT * ... GROUP BY`) is a combination of reward shaping failure and model capability limitation:

- **Reward shaping failure:** The 0.5 reward level (correct row count) is being exploited. `SELECT * FROM orders GROUP BY customer_id` returns one row per customer — if the expected row count happens to match, this scores 0.5 without using JOIN or selecting the right columns. The model has learned a "shortcut": group by primary key instead of joining to get revenue data.
- **Model capability limitation:** The model may have catastrophically forgotten how to generate JOINs during DPO or GRPO training if JOIN examples were underrepresented.

**Reward change:** Upgrade the 0.5 level. Instead of "correct row count," require "correct column names in result." If the expected output has columns `[customer_id, total_revenue]` but the actual has `[customer_id, product_id, order_date, ...]`, score it 0.1 even if the row count matches. This blocks the `SELECT *` shortcut.

```python
# Check column names if expected_columns is available
if actual_column_names != expected_column_names:
    return 0.2  # Wrong schema, even if row count matches
```

**Training data change:** Add 200 prompts specifically requiring JOINs with expected outputs that cannot be produced by single-table queries. Use these prompts exclusively in GRPO training for 200 steps to force the model to re-learn JOIN patterns. After 200 steps, return to the full prompt distribution.
