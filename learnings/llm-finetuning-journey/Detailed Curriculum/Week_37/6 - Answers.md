# Week 37 Quiz Answers

## Q1 — Answer: B

**Answer:** B. Marginal quality gains above 10K are small; long tail includes lower-quality examples.

**Why:** For a 7B model with 7B parameters of pretrained knowledge, the marginal benefit of each additional training example decreases as dataset size grows. Going from 1K to 10K examples yields large quality improvements. Going from 10K to 60K yields smaller improvements at significantly higher compute cost (6× longer training per epoch). Additionally, the long tail of any scraped dataset (examples 50K–60K) tends to be lower quality, more repetitive, or more formulaic — including these can hurt diversity and waste training capacity. 10K carefully curated examples often outperforms 60K uncurated examples.

---

## Q2 — Answer: C

**Answer:** C. Remove all 200 automatically.

**Why:** Any SQL that `sqlparse` cannot classify is either invalid SQL, a malformed statement, or contains unusual syntax that the tokenizer cannot parse. Including such examples in training teaches the model to generate syntactically ambiguous SQL, which is strictly worse for your use case. The 200 examples represent only 4% of your synthetic dataset — the loss is acceptable. Generate replacement examples rather than trying to manually fix 200 problematic outputs.

---

## Q3 — Answer: B

**Answer:** B. Strong at simple SELECT, weak at JOINs and complex queries.

**Why:** SFT reinforces the patterns it sees most frequently. A model trained on 70% simple SELECT queries will learn to generate simple SELECT statements very fluently and will be less practiced at JOINs, GROUP BY, and complex patterns. On held-out complex queries, the model may default to simple SELECT structure even when a JOIN is needed. Fix: resample the dataset to a more balanced distribution — target 30–40% SELECT, 30–35% JOIN, 20–25% GROUP BY, 10–15% complex — by upsampling rare types or downsampling common ones.

---

## Q4 — Answer: B

**Answer:** B. Only the assistant tokens (the SQL answer).

**Why:** SFTTrainer applies input masking: the system and user tokens are part of the forward pass context (the model attends to them) but their positions are excluded from the cross-entropy loss via label=-100. The model is not penalized for failing to predict the prompt — only for failing to predict the correct SQL answer. This is standard SFT behavior and was the topic of Week 29's curriculum.

---

## Q5 — Answer: B

**Answer:** B. 5 test examples is too few for reliable TimescaleDB evaluation.

**Why:** With 5 test examples, your evaluation metric for TimescaleDB performance is (0 to 5)/5 = 0%, 20%, 40%, 60%, 80%, or 100% — extremely coarse-grained. A single incorrect prediction changes the metric by 20 percentage points. This makes it impossible to reliably measure improvement from one training run to another. Fix: add at least 20–30 TimescaleDB examples to your held-out test set, ensuring they are not in the training data.

---

## Q6 — Short Answer

MySQL and PostgreSQL differ in specific token-level syntax patterns: MySQL uses backtick identifiers (`` `column` ``), `AUTO_INCREMENT`, `TINYINT(1)` for booleans, `LIMIT x, y` instead of `LIMIT x OFFSET y`, and `IFNULL` instead of `COALESCE`. If you train the model on these patterns, the model's weights will assign high probability to generating backticks and MySQL keywords even when the context calls for PostgreSQL syntax.

For example, after training on MySQL examples, a fine-tuned model might generate `SELECT * FROM \`employees\`` (with backticks) instead of `SELECT * FROM employees` for a PostgreSQL query. Since backtick identifiers are invalid in PostgreSQL, this causes immediate execution failure. The reinforced MySQL patterns are particularly insidious because the model's pretraining may have seen much more MySQL than PostgreSQL, and SFT on MySQL examples further entrenches these patterns.

---

## Q7 — Short Answer

Automated detection approaches:

1. **Schema-question alignment check:** For each generated example, verify that the column names referenced in the SQL appear in the CREATE TABLE schema. If the SQL references columns not in the schema, it is likely a generation error. Implement as: extract column references from SQL via sqlparse, check against schema column names.

2. **LLM-as-judge:** Send (schema, question, sql) to a separate LLM (Claude or GPT-4o-mini) with the prompt: "Does this SQL correctly answer the question given the schema? Answer yes or no." This is expensive but more reliable. A sample of 200 random examples with LLM judgment is sufficient to estimate error rate.

Acceptable error rate: < 5% for SQL-question misalignment. At 5K synthetic examples with 5% error rate = 250 misaligned examples in training — likely acceptable as noise. Above 10% starts significantly degrading model quality on complex queries.

---

## Q8 — Short Answer

The contamination risk: examples from the held-out test set (100 examples built in Week 32) may have similar schemas or questions to examples in your training set, especially if both come from the same public source (sql-create-context). If a test example uses the same schema as a training example but a different question, the model may have partially "memorized" that schema during training, inflating test performance.

Prevention: after building your 15K dataset, run a cross-dataset deduplication check against `held_out_test.json` using schema-level hashing (hash the CREATE TABLE statements). Remove any training example whose schema matches a held-out test schema exactly. This ensures the model is being tested on truly novel schemas, not schemas it has seen during training.

---

## Q9 — Scenario Answer

Revised strategy without LLM API access:

**Source 1 — sql-create-context:** Already have 8K. Increase sample to 10K by relaxing filters slightly (allow some edge cases that sqlparse flags but are likely valid).

**Source 2 — gretel/gretel-text-to-sql:** `load_dataset("gretel/gretel-text-to-sql")`. This dataset contains synthetically generated examples with diverse schemas and PostgreSQL-compatible syntax. Sample 3K after the same MySQL filter and sqlparse validation.

**Source 3 — wikisql:** Contains 80K simple single-table queries with natural language questions. These are lower complexity but add diversity. Sample 1.5K after filtering (wikisql uses slightly different format — convert to schema + question + sql).

**Source 4 — Hand-crafted TimescaleDB:** Already planned 30–100 examples; complete these as planned.

**Total:** 10K (sql-create-context) + 3K (gretel) + 1.5K (wikisql) + 0.1K (timescale) ≈ 14.6K. Round up to 15K by sampling slightly more from gretel or relaxing filters on sql-create-context.

The quality of this dataset will be slightly lower than the API-generated version (less PostgreSQL-specific coverage), but the training in Week 38 will still be valuable. Note in your dataset statistics that synthetic PostgreSQL-specific examples are missing; plan to add them in a future iteration.
