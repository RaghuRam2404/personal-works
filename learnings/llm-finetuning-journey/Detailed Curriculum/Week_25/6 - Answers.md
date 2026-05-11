# Week 25 Quiz Answers

## Q1 — Answer: B

**Answer:** B — Only on the assistant's response tokens.

**Why:** Training the model on the loss of system and user tokens would teach it to "predict the question" rather than "answer the question." The model already has strong language capabilities — you are fine-tuning it to improve only its response behavior. Computing loss on user turns is a common bug that produces models that generate user-side text when prompted for assistant output. In HuggingFace TRL's `SFTTrainer`, pass `dataset_text_field` or use `DataCollatorForCompletionOnlyLM` with the assistant token mask.

**Why others are wrong:**
- A: computing loss on all turns conflates question-generation with answer-generation
- C: computing loss on only 50 tokens truncates SQL queries prematurely
- D: computing loss only on system prompts would train the model to generate system prompts

---

## Q2 — Answer: B

**Answer:** B — Using a language model to generate new instructions from seed examples, then generating responses.

**Why:** The Self-Instruct paper (Wang et al. 2022) introduced a bootstrapping approach: (1) start with ~175 hand-written seed instructions, (2) prompt GPT-3 to generate 8 new instructions inspired by 8 randomly sampled seeds, (3) filter out near-duplicates using ROUGE similarity, (4) generate input-output completions for surviving instructions, (5) add to the growing pool and repeat. This enables generating large instruction datasets at low cost by using the model's own generation capability.

**Why others are wrong:**
- A: CC mining for instructions is a different technique (used by some datasets but not Self-Instruct)
- C: human paraphrasers are not involved in Self-Instruct; it is fully automated
- D: a separate generation model is not trained; the same base model is used

---

## Q3 — Answer: B

**Answer:** B — Up-weight hand-written examples by repeating them more frequently.

**Why:** Hand-written examples by a domain expert represent higher-quality signals than auto-generated examples. The model benefits from seeing these correct patterns more frequently. Common strategies: repeat hand-written examples 3–5× in the training data, or create a weighted sampler. Even 500 high-quality examples, repeated 5×, contribute 2,500 effective training examples that strongly reinforce the target behavior.

**Why others are wrong:**
- A: 500 examples is substantial when each one is high-quality; discarding them wastes valuable signal
- C: synthetic examples have systematic biases (GPT-4 preferences, generic schemas); exclusive use of synthetic data risks those biases
- D: using hand-written examples only for validation wastes them as training signal

---

## Q4 — Answer: B

**Answer:** B — `GROUP_CONCAT` is SQLite-specific; PostgreSQL uses `STRING_AGG`.

**Why:** `GROUP_CONCAT(col, ',')` is SQLite's aggregate for concatenating string values within a group. PostgreSQL's equivalent is `STRING_AGG(col, ',')`. If you include Spider examples with `GROUP_CONCAT` in your PostgreSQL training data, the model will learn to generate invalid SQL that fails at runtime. This is one of the most common conversion errors when adapting Spider data for PostgreSQL.

**Why others are wrong:**
- A: `GROUP_CONCAT` is a string aggregation function, not a mathematical operation
- C: `GROUP_CONCAT` does not affect query planning; it would simply error at parse/execution time
- D: `GROUP_CONCAT` is still valid in SQLite, MySQL, and other databases; it is not deprecated

---

## Q5 — Answer: B

**Answer:** B — To give the model table names, column names, and data types.

**Why:** SQL queries reference specific table and column names. Without the schema, the model must guess or hallucinate table names. Including the CREATE TABLE statements in the prompt conditions the model on the actual schema, enabling it to write correct SQL that references real columns. This is the critical difference between a SQL assistant that works on specific databases vs. one that produces generic, non-executable SQL.

**Why others are wrong:**
- A: DDL learning is a secondary benefit; the primary purpose is schema conditioning for each query
- C: schema quality assessment is not the objective of fine-tuning
- D: PostgreSQL does not require DDL before queries at runtime; the schema exists in the database

---

## Q6 — Short Answer

When the assistant turn contains explanatory text ("Sure! Here is a query to do that: SELECT ..."), the model learns to output explanations before SQL. In deployment, you want the model to output only the SQL query so it can be executed directly. Training with explanation-contaminated responses produces a model that "thinks aloud" before generating SQL, requiring you to post-process every response to extract the SQL — fragile and error-prone. Furthermore, explanation text is not verified (it may be wrong even when the SQL is correct), and including it in the loss trains the model to optimize for producing plausible-sounding explanations rather than correct SQL. Your assistant turn must contain exactly what you want the model to output: the SQL query only.

---

## Q7 — Short Answer (SQLite vs PostgreSQL)

1. `GROUP_CONCAT(col, ',')` (SQLite) → `STRING_AGG(col, ',')` (PostgreSQL)
2. `AUTOINCREMENT` (SQLite) → `SERIAL` or `GENERATED ALWAYS AS IDENTITY` (PostgreSQL)
3. `LIMIT 10 OFFSET 5` written as `LIMIT 10, 5` in SQLite → must be `LIMIT 10 OFFSET 5` in PostgreSQL (the comma shorthand is SQLite/MySQL specific)
4. `strftime('%Y-%m', date_col)` (SQLite) → `TO_CHAR(date_col, 'YYYY-MM')` or `date_trunc('month', date_col)` (PostgreSQL)

Bonus: `INSERT OR REPLACE` (SQLite) → `INSERT ... ON CONFLICT DO UPDATE` (PostgreSQL)

---

## Q8 — Short Answer (1,100 more examples needed)

1. **Generate more synthetic examples with adjusted prompts.** Your 1,200 filtered-out examples failed because they had invalid SQL (sqlglot rejection) or non-SQL output. Adjust the generation prompt to be more explicit: "Output ONLY a valid PostgreSQL SQL query starting with SELECT, INSERT, UPDATE, DELETE, CREATE, or WITH. Do not include any explanation." Re-run generation on the 1,200 failed instructions with this stricter prompt. Expect ~70% pass rate → ~840 additional examples.

2. **Add more benchmark data.** Include BIRD examples that pass your filter (you used only 500 of the available 8,000+ BIRD train examples). Run your filter on all BIRD examples and add more passing ones. BIRD has more complex, real-world queries than Spider.

3. **Augment your hand-written examples.** Take your 20 hand-written examples and create variations: different table names, different column names, different WHERE conditions, different time ranges. This "template filling" approach produces controlled-quality examples without additional API calls. 20 templates × 10 variations = 200 additional high-quality examples; the remaining 900 come from improved synthetic generation.

---

## Q9 — Scenario Model Answer

**1. Structural problems with WikiSQL:**
(a) WikiSQL uses single-table queries only — the entire dataset is limited to "SELECT col FROM table WHERE col = val" style queries. There are no JOINs, no subqueries, no CTEs, no window functions. Your TimescaleDB assistant needs to write multi-table queries with complex temporal predicates.
(b) WikiSQL uses a fixed SQL grammar (column select, aggregation, WHERE clause) — it does not support the full SQL syntax. Queries in WikiSQL are generated from a formal grammar, not written by humans, producing an unrealistically limited SQL distribution.

**2. Missing SQL features for TimescaleDB:**
WikiSQL (2017) predates: window functions (standardized widely post-2018 adoption), CTEs (`WITH` clause), `ON CONFLICT` syntax, `RETURNING` clause, `JSONB` operators, time-series functions like `time_bucket()`, `generate_series()`, and continuous aggregates. Essentially, WikiSQL covers only the most basic SQL subset.

**3. Processing steps needed:**
- Filter to single-table examples only (they are all single-table, so no join examples to filter)
- Convert from WikiSQL's formal SQL grammar notation to standard SQL syntax
- Add table schemas (WikiSQL has table metadata but not CREATE TABLE syntax)
- Convert field names (WikiSQL uses underscored lowercase table names; add realistic PostgreSQL schemas)
- Still would not cover TimescaleDB-specific features regardless of processing

**4. Why 5K high-quality beats 80K WikiSQL:**
Your deployment scenario requires correct PostgreSQL syntax on your specific database schemas, including TimescaleDB extensions. WikiSQL trains the model on a distribution that is systematically different from your target: single-table, grammar-restricted, legacy SQL without any time-series features. Training on 80K examples of the wrong distribution (simple, generic, SQLite-style SQL) actively harms your model by teaching it patterns that are wrong for your target. Your 5K examples from Spider/BIRD (diverse SQL), hand-written (TimescaleDB-specific), and self-instruct (conditioned on your schemas) represent the actual target distribution. In transfer learning, data distribution alignment matters far more than raw example count.
