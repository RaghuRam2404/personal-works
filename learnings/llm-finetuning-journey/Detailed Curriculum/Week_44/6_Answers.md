# Week 44 Quiz Answers

## Q1. Answer: C

**Answer:** C — Rule-based reward from a deterministic process.

**Why:** SQL execution is a deterministic, rule-based process: a SQL query either executes without error and returns rows, or it does not. There is no human judgment and no AI model involved in the labeling decision. RLAIF involves an AI model as the labeler; RLHF involves humans. Constitutional AI is a methodology for writing principles, not a labeling mechanism by itself. Execution-based labeling is the purest form of verifiable reward, the same category as unit test passing for code.

---

## Q2. Answer: C

**Answer:** C — Discard them — DPO requires a clear quality signal.

**Why:** DPO's loss is `−log σ(reward_margin)`. If both SQL are equally correct (same rows), the reward margin should be zero, and the loss gives no gradient signal. Worse, if you randomly label one as "chosen", you are training the model to prefer arbitrary syntactic features over correct execution — which is anti-correlated with your actual goal. DPO quality depends critically on having real quality differences between chosen and rejected.

---

## Q3. Answer: B

**Answer:** B — The prompt distribution does not match the training schema.

**Why:** If your prompts reference tables or columns that do not exist in your test database (e.g., Spider prompts referencing `concert_singer` schema but your DB has a `timescale` schema), both candidate models will likely generate SQL that fails with "table does not exist". This is by far the most common cause of high discard rates. Fix: ensure your prompts reference your actual schema, or load the Spider/WikiSQL schema into your test database.

**Why others are wrong:**
- A: A 5-second timeout is generous for simple SELECT queries; timeouts would cause partial discards, not 70%.
- C: Test the harness with a known-good query first to rule this out.
- D: DPO compatibility is unrelated to execution success rates.

---

## Q4. Answer: A

**Answer:** A — When you have no Postgres database to execute queries against.

**Why:** Constitutional AI principles provide a framework for evaluating SQL quality using a strong language model (GPT-4o, Claude) as the evaluator. You ask: "Does this SQL satisfy principle 3: correct table/column names?" This is useful when execution is not possible — perhaps you are generating data for a schema that is not yet deployed, or the queries are too complex to execute in a sandbox. When execution is available, it is always preferred over CAI-based labeling because it is objective.

---

## Q5. Labeling with Expected Output

To correctly label this pair, you need a reference SQL query (the "ground truth") that you know returns the correct answer. Execute the reference SQL to get expected_rows (12 rows). Then:
- SQL_B returns 12 rows = matches expected → SQL_B is "chosen"
- SQL_A returns 15 rows ≠ expected → SQL_A is "rejected"

The infrastructure needed at scale: a reference SQL for every prompt. This can come from: the original Spider/WikiSQL dataset (which includes reference SQL), or from manually verified reference queries for your domain. Without reference SQL, you can only label pairs where one fails to execute — execution-correct but wrong pairs require ground truth to resolve. Approximate solution: use an LLM to generate a high-confidence reference SQL for each prompt and cross-validate with a second model.

---

## Q6. Imbalance in Model Wins

The 1800:700 imbalance (v1:base) is generally not a problem for DPO and may actually be desired. DPO does not require a balanced dataset of which model wins — it requires a balanced mix of hard and easy examples, and sufficient diversity of failure modes. The imbalance simply reflects that your SFT model (v1) is better at your domain SQL, which is the expected outcome after fine-tuning.

What you should check: do the 700 "base model chosen" pairs represent a systematic weakness in v1? For example, if base model wins on all JOIN queries, that suggests v1 overfits to single-table patterns during SFT. This is valuable signal — these pairs will teach DPO where v1 needs improvement. Do not rebalance by discarding v1-wins; include all clean pairs.

---

## Q7. Preference Data vs. GRPO Reward

**Preference data (DPO, Week 45):**
- Format: static dataset of (prompt, chosen, rejected) triples, created before training
- Signal computed: offline, before training begins, using execution on a test database
- Based on: comparison between two fixed candidates; winner determined by execution correctness

**GRPO reward (Weeks 47–48):**
- Format: a reward function called live during training on freshly generated completions
- Signal computed: online, during each training step, as the model generates new SQL
- Based on: absolute execution correctness of each candidate in a group of K completions (not comparative)

---

## Q8. Scenario — Colleague Concerns

**Concern 1 (valid):** If 30% of "chosen" examples execute but return wrong rows, the DPO model will learn that "SQL which executes and returns some rows" is preferred — not "SQL which returns the correct rows." This is a real labeling error. Fix: add a reference comparison step. For every pair where SQL executes, compare its output to the expected reference output. If the row set does not match expected (within a tolerance), relabel as "rejected" or discard the pair. This requires ground-truth expected outputs for each prompt, which is the main effort but is essential for data quality.

**Concern 2 (partially valid):** If 80% of rejected examples have syntax errors, DPO will primarily teach the model to avoid syntax errors rather than to produce semantically correct SQL. This is still useful — a model that does not produce syntax errors is better — but it misses the harder problem of semantic correctness. Fix: improve pair generation to include semantically wrong SQL (not just syntactically broken) as the rejected side. Generate pairs where both execute but one is semantically wrong (e.g., wrong JOIN condition, missing WHERE clause). This requires reference SQL for comparison but produces a much more informative dataset.
