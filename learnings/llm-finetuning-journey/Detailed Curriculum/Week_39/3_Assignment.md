# Week 39 Assignment — Build the Execution-Based Evaluation Harness

## Setup Checklist

- [ ] Colab Pro notebook open with GPU runtime (T4 or A100)
- [ ] Week 38 model adapter saved locally or on HuggingFace Hub (`<your-handle>/postgres-sqlcoder-7b-v1`)
- [ ] `held_out_test.json` from Week 32 (100 examples, never used in training)
- [ ] Python packages: `psycopg2-binary`, `sqlparse`, `datasets`, `transformers`, `peft`
- [ ] PostgreSQL installed (Option A) or Docker available (Option B)
- [ ] GitHub repo open for committing the harness

Install dependencies in Colab:
```bash
!apt-get install -y postgresql > /dev/null
!service postgresql start
!sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
!pip install psycopg2-binary sqlparse -q
```

---

## Task 1 — Implement the Core Harness

**Goal:** Write a self-contained `eval_harness.py` that takes a JSONL test file, spins up Postgres, runs both expected and generated SQL, and reports execution correctness.

**Requirements:**

- Load `held_out_test.json`; each record has `schema_sql`, `question`, `expected_sql`, and `generated_sql` (you will add `generated_sql` in Task 2)
- For each example:
  - Create a fresh database (or schema) per example, or drop and recreate tables between runs
  - Execute `expected_sql` to produce the reference result set
  - Execute `generated_sql` with a 5-second statement timeout
  - Reject any non-SELECT statement before execution
  - Compare result sets using row-sorted comparison with NULL normalization
- Output a JSONL file `eval_results.jsonl` where each line has: `id`, `question`, `expected_sql`, `generated_sql`, `exec_success` (bool), `exec_correct` (bool), `error_msg` (str or null)
- Print a summary at the end: total, exec_success %, exec_correct %

**Deliverable:** `week39/eval_harness.py` committed to your GitHub repo

---

## Task 2 — Generate SQL from Three Models

**Goal:** Run inference with three models on `held_out_test.json` and attach generated SQL to the eval inputs.

**Requirements:**

- Generate SQL predictions from:
  1. Base model: `Qwen/Qwen2.5-Coder-7B` (no fine-tuning)
  2. Week 33 model: your first QLoRA fine-tune (from Week 33)
  3. Week 38 model: `<your-handle>/postgres-sqlcoder-7b-v1`
- For each model, write predictions to a separate file: `preds_base.jsonl`, `preds_week33.jsonl`, `preds_week38.jsonl`
- Use the same prompt format you used in Week 38 training (system prompt + schema + question → SQL)
- Limit generation to 256 tokens max; use greedy decoding (temperature=0, do_sample=False)
- Log inference time per model

**Hints:**

- Load the base model in BitsAndBytes 4-bit (NF4) so it fits on a Colab T4
- For adapter models, use `PeftModel.from_pretrained(base_model, adapter_path)`
- Extract only the SQL part after the model's stop token; strip markdown fences if present

**Deliverable:** `preds_base.jsonl`, `preds_week33.jsonl`, `preds_week38.jsonl` (each with 100 rows)

---

## Task 3 — Run the Harness on All Three Models

**Goal:** Produce execution correctness numbers for all three models on the same 100-example test set.

**Requirements:**

- Run `eval_harness.py` three times (once per predictions file)
- Collect into a single comparison table:

| Model | Exec Success % | Exec Correct % |
|---|---|---|
| Base Qwen2.5-Coder-7B | ? | ? |
| Week 33 QLoRA | ? | ? |
| Week 38 QLoRA (v1) | ? | ? |

- Also compute exact match % for each model (string equality after stripping whitespace)
- Save full results to `eval_results_base.jsonl`, `eval_results_week33.jsonl`, `eval_results_week38.jsonl`

**Deliverable:** Completed comparison table in `week39_eval_report.md`

---

## Task 4 — Error Analysis

**Goal:** Understand where your Week 38 model fails.

**Requirements:**

- From `eval_results_week38.jsonl`, extract examples where `exec_correct = false`
- Categorize each failure into one of:
  - **Syntax error:** model SQL fails to execute
  - **Wrong table/column:** references table or column not in schema
  - **Wrong aggregation:** uses SUM instead of COUNT, or similar
  - **Wrong filter:** correct structure, wrong WHERE clause
  - **Wrong join:** incorrect join condition or missing join
  - **Other**
- Count failures in each category
- Add a section "Error Analysis" to `week39_eval_report.md` with:
  - Count per category
  - 2–3 concrete example failures with the expected SQL and generated SQL side by side
  - Your hypothesis for the top failure category

**Deliverable:** "Error Analysis" section in `week39_eval_report.md`

---

## Task 5 — Write the Eval Report

**Goal:** Produce `week39_eval_report.md` — a clean, complete evaluation report.

**Requirements:**

- Sections: Setup, Results Table, Error Analysis, Observations, Next Steps
- In Observations: note whether execution correctness > exact match (it should be)
- In Next Steps: list 3 specific interventions for Phase 5 (e.g., add GRPO training with this harness as reward signal)
- Keep it under 2 pages — dense and factual, not a narrative

**Deliverable:** `week39_eval_report.md` committed to GitHub

---

## Stretch Goals

- Add a GPT-4o baseline: call the OpenAI API on each of the 100 questions and record its exec correctness (requires API key)
- Add confidence-based filtering: only count a prediction as "attempted" if model log-probability of the generated SQL exceeds a threshold
- Extend the harness to support TimescaleDB time-series queries (install TimescaleDB extension in your Postgres instance)
- Implement partial credit: award 0.5 if row count matches but values differ
