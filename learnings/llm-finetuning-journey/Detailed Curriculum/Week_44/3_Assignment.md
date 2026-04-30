# Week 44 Assignment — Build SQL Preference Dataset

## Setup Checklist

- [ ] PostgreSQL running locally (or a cloud Postgres instance). Your TimescaleDB setup from earlier phases works.
- [ ] Python packages: `psycopg2-binary`, `transformers`, `datasets`, `huggingface_hub`, `torch`
- [ ] HuggingFace account and `huggingface-cli login` completed
- [ ] Your SFT model checkpoint from Phase 4 (`postgres-sqlcoder-7b-v1`) accessible
- [ ] W&B project for logging stats (optional but recommended)
- [ ] At least 3000 SQL prompts (adapted from Spider/WikiSQL or synthetically generated)

---

## Task 1 — Build the Execution Harness

**Goal:** Write a robust `execute_sql()` function that safely runs SQL on Postgres and returns structured results.

**Requirements:**
- Connect to Postgres using `psycopg2`
- Execute SQL with a 5-second timeout (use `set_session` or `statement_timeout`)
- Return a dict: `{"success": bool, "rows": list[tuple] | None, "error": str | None, "row_count": int}`
- Handle: syntax errors, runtime errors, empty results, connection errors — all must return `success=False` gracefully
- Do NOT allow DML (INSERT, UPDATE, DELETE, DROP) — validate that the SQL starts with SELECT or WITH
- Test with at least: a valid SELECT, a syntax error, a table-not-found error, an empty result

**Deliverable:** `week-44-prefdata/execute_sql.py`

**Hints:**
```python
import psycopg2
import psycopg2.extras

def execute_sql(sql: str, conn_str: str, timeout_ms: int = 5000) -> dict:
    # Validate it's a SELECT
    if not sql.strip().upper().startswith(("SELECT", "WITH")):
        return {"success": False, "rows": None, "error": "Non-SELECT blocked", "row_count": 0}
    try:
        with psycopg2.connect(conn_str) as conn:
            conn.set_session(options={"statement_timeout": str(timeout_ms)})
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
                return {"success": True, "rows": rows, "error": None, "row_count": len(rows)}
    except Exception as e:
        return {"success": False, "rows": None, "error": str(e), "row_count": 0}
```

---

## Task 2 — Generate SQL Candidates

**Goal:** For each prompt, generate 2 SQL completions using two different models.

**Requirements:**
- Load your SFT model (v1) and the base model (Qwen2.5-Coder-7B-Instruct)
- For each prompt in your prompt set, generate one completion from each
- Use greedy decoding (temperature=0.0, do_sample=False) for consistency
- Max new tokens: 256
- Log: for each prompt, save the prompt text, SQL_A (from v1), SQL_B (from base model)
- Process at least 3000 prompts

**Deliverable:** `week-44-prefdata/candidates.jsonl` — one JSON object per line with keys: `prompt`, `sql_a`, `model_a`, `sql_b`, `model_b`

---

## Task 3 — Label Preferences via Execution

**Goal:** Execute both candidates and produce clean preference pairs.

**Requirements:**
- Load `candidates.jsonl`
- For each pair: execute both SQL_A and SQL_B using your `execute_sql()` harness
- Labeling logic:
  - If A executes successfully (success=True, row_count > 0) and B does not: label = `(prompt, A, B)` with A as chosen
  - If B executes successfully and A does not: label = `(prompt, B, A)` with B as chosen  
  - If both execute (compare row_count to expected if you have ground truth): label based on correctness
  - If neither executes: discard
  - If both execute with same row_count: discard (ambiguous)
- Target: at least 2000 clean pairs
- Log statistics: total attempted, successful pairs, discard rate, model_a wins, model_b wins

**Deliverable:** `week-44-prefdata/preferences.jsonl`

---

## Task 4 — Push to HuggingFace Hub

**Goal:** Convert to HuggingFace Dataset format and push.

**Requirements:**
- Load `preferences.jsonl` into a `datasets.Dataset`
- Split: 90% train, 10% validation
- Schema: `prompt` (str), `chosen` (str), `rejected` (str)
- Add metadata fields: `chosen_model`, `rejected_model`, `chosen_executes` (always True)
- Push to Hub as: `<your-handle>/postgres-sql-preferences-v1`
- Write a dataset card explaining how it was built

**Deliverable:** HuggingFace dataset `<your-handle>/postgres-sql-preferences-v1`

---

## Stretch Goals

- Add a "hardness" field: easy (single table), medium (join), hard (subquery/CTE). Analyze win rates by hardness.
- Use a third model (GPT-4o via API) to judge ambiguous cases (both execute, different row counts). Compare AI labeling vs. execution labeling.
- Run a "constitution check": for each chosen SQL, verify it satisfies all 5 principles from your SQL constitution. Flag any that violate principle 3 (schema hallucinations).
