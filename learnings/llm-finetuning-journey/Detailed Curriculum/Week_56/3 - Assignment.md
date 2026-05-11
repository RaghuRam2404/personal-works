# Week 56 Assignment — Add 5K Multi-Turn SQL Examples to v3

## Setup Checklist

- [ ] CoSQL downloaded: `git clone https://github.com/taoyds/cosql`
- [ ] SParC downloaded: `git clone https://github.com/taoyds/sparc`
- [ ] PostgreSQL with Spider databases loaded (convert SQLite to Postgres using pgloader or manual)
- [ ] Teacher API access for synthetic multi-turn generation
- [ ] `v3_filtered.jsonl` from Week 55 accessible
- [ ] W&B project `week-56-multiturn` created

---

## Task 1 — Convert CoSQL and SParC to PostgreSQL Format

**Goal:** Extract and validate CoSQL/SParC examples that work with PostgreSQL.

**Requirements:**
Write `convert_cosql.py` that:
- Loads CoSQL `sql_state_tracking/cosql_train.json`
- For each conversation: extracts all (utterance, sql) pairs as turns
- Converts SQL from SQLite to PostgreSQL dialect using `sqlglot.transpile(sql, read="sqlite", write="postgres")`
- Runs each converted SQL against Postgres (load Spider SQLite DBs as Postgres schemas first)
- Marks conversations as `valid` only if ALL turns execute without error
- Reformats valid conversations to your training chat template format (see Curriculum)
- Saves to `cosql_postgres_valid.jsonl`
- Repeat the same for SParC: `sparc_postgres_valid.jsonl`

Log:
- Number of conversations loaded
- Number valid (all turns pass execution)
- Number partially valid (some turns pass)
- SQL syntax errors by type

**Expected yield:** ~2,000–3,000 valid conversations from CoSQL + SParC combined.

**Deliverable:** `convert_cosql.py` + `cosql_postgres_valid.jsonl` + `sparc_postgres_valid.jsonl` committed.

---

## Task 2 — Generate Synthetic TimescaleDB Multi-Turn Examples

**Goal:** Create 2,000 multi-turn conversations specific to your TimescaleDB domain.

**Requirements:**
Write `generate_multiturn.py` that:
- Selects a single-turn example from your filtered v3 dataset (preferably TimescaleDB or complex multi-table)
- Calls the teacher model with this prompt:

```
Given this text-to-SQL example:
Schema: {schema_ddl}
Question: {question}
SQL: {sql}

Generate a 3-turn conversation where:
Turn 1: The question above (use it exactly)
Turn 2: A realistic follow-up that refines or extends the query (add a filter, change aggregation granularity, add a column)
Turn 3: A further refinement (e.g., add gap-filling for time-series, change time range, add a join)

Each turn must have a valid SQL answer executable against the schema.
Return JSON: [{"user": "...", "sql": "..."}, {"user": "...", "sql": "..."}, {"user": "...", "sql": "..."}]
```

- Validates ALL turn SQLs execute against Postgres
- Only keeps conversations where all 3 turns pass
- Targets at least 800 conversations with TimescaleDB-specific turn 2 or 3 (time_bucket, hyperfunctions)
- Saves to `synthetic_multiturn.jsonl`

**Deliverable:** `generate_multiturn.py` + `synthetic_multiturn.jsonl` with ≥ 2,000 valid conversations.

---

## Task 3 — Merge and Format Final Dataset v3

**Goal:** Combine single-turn and multi-turn examples into the final training dataset.

**Requirements:**
Write `build_v3_final.py` that:
- Loads `v3_filtered.jsonl` (single-turn)
- Loads `cosql_postgres_valid.jsonl`, `sparc_postgres_valid.jsonl`, `synthetic_multiturn.jsonl`
- Reformats all examples into a consistent training format:

```python
# Single-turn format
{"messages": [
    {"role": "system", "content": "You are an expert PostgreSQL/TimescaleDB engineer.\nSchema:\n{ddl}"},
    {"role": "user", "content": question},
    {"role": "assistant", "content": sql}
]}

# Multi-turn format  
{"messages": [
    {"role": "system", "content": "You are an expert PostgreSQL/TimescaleDB engineer.\nSchema:\n{ddl}"},
    {"role": "user", "content": turn1_question},
    {"role": "assistant", "content": turn1_sql},
    {"role": "user", "content": turn2_question},
    {"role": "assistant", "content": turn2_sql},
    {"role": "user", "content": turn3_question},
    {"role": "assistant", "content": turn3_sql},
]}
```

- Runs final deduplication across single + multi-turn
- Saves to `v3_final.jsonl`
- Logs dataset statistics: total examples, single vs multi-turn count, skill distribution, token length distribution

**Acceptance criteria:**
- At least 25,000 total examples (single + multi-turn)
- At least 4,000 multi-turn examples
- Maximum example token length ≤ 4,096 tokens (truncate or split if longer)
- No example appears in Spider or BIRD test sets

**Deliverable:** `v3_final.jsonl` pushed to HuggingFace as `<your-handle>/postgres-sql-v3`.

---

## Task 4 — Sample Inspection

**Goal:** Manually inspect 20 random examples from the final dataset.

**Requirements:**
- Write a small script `inspect_v3.py` that prints 20 random examples from `v3_final.jsonl` in readable format
- Manually review each — note any obvious errors, weird phrasings, or schema mismatches
- Write `v3_inspection_notes.md` with your findings and any examples removed as a result

**Deliverable:** `v3_inspection_notes.md` committed.

---

## Stretch Goals

- Build a token length histogram of your final dataset and identify the p90, p95, p99 token lengths; this informs your training sequence length setting in Week 58
- Implement a "conversation coherence check": for each multi-turn conversation, verify that the second turn's question references at least one entity from the first turn's SQL result (using simple keyword overlap)
- Generate 5 "long-form" conversations (6–8 turns) simulating a full analytics session in TimescaleDB
