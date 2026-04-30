# Week 26 Assignment — Build postgres-sql-v1 Dataset

## Setup Checklist

- [ ] Docker installed and running: `docker run -d -e POSTGRES_PASSWORD=test -p 5432:5432 postgres:16`
- [ ] `pip install psycopg2-binary sqlglot datasketch datasets huggingface_hub`
- [ ] All Week 25 scripts available: `converters.py`, `quality_filter.py`, `self_instruct.py`
- [ ] HuggingFace account with write access (`huggingface-cli login`)
- [ ] Ollama installed (optional, for free local generation): `ollama pull qwen2.5-coder:7b`

---

## Task 1 — Complete 80 Hand-Written Examples

**Goal:** Bring Tier 2 from 20 → 100 verified, executed examples.

**Requirements:**

Write 80 new examples in `postgres-sql-v1/hand_written_examples.jsonl` (append to the 20 from Week 25).

Each example must follow these rules:
- Include CREATE TABLE schema in the user message
- SQL output is PostgreSQL-valid (parses with sqlglot dialect=postgres)
- SQL executes without error on a real PostgreSQL 16 instance (Docker)
- Assistant turn contains SQL only — no explanation text
- 20 examples must use TimescaleDB-specific functions (`time_bucket`, continuous aggregates)
- 10 examples must use JSONB operators (`->`, `->>`, `@>`, `jsonb_agg`)
- 10 examples must use window functions (LAG, LEAD, RANK, PERCENT_RANK)
- 10 examples must use CTEs (WITH clause)
- 10 examples must use ON CONFLICT or RETURNING

**Verification script:**

```python
import psycopg2, sqlglot, json

conn = psycopg2.connect("dbname=postgres user=postgres password=test host=localhost")
cur = conn.cursor()

with open("hand_written_examples.jsonl") as f:
    for line in f:
        ex = json.loads(line)
        sql = ex['messages'][2]['content']  # assistant turn
        try:
            sqlglot.parse(sql, dialect="postgres")  # syntax check
            # cur.execute(sql)  # runtime check (may fail without actual tables)
            print(f"PASS: {sql[:50]}...")
        except Exception as e:
            print(f"FAIL: {e} for SQL: {sql[:50]}...")
```

**Deliverable:** `hand_written_examples.jsonl` with 100 examples (80 new + 20 from Week 25).

---

## Task 2 — Run Full Tier 1 Processing Pipeline

**Goal:** Process Spider and BIRD into 2,000 PostgreSQL-compatible ChatML examples.

**Requirements:**

Write `postgres-sql-v1/process_tier1.py` that:
1. Loads Spider train split via `load_dataset("spider")`
2. For each example: extracts schema from the tables in the db, converts to ChatML using `spider_to_chatml()`
3. Applies `sql_quality_filter()` from Week 25
4. Applies `filter_for_postgres_compat()` to remove SQLite-only syntax
5. Deduplicates with MinHash (threshold=0.7) using the question text
6. Samples up to 1,500 examples from Spider
7. Loads BIRD (from local download or HuggingFace) and adds up to 500 examples
8. Saves to `postgres-sql-v1/tier1_examples.jsonl`

Report in output:
```
Spider examples before filter: 7,000
Spider examples after all filters: 4,200
Spider after dedup: 3,800
Spider sampled: 1,500
BIRD examples after filter: 800
Final Tier 1 total: 2,000
```

**Deliverable:** `process_tier1.py` + `tier1_examples.jsonl` (2,000 examples).

---

## Task 3 — Run Self-Instruct Generation

**Goal:** Generate and filter 2,900 synthetic examples for Tier 3.

**Requirements:**

Run `self_instruct.py` (from Week 25) with:
- Seeds: all 100 examples from `hand_written_examples.jsonl`
- Target: 5,000 generation candidates (expecting 2,900 to pass filters after dedup)
- Filter each generated example through `sql_quality_filter()`
- Deduplicate Tier 3 internally (threshold=0.7)
- Also cross-deduplicate against Tier 1+2 combined

**Generation options (choose one):**
- Option A: Ollama + Qwen2.5-Coder-7B (free, slower)
- Option B: GPT-3.5-turbo API (~$5, faster)

**Stop condition:** Stop when you have 2,900 examples passing all filters, or when you have spent > $10 on API costs. If you run out of generation budget, use fewer Tier 3 examples (2,500 is acceptable; document it).

**Deliverable:** `tier3_examples.jsonl` (2,500–2,900 examples).

---

## Task 4 — Merge, Split, and Publish

**Goal:** Produce the final v1 dataset and publish it.

**Requirements:**

Write `postgres-sql-v1/build_dataset.py` that:

1. Loads `tier1_examples.jsonl`, `hand_written_examples.jsonl`, `tier3_examples.jsonl`
2. Merges all tiers, ensuring no duplicates across tiers (one final dedup pass)
3. Shuffles the full dataset (random seed=42)
4. Splits 80/20: `train.jsonl` (4,000) and `val.jsonl` (1,000)
5. Computes and prints dataset statistics:
   - Total examples per tier
   - Average SQL token count
   - Average question length
   - SQL keyword distribution (SELECT, WITH, INSERT, etc.)
6. Publishes to HuggingFace Hub:

```python
from datasets import Dataset
import json

examples = [json.loads(l) for l in open("train.jsonl")]
ds = Dataset.from_list(examples)
ds.push_to_hub("<your-handle>/postgres-sql-v1", split="train", private=True)
```

7. Writes `postgres-sql-v1/README.md` with dataset card (see Curriculum for template)

**Deliverable:** HuggingFace Hub URL + `build_dataset.py` + final JSONL files.

GitHub commit: `weeks-25-26-dataset-v1`

---

## Task 5 — Dataset Analysis Report

**Goal:** Understand your dataset's strengths and gaps before using it for fine-tuning.

**Requirements:**

Write `postgres-sql-v1/dataset_analysis.md` answering:
1. What SQL constructs are well-represented (> 10% of examples)?
2. What constructs are underrepresented (< 1% of examples)?
3. Which schemas appear most frequently? Is this a diversity problem?
4. What is the distribution of SQL complexity (token length)? Are there too many very short queries?
5. What would you add in v2 to address the gaps?

**Acceptance criteria:** Report is at least 400 words with actual statistics from the dataset.

---

## Stretch Goals

- Run your 100 hand-written examples against PostgreSQL 16 with real table creation and data insertion; verify the query returns the correct rows, not just that it runs without error
- Write a `validate_schema_consistency.py` that checks: for every column referenced in the SQL, that column exists in the schema provided in the user message
- Add 50 "negative examples" (common mistakes) with incorrect SQL; these can be used for DPO fine-tuning later (keep them in a separate file, not in training data)
