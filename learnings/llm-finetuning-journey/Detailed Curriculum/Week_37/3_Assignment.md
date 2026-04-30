# Week 37 Assignment — Build Your 15K SQL Training Dataset

## Setup Checklist

- [ ] `pip install datasets sqlparse anthropic openai` (or skip API packages if using free sources)
- [ ] HuggingFace `datasets` library installed
- [ ] Mac or Colab (no GPU needed for data processing)
- [ ] Optional: Claude/GPT-4o API key for synthetic generation ($10–20 budget recommended)

---

## Task 1 — Gather and Filter Public Data

**Goal:** Build a high-quality base of ~10K examples from existing datasets.

**Requirements:**
- Download at least 2 of the following datasets:
  - `b-mc2/sql-create-context` (78K examples — sample 8K after filtering)
  - `gretel/gretel-text-to-sql` (synthetic but PostgreSQL-focused)
  - Spider dev set (JSON format — download manually from [yale-nlp.github.io/spider](https://yale-nlp.github.io/spider/))
- For each dataset, apply the following filters:
  1. Remove examples where tokenized length (schema + question + sql) exceeds 400 tokens
  2. Remove examples containing MySQL-specific syntax: `AUTO_INCREMENT`, `TINYINT(1)`, `` ` `` (backtick identifiers)
  3. Remove examples where `sqlparse` cannot parse the SQL answer
  4. Remove exact duplicates (hash on `(question, sql)` pair)
- After filtering, sample to get approximately 8K–10K examples
- Save as `dataset_public.jsonl` with fields: `schema`, `question`, `sql`

**Deliverable:** `prepare_public_data.py` + `dataset_public.jsonl` committed (or a download script if too large).

---

## Task 2 — Generate Synthetic PostgreSQL Examples

**Goal:** Add 3–5K examples covering PostgreSQL-specific features absent from public datasets.

**Option A — LLM API generation (recommended, ~$5–15):**
- Generate 200 examples per feature type × 15 feature types = 3K examples
- Feature types: window functions, CTEs, JSONB, LATERAL, ARRAY, GENERATE_SERIES, date arithmetic, FULL OUTER JOIN, DISTINCT ON, COALESCE/NULLIF, HAVING, subqueries, multi-table JOIN, string functions, aggregate functions
- Use the prompt template from Curriculum.md
- Validate each generated example with `sqlparse`
- Save as `dataset_synthetic.jsonl`

**Option B — Use existing synthetic dataset (free):**
- Download `gretel/gretel-text-to-sql` or similar synthetic dataset
- Filter for PostgreSQL-compatible examples
- Sample 3–5K examples

**Deliverable:** `generate_synthetic.py` (or `download_synthetic.py`) + `dataset_synthetic.jsonl`.

---

## Task 3 — Hand-Craft TimescaleDB Examples

**Goal:** Add domain-specific examples that no public dataset contains.

**Requirements:**
- Create at least 30 (and up to 100) TimescaleDB/PostgreSQL examples by hand or by prompting an LLM with your specific TimescaleDB schema
- Cover at minimum: `time_bucket`, `time_bucket_gapfill`, interval arithmetic, `generate_series`, continuous aggregates, `DATE_TRUNC`, hypertable-specific filtering
- Save as `dataset_timescale.jsonl`
- Each example must be verified: the SQL should be syntactically correct (sqlparse) and semantically reasonable for a time-series database

**Deliverable:** `dataset_timescale.jsonl` with at least 30 entries. `create_timescale_examples.py` or a markdown doc explaining your methodology.

---

## Task 4 — Merge, Deduplicate, Format, and Split

**Goal:** Produce the final clean 15K dataset ready for Week 38 training.

**Requirements:**
- Combine all three datasets: public (8–10K) + synthetic (3–5K) + timescale (30–100)
- Run deduplication: exact-match on `(question, sql)` hash
- Run final quality checks:
  - All examples have non-empty schema, question, and sql fields
  - All SQL passes sqlparse validation
  - No example exceeds 400 tokens when tokenized with Qwen2.5 tokenizer
- Format all examples using the ChatML chat template (same as Week 29)
- Split: 14,500 training examples, 500 validation examples (random split, no test data from held-out set)
- Save: `train_15k.jsonl` and `val_500.jsonl`
- Save the dataset to HuggingFace Hub (optional but recommended): `<your-handle>/postgres-sql-15k`

**Dataset statistics to report in `week37_dataset_stats.md`:**
- Total examples before filtering
- Total after each filtering step
- Distribution by SQL feature type (% with JOINs, % with GROUP BY, % with subqueries, etc.)
- Distribution by source dataset
- Average token length (schema + question + sql)

**Deliverable:** `train_15k.jsonl`, `val_500.jsonl`, `week37_dataset_stats.md`, `build_dataset.py`. GitHub commit: `week-37-dataset`.

---

## Task 5 — Verify Dataset with a Quick Training Check

**Goal:** Confirm the dataset is correctly formatted before investing Week 38 training time.

**Requirements:**
- Run a 10-step training loop (not a full run — just 10 steps) on 100 examples from `train_15k.jsonl` using your Week 34 Unsloth script
- Confirm: loss decreases (or at least does not explode) in the first 10 steps
- Confirm: the formatted text looks correct when printed (correct `<|im_start|>` structure)
- Record: step-0 loss, step-10 loss in `week37_dataset_stats.md`

**Deliverable:** Section in `week37_dataset_stats.md` with training check results.

---

## Stretch Goals

- Use the `defog-ai/sql-eval` framework (see Resources) for an early quality check: run your Week 33 model on a subset of your new examples and measure exact match
- Add 50–100 examples with error-containing SQL and a correction question ("The following SQL has a bug: ... Fix it.") — this type of data helps with debugging tasks in Phase 6
- Compute semantic similarity within the dataset using a fast embedding model and remove near-duplicates (questions that ask the same thing with different wording)
