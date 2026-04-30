# Week 53 Assignment — Design the v3 Data Strategy

## Setup Checklist

- [ ] Your v2 dataset is accessible (HuggingFace or local JSONL)
- [ ] Python environment with `datasets`, `pandas`, `numpy` installed
- [ ] A PostgreSQL instance running locally (for schema introspection)
- [ ] Access to Spider and BIRD-SQL test sets (for contamination checking)
- [ ] GitHub repo `llm-finetuning-journey` with Week 53 branch ready

---

## Task 1 — Annotated Paper Reading

**Goal:** Extract actionable design decisions from LIMA and Tulu 3 for your specific domain.

**Requirements:**
- Read LIMA (arXiv 2305.11206) and create a `lima_notes.md` file with:
  - The exact 1,000-example selection criteria they used
  - Which 19 task categories they covered
  - Their key finding on diversity vs. quantity (quote the relevant table)
  - One concrete decision you will apply to your dataset
- Read Tulu 3 (arXiv 2411.15124) Sections 1–4 and create `tulu3_notes.md` with:
  - Their skill taxonomy (list every skill category)
  - Their deduplication strategy (exact steps, thresholds)
  - Their contamination removal approach
  - Two specific techniques you will adopt for v3

**Deliverable:** `lima_notes.md` and `tulu3_notes.md` committed to `week-53-strategy` branch.

---

## Task 2 — v2 Dataset Audit Script

**Goal:** Understand exactly what you have before deciding what to build.

**Requirements:**
- Write a Python script `audit_v2.py` that loads your v2 dataset and produces:
  - Total example count
  - Distribution of SQL constructs (SELECT-only, JOIN, GROUP BY, WINDOW, CTE, subquery, TimescaleDB-specific) — use regex or AST parsing via `sqlglot`
  - Number of distinct schemas (count unique table name sets)
  - Approximate difficulty distribution (use query length as proxy: <50 tokens = easy, 50–150 = medium, >150 = hard)
  - Near-duplicate count using MinHash (use `datasketch` library: jaccard threshold 0.85)
  - Contamination estimate: how many training examples share 5+ consecutive tokens with Spider or BIRD test sets
- Log all metrics to a file `v2_audit_results.json`

**Hints:**
- Install `sqlglot` for SQL AST parsing: `pip install sqlglot`
- Install `datasketch` for MinHash: `pip install datasketch`
- For contamination, convert each example to a set of 5-grams (word-level), then check intersection with test set 5-grams

**Deliverable:** `audit_v2.py` + `v2_audit_results.json` committed. W&B run `week-53-audit` with logged metrics.

---

## Task 3 — v3 Data Card

**Goal:** Write a binding specification for Dataset v3 before generating a single example.

**Requirements:**
Write `data_card_v3.md` (minimum 600 words) covering:
- Dataset name: `<your-handle>/postgres-sql-v3`
- Intended use (what task, what model, what not to use for)
- Target size: 50K examples (or justify a different number)
- Skill taxonomy: list every SQL/PostgreSQL/TimescaleDB skill with target count
- Difficulty distribution targets (Easy/Medium/Hard/Expert with %)
- Schema diversity target (minimum number of distinct schemas)
- Source breakdown (Spider-augmented / hand-curated / synthetic / CoSQL)
- Quality acceptance criteria (the checklist an example must pass to be included)
- Contamination exclusion list (which benchmarks are off-limits for training)
- Known limitations

**Deliverable:** `data_card_v3.md` committed. This document will become Section 2.1 of your technical report.

---

## Task 4 — Gap Analysis

**Goal:** Identify precisely which skills are under-represented in v2.

**Requirements:**
- Using your audit results, create a table in `gap_analysis.md`:
  - Column 1: SQL/TimescaleDB skill
  - Column 2: Current count in v2
  - Column 3: Target count in v3
  - Column 4: Gap (= target - current)
  - Column 5: Planned source (synthetic / hand-written / augmented)
- Prioritize the top 5 gaps — these drive Week 54's generation prompts

**Deliverable:** `gap_analysis.md` committed.

---

## Stretch Goals

- Implement a query complexity scorer using `sqlglot`'s AST depth (not just token length)
- Check your v2 dataset's perplexity distribution under your base Qwen2.5-Coder-7B model — high perplexity outliers may be errors or invaluable hard examples; low perplexity may be duplicates of pretraining data
- Draft the teacher model prompts you plan to use in Week 54 (store as `generation_prompts_draft.md`)
