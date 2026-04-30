# Week 37 — Domain-Tuning Sprint Week 1: Build Your 15K SQL Dataset

## Learning Objectives

By the end of this week, you will be able to:

- Curate and combine SQL datasets from multiple public sources into a unified PostgreSQL-focused dataset
- Generate synthetic training examples using an LLM API (Claude or GPT-4) with controlled prompts
- Design schema-diverse examples that cover PostgreSQL-specific SQL features
- Validate and deduplicate your dataset to ensure quality
- Deliver a clean 15K-example dataset ready for QLoRA fine-tuning in Week 38

---

## Concepts

### 1. Why 15K? The Dataset Size Target

Your Week 33 run used 5K examples and achieved ~35–50% exact match on a simple held-out test. For a production-quality PostgreSQL SQL expert, you need:

- More schema diversity (more table types, column patterns)
- More question types (aggregation, window functions, CTEs, subqueries, multi-table JOINs)
- PostgreSQL-specific features: `GENERATE_SERIES`, `LATERAL`, `JSONB`, `ARRAY`, window functions, `WITH RECURSIVE`

15K examples is not a magic number — it is the minimum to cover enough variation in the PostgreSQL SQL space that the model encounters novel schemas at inference time and still generalizes. Raschka's empirical finding: model quality on structured output tasks improves meaningfully up to ~50K examples, then gains flatten.

### 2. Data Sources: Public Datasets

Start with existing public datasets, then supplement with synthetic generation:

**Tier 1 — High-quality, PostgreSQL-compatible:**
- [sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) — 78K examples with CREATE TABLE context. Most are SQLite-compatible; filter for PostgreSQL-compatible queries.
- [Spider](https://yale-nlp.github.io/spider/) — 10K expert-labeled, cross-domain. SQL style is standard; most queries are PostgreSQL-compatible.
- [WikiSQL](https://huggingface.co/datasets/wikisql) — 80K simple single-table queries. Lower complexity but good volume.

**Tier 2 — Moderate quality, needs filtering:**
- [gretel/gretel-text-to-sql](https://huggingface.co/datasets/gretel/gretel-text-to-sql) — Synthetically generated; diverse schemas and PostgreSQL-specific features.
- [Clinton/Text-to-SQL](https://huggingface.co/datasets/Clinton/Text-to-sql-v1) — Community aggregation; variable quality.

**Tier 3 — PostgreSQL-specific, smaller:**
- Your own TimescaleDB/PostgreSQL schema examples (hand-crafted)

**Strategy:** Take 5–7K high-quality examples from Tier 1, add 5–7K from synthetic generation, and add 2–3K PostgreSQL-specific examples. Total: ~15K.

### 3. Synthetic Data Generation with LLMs

For PostgreSQL-specific features that are rare in public datasets, use an LLM API to generate (schema, question, SQL) triples:

```python
PROMPT_TEMPLATE = """You are a PostgreSQL expert. Generate a realistic PostgreSQL training example.

Schema type: {schema_type}
SQL features to include: {features}

Output format (JSON):
{{
  "schema": "CREATE TABLE ... (PostgreSQL syntax)",
  "question": "Natural language question about the data",
  "sql": "Valid PostgreSQL query answering the question"
}}

Requirements:
- Use PostgreSQL-specific syntax (NOT MySQL/SQLite)
- The SQL must be executable on standard PostgreSQL 15+
- Include realistic column names and data types
- The schema should have 2–4 tables with proper foreign keys
"""

schema_types = ["e-commerce", "healthcare", "time-series metrics", "financial transactions", 
                "inventory management", "user analytics"]
features = ["window functions", "CTEs", "JSONB queries", "LATERAL joins", 
            "GENERATE_SERIES", "array operations", "date arithmetic"]
```

Budget estimate: 1,000 synthetic examples at ~500 tokens each ≈ 500K tokens. Claude 3.5 Haiku or GPT-4o-mini costs ~$0.10–0.25 for 500K tokens. This is outside the $200 budget constraint, treated as an optional $10–20 expenditure.

**If you don't want to spend money on API calls:** use [gretel/gretel-text-to-sql](https://huggingface.co/datasets/gretel/gretel-text-to-sql) or [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) as your "synthetic" layer — they are pre-generated and free.

### 4. Dataset Quality Checks

Before training, validate your dataset:

1. **Length filtering:** Remove examples longer than 512 tokens (which would be truncated). For SQL, the typical limit is 384 tokens for schema+question, 128 for SQL.
2. **SQL validation:** Use `sqlparse` to parse each SQL answer and check for syntax errors. Filter out examples where `sqlparse.parse(sql)[0].get_type()` returns `None` (not a valid SQL statement).
3. **Deduplication:** Remove near-duplicate examples. Use a simple hash on the `(question, expected_sql)` pair, or use a semantic similarity threshold.
4. **PostgreSQL filtering:** Remove examples using MySQL-specific syntax (`AUTO_INCREMENT`, `TINYINT`, backtick identifiers) — these will confuse your model.
5. **Diversity check:** Compute the distribution of SQL query types (SELECT, SELECT with JOIN, SELECT with GROUP BY, etc.) and ensure no single type dominates (>60%) the dataset.

```python
import sqlparse

def is_valid_sql(sql_str):
    try:
        parsed = sqlparse.parse(sql_str)
        return len(parsed) > 0 and parsed[0].get_type() is not None
    except:
        return False
```

### 5. Dataset Format and Schema

Your final dataset must match the Week 29/31/33 format:

```python
{
    "messages": [
        {"role": "system", "content": "You are a PostgreSQL expert. Given a table schema and a question, write a valid PostgreSQL SQL query."},
        {"role": "user", "content": f"Schema:\n{schema}\nQuestion:\n{question}"},
        {"role": "assistant", "content": sql}
    ]
}
```

Or the pre-formatted text version:
```python
{
    "text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
}
```

### 6. Adding TimescaleDB Examples

Since your target is PostgreSQL/TimescaleDB, manually create at least 50–100 examples using TimescaleDB-specific features:
- `time_bucket('1 hour', timestamp_col)` — time bucketing
- `SELECT ... FROM hypertable WHERE timestamp > NOW() - INTERVAL '7 days'`
- Continuous aggregates: `SELECT * FROM cagg WHERE bucket BETWEEN ... AND ...`

These are high-value examples that no public dataset contains — they are part of what makes your model better than the generic base for your specific domain.

---

## Connections

**Builds on:** All Phase 4 training infrastructure (Weeks 29–36). Dataset is the fuel for Week 38's training run.

**Needed for:** Week 38 (the training), Week 39 (evaluation harness uses your held-out test set).

---

## Common Misconceptions / Pitfalls

- **"More data is always better."** Not if it includes low-quality synthetic examples that confuse the model. Quality > quantity at this scale.
- **"I can use MySQL examples from WikiSQL."** MySQL syntax differs from PostgreSQL in important ways. Filter or convert.
- **"15K examples is enough to beat GPT-4o."** Your Phase 4 goal is to beat the base Qwen2.5-Coder-7B, not GPT-4o. GPT-4o beating happens in Phase 6 with 100K+ examples and GRPO.
- **"I should include all 78K examples from sql-create-context."** More data helps, but training cost increases. 15K is the sweet spot for this compute budget and time horizon.

---

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Survey and download public datasets | 1h |
| Write data processing and filtering pipeline | 2h |
| Generate synthetic examples (API or from existing synthetic datasets) | 1.5h |
| Run quality checks and deduplication | 1h |
| Hand-craft 20–50 TimescaleDB-specific examples | 1h |
| Format final dataset, verify statistics | 30m |
