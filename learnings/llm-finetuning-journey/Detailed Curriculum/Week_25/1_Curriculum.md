# Week 25 — Domain Dataset Construction: Formats, Self-Instruct, and Planning

## Learning Objectives

By the end of this week, you will be able to:

- Explain the Alpaca, ShareGPT, and ChatML dataset formats and convert between them
- Implement Self-Instruct to generate synthetic instruction-response pairs
- Plan the full PostgreSQL/TimescaleDB dataset v1 (5K examples)
- Write data curation scripts that enforce schema consistency and quality filters
- Identify the three sources you will use for your dataset and their trade-offs

---

## Concepts

### Why Dataset Format Matters

The same data can be stored in multiple formats. Most modern fine-tuning frameworks expect the data in a specific structure. If your data is in the wrong format, your training will silently produce garbage (no error, but the model learns the wrong thing).

**The three most common formats:**

**Alpaca format** (single-turn instruction → response):
```json
{
    "instruction": "Write a SQL query to find all users who signed up last month.",
    "input": "Table: users(id, email, created_at)",
    "output": "SELECT id, email FROM users WHERE created_at >= date_trunc('month', NOW() - INTERVAL '1 month') AND created_at < date_trunc('month', NOW());"
}
```
Good for: simple single-turn Q&A, short instruction → short output pairs. Used by Stanford Alpaca, WizardLM.

**ShareGPT format** (multi-turn conversation):
```json
{
    "conversations": [
        {"from": "human", "value": "How do I query hourly averages in TimescaleDB?"},
        {"from": "gpt",   "value": "Use time_bucket: SELECT time_bucket('1 hour', ts) AS hour, AVG(value) FROM readings GROUP BY hour ORDER BY hour;"},
        {"from": "human", "value": "What if I want to exclude NULL values?"},
        {"from": "gpt",   "value": "Add a WHERE clause: ... WHERE value IS NOT NULL ..."}
    ]
}
```
Good for: multi-turn dialogues, follow-up questions. Used by Vicuna, OpenHermes.

**ChatML format** (the current standard):
```
<|im_start|>system
You are a PostgreSQL expert. Generate correct SQL for the given question and schema.
<|im_end|>
<|im_start|>user
Table: orders(id, user_id, amount, created_at)
Question: Total revenue by month for 2024
<|im_end|>
<|im_start|>assistant
SELECT
    date_trunc('month', created_at) AS month,
    SUM(amount) AS total_revenue
FROM orders
WHERE created_at BETWEEN '2024-01-01' AND '2024-12-31'
GROUP BY month
ORDER BY month;
<|im_end|>
```

ChatML is now the standard used by OpenAI, Qwen2.5, Llama 3 (with slight variations), and most HuggingFace models. When you fine-tune Qwen2.5-Coder, you must use the ChatML format that matches its tokenizer's `chat_template`.

**Converting formats:**

```python
def alpaca_to_chatml(example: dict) -> dict:
    """Convert Alpaca format to ChatML format."""
    user_content = example['instruction']
    if example.get('input'):
        user_content += f"\n\n{example['input']}"
    return {
        "messages": [
            {"role": "system", "content": "You are a PostgreSQL expert."},
            {"role": "user",   "content": user_content},
            {"role": "assistant", "content": example['output']}
        ]
    }
```

### Self-Instruct: Generating Synthetic Instruction Data

[Self-Instruct](https://arxiv.org/abs/2212.10560) (Wang et al. 2022) is the paper that launched synthetic data generation for LLMs. The key idea:

1. Start with a small seed set of hand-written (instruction, response) pairs
2. Prompt a language model to generate new instructions inspired by the seed set
3. Use the same LM to generate responses to the new instructions
4. Filter for quality and add to the training set
5. Repeat

**Self-Instruct for SQL generation:**

```python
SEED_EXAMPLES = [
    {"instruction": "Find all orders placed in Q1 2024", 
     "output": "SELECT * FROM orders WHERE created_at BETWEEN '2024-01-01' AND '2024-03-31';"},
    # ... more examples
]

GENERATION_PROMPT = """Generate {n} diverse SQL-related questions for a PostgreSQL database assistant.
Each question should be answerable with a single SQL query.
Use these examples as inspiration:
{examples}

New questions (different from the above):"""
```

For your project, you will use GPT-3.5-turbo or Claude Haiku (cheap) as the generator LM, or use Qwen2.5-Coder-7B locally with Ollama.

### Your Dataset Plan: 5,000 Examples in Three Tiers

**Tier 1: Converted Standard Benchmarks (2,000 examples)**

Source: Spider, BIRD, WikiSQL — existing NL→SQL datasets with high-quality human-written SQL.

Conversion plan:
- Download Spider train split (7,000 examples) → filter for PostgreSQL-compatible syntax → convert to ChatML → take 1,500 examples
- Download BIRD train split → filter → take 500 examples

What to filter:
- Remove SQLite-specific syntax (`GROUP_CONCAT`, no `RETURNING`, no `ON CONFLICT`)
- Remove examples with > 5 tables (too complex for initial v1 training)
- Remove very short queries (< 20 tokens) — likely too easy

**Tier 2: Hand-Written PostgreSQL/TimescaleDB Pairs (100 examples)**

These are the highest-quality examples because they reflect your actual domain. Write them from scratch. Include:
- 30 basic TimescaleDB examples (time_bucket, continuous aggregates, hypertable creation)
- 30 PostgreSQL-specific examples (ON CONFLICT, RETURNING, JSONB, window functions)
- 20 multi-table join examples using realistic TimescaleDB schemas
- 20 "tricky" examples: correlated subqueries, recursive CTEs, LATERAL joins

**Tier 3: Synthetic Self-Instruct Examples (2,900 examples)**

Generate using Qwen2.5-Coder-7B or an API model. Process:
1. Start with your 100 hand-written examples as seeds
2. Generate 300 instruction prompts using Self-Instruct
3. Generate SQL responses for each prompt
4. Filter by: syntax validity (`sqlglot.parse()`), length (20–200 tokens), SQL-only output (no explanation text)
5. Deduplicate similar instructions (MinHash from Week 18)
6. Target: 2,900 passing examples

### Quality Filters for SQL Training Data

Unlike web text, SQL training data has precise quality criteria:

```python
import sqlglot

def is_valid_sql(sql: str) -> bool:
    """Check if SQL is parseable (PostgreSQL dialect)."""
    try:
        parsed = sqlglot.parse(sql, dialect="postgres")
        return len(parsed) > 0 and parsed[0] is not None
    except Exception:
        return False

def sql_quality_filter(example: dict) -> bool:
    sql = example.get('output', example.get('response', ''))
    question = example.get('instruction', example.get('question', ''))

    if not is_valid_sql(sql): return False
    tokens = sql.split()
    if not (5 <= len(tokens) <= 300): return False  # too short or too long
    if not any(kw in sql.upper() for kw in ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'WITH']): return False
    if len(question) < 10: return False  # question too short

    return True
```

### Schema Documentation in Prompts

A critical skill for text-to-SQL is schema conditioning — the model must know the table names and columns to write correct SQL. Your training examples must include schema information in the question/context:

```
System: You are a PostgreSQL expert. Write correct SQL given the schema.
User: Schema:
  CREATE TABLE sensors (
      id SERIAL PRIMARY KEY,
      name VARCHAR(100),
      location VARCHAR(50)
  );
  SELECT * FROM timescaledb_information.hypertables;  -- shows time-series tables

  Question: How many sensors are in each location?
Assistant: SELECT location, COUNT(*) AS sensor_count FROM sensors GROUP BY location ORDER BY sensor_count DESC;
```

Without schema context, the model must guess column names — useless for real deployment.

---

## Connections

**Week 18:** Data pipeline skills (filtering, deduplication) apply here with SQL-specific criteria.

**Week 24:** Qwen2.5-Coder-7B's ChatML format must match your dataset format exactly. Check the model card's `chat_template` field.

**Week 26:** This week you plan and set up the infrastructure. Week 26 is where you actually generate the 5K examples.

---

## Common Misconceptions

- **"More data always means better fine-tuning."** Quality matters more than quantity. 500 excellent hand-written PostgreSQL examples will improve SQL quality more than 10,000 GPT-generated generic SQL examples.
- **"I can use any format and convert later."** Format consistency must be established at the start. Mid-dataset format changes cause subtle bugs that are hard to detect and corrupt training.
- **"Self-Instruct is free."** Using GPT-3.5/4 API for generation has real costs. At 2,900 examples × avg 500 tokens per example = 1.45M tokens ≈ $2–15 depending on the model and provider.
- **"I should remove all ChatML control tokens from the training data."** The opposite: you must INCLUDE the control tokens (`<|im_start|>`, `<|im_end|>`, etc.) when computing the loss, but you ONLY compute loss on the assistant turn, not the user/system turns.

---

## Time Allocation (6–8 hrs)

- 1h: Read Self-Instruct paper (Sections 1–3); read Alpaca and ShareGPT format docs
- 1h: Download Spider and BIRD datasets; examine the format; write conversion scripts
- 2h: Write data pipeline: format conversion, quality filters, deduplication
- 1.5h: Write 20 hand-crafted PostgreSQL/TimescaleDB examples
- 1h: Set up Self-Instruct generation script (do not run the full generation yet — that is Week 26)
- 0.5h: Commit and document your dataset plan in `dataset_plan.md`
