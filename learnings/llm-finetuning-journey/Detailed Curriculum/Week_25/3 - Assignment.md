# Week 25 Assignment — Dataset Construction Kickoff

## Setup Checklist

- [ ] `pip install datasets sqlglot datasketch openai` (or `anthropic` if using Claude)
- [ ] Spider dataset available: `load_dataset("spider")`
- [ ] BIRD dataset available: check [bird-bench.github.io](https://bird-bench.github.io/) for download instructions
- [ ] GitHub repo with `postgres-sql-v1/` directory created

---

## Task 1 — Implement Format Conversion Pipeline

**Goal:** Convert Spider and BIRD from their native formats into ChatML format for Qwen2.5-Coder-7B.

**Requirements:**

Write `postgres-sql-v1/converters.py` with these functions:

```python
def spider_to_chatml(example: dict, schema_str: str) -> dict:
    """Convert a Spider example to ChatML format.
    
    Spider format: {'question': str, 'query': str, 'db_id': str, ...}
    Output format: ChatML messages list
    """
    ...

def bird_to_chatml(example: dict) -> dict:
    """Convert a BIRD example to ChatML format."""
    ...

def alpaca_to_chatml(example: dict) -> dict:
    """Convert Alpaca format to ChatML format."""
    ...
```

**System prompt template to use:**
```
You are an expert PostgreSQL database engineer. Given a database schema and a natural language question, write a correct and efficient PostgreSQL SQL query. Output only the SQL query with no explanation.
```

**Include schema in user message.** For Spider, the schema is in `db_schema`. Format it as:
```
Schema:
<CREATE TABLE statements>

Question: <question>
```

**Deliverable:** `converters.py` with all 3 functions, tested on 5 examples each.

**Acceptance criteria:**
- Each output has `messages` key with list of `{"role": ..., "content": ...}` dicts
- System, user, and assistant roles are all present
- Schema appears in the user message for Spider/BIRD examples

---

## Task 2 — Quality Filter Implementation

**Goal:** Write a reusable SQL quality filter.

**Requirements:**

Write `postgres-sql-v1/quality_filter.py` with:

```python
def sql_quality_filter(example: dict) -> bool:
    """Return True if example should be kept."""
    ...

def filter_for_postgres_compat(sql: str) -> bool:
    """Return True if SQL is compatible with PostgreSQL (no SQLite-only syntax)."""
    ...

def run_quality_pipeline(examples: list) -> tuple[list, dict]:
    """Apply all filters, return (kept_examples, rejection_stats)."""
    ...
```

Filters to implement:
1. `sqlglot.parse(sql, dialect="postgres")` — reject if parse fails
2. `len(sql.split()) >= 5` — reject if fewer than 5 tokens
3. `len(sql.split()) <= 400` — reject if more than 400 tokens
4. No SQLite-specific syntax: `GROUP_CONCAT`, `LIMIT ... OFFSET` as a standalone clause (SQLite style), `AUTOINCREMENT`
5. Must contain at least one SQL keyword: SELECT, INSERT, UPDATE, DELETE, CREATE, WITH
6. Question length >= 10 characters

**Test on 100 Spider examples and report:** how many pass each filter (cumulative), what is the overall pass rate.

**Deliverable:** `quality_filter.py` + a report section in `postgres-sql-v1/dataset_plan.md`.

---

## Task 3 — Hand-Write 20 PostgreSQL/TimescaleDB Examples

**Goal:** Create the highest-quality 20 examples in your dataset — ones you personally wrote and verified.

**Requirements:**

Create `postgres-sql-v1/hand_written_examples.jsonl` with 20 examples in ChatML format covering:
- 5 basic TimescaleDB queries (time_bucket, continuous aggregates)
- 5 PostgreSQL-specific queries (ON CONFLICT, RETURNING, JSONB, window functions)
- 5 multi-table JOIN queries using realistic schemas
- 5 "advanced" queries (recursive CTE, LATERAL join, or correlated subquery)

Each example must:
- Include the schema in the user message (CREATE TABLE statements)
- Have a question that a junior developer might ask about a real database
- Have correct, executable PostgreSQL SQL as the response
- NOT have any explanation in the response — SQL only

**Verify**: Run each SQL through `sqlglot.parse(sql, dialect="postgres")` — all must parse without errors.

**Deliverable:** `hand_written_examples.jsonl` (20 lines)

---

## Task 4 — Set Up Self-Instruct Generation Script

**Goal:** Build the generation infrastructure for Week 26.

**Requirements:**

Write `postgres-sql-v1/self_instruct.py` with:

```python
def generate_instructions(
    seed_examples: list,       # hand-written examples as seeds
    n_to_generate: int,        # how many new instructions to generate
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.9
) -> list:
    """Use an LLM to generate diverse SQL instructions."""
    ...

def generate_responses(
    instructions: list,        # list of question + schema strings
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.1   # low temp for SQL — be precise
) -> list:
    """Use an LLM to generate SQL responses for each instruction."""
    ...
```

- You can use `openai` library with `gpt-3.5-turbo` OR use Ollama with `qwen2.5-coder:7b` locally for free
- Implement rate limiting: `time.sleep(0.5)` between API calls
- Implement error handling: if a generation fails, skip and log

**Test with 5 instructions** (cost: <$0.05 with GPT-3.5, free with Ollama).

**Deliverable:** `self_instruct.py` that generates 5 test examples end-to-end.

GitHub commit: `week-25-dataset-v1-kickoff`

---

## Stretch Goals

- Implement de-duplication of your generated instructions using MinHash from Week 18
- Write a `verify_sql.py` that actually executes the SQL against a local PostgreSQL instance (or a Docker PostgreSQL container) and checks for runtime errors
- Add a quality scoring prompt: for each generated (question, SQL) pair, use the LLM to score it 1–5 on "Would this be useful training data?" Keep only 4–5 scored examples
