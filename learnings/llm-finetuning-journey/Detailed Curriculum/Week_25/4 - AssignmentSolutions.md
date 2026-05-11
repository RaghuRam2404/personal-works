# Week 25 Assignment Solutions

## Task 1 — Key Snippet: Format Conversion

```python
SYSTEM_PROMPT = (
    "You are an expert PostgreSQL database engineer. "
    "Given a database schema and a natural language question, "
    "write a correct and efficient PostgreSQL SQL query. "
    "Output only the SQL query with no explanation."
)

def spider_to_chatml(example: dict, schema_str: str) -> dict:
    user_content = f"Schema:\n{schema_str}\n\nQuestion: {example['question']}"
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": example['query'].strip()}
        ]
    }

def alpaca_to_chatml(example: dict) -> dict:
    user_content = example['instruction']
    if example.get('input', '').strip():
        user_content = f"{user_content}\n\n{example['input'].strip()}"
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": example['output'].strip()}
        ]
    }
```

**Common gotchas:**
- Spider's `query` field vs. BIRD's `SQL` field — field names differ by dataset
- Schema must be formatted as readable CREATE TABLE statements, not as JSON metadata
- Some Spider schemas have foreign key information — include it as comments in the schema
- Empty `input` in Alpaca format — always check with `.get('input', '')` and strip

---

## Task 2 — Key Snippet: Quality Filter

```python
import sqlglot

SQLITE_ONLY_PATTERNS = [
    'GROUP_CONCAT', 'AUTOINCREMENT', 'PRAGMA ',
    'GLOB ', 'SQLITE_', 'ROWID'
]
SQL_KEYWORDS = {'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'WITH', 'ALTER'}

def filter_for_postgres_compat(sql: str) -> bool:
    sql_upper = sql.upper()
    for pattern in SQLITE_ONLY_PATTERNS:
        if pattern in sql_upper:
            return False
    return True

def sql_quality_filter(example: dict) -> bool:
    sql   = (example.get('output') or example.get('response') or
             example.get('query') or '').strip()
    question = (example.get('instruction') or example.get('question') or '').strip()

    if len(question) < 10:
        return False

    tokens = sql.split()
    if not (5 <= len(tokens) <= 400):
        return False

    if not any(kw in sql.upper() for kw in SQL_KEYWORDS):
        return False

    if not filter_for_postgres_compat(sql):
        return False

    try:
        parsed = sqlglot.parse(sql, dialect="postgres")
        if not parsed or parsed[0] is None:
            return False
    except Exception:
        return False

    return True
```

**Expected pass rates on Spider train split:**
- sqlglot parse: ~85% (some Spider queries use SQLite-specific idioms)
- Token length: ~95% of those remaining
- SQLite keyword filter: ~90% of those remaining
- Combined pass rate: ~70–80%

**Common gotchas:**
- `sqlglot` may reject valid PostgreSQL extensions — check specific error messages; use `error_level=sqlglot.ErrorLevel.RAISE` only for strict mode
- Some SQLite queries ARE valid PostgreSQL (most basic SQL) — the filter is conservative

---

## Task 4 — Key Snippet: Self-Instruct Generation

```python
import openai, time, random

client = openai.OpenAI()

def generate_instructions(seed_examples, n=5, model="gpt-3.5-turbo"):
    seed_str = "\n".join(
        f"{i+1}. {e['messages'][1]['content'][:200]}"
        for i, e in enumerate(random.sample(seed_examples, min(8, len(seed_examples))))
    )
    prompt = f"""Generate {n} diverse natural language questions that require writing a PostgreSQL query to answer.
Each question should involve a realistic database scenario (e-commerce, analytics, IoT sensors, time-series data).
Include the table schema in each question.

Inspiration examples (generate different ones):
{seed_str}

Output {n} questions, one per line, numbered 1-{n}:"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9
    )
    raw = response.choices[0].message.content
    lines = [l.strip() for l in raw.split('\n') if l.strip() and l[0].isdigit()]
    return [l[l.index('.')+1:].strip() for l in lines]

def generate_sql_response(instruction, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": instruction}
        ],
        temperature=0.1
    )
    return response.choices[0].message.content.strip()
```

**For free local generation (no API cost):**
```bash
# Install Ollama, then:
ollama pull qwen2.5-coder:7b
# Use ollama.chat() instead of openai.chat.completions.create()
```

**Expected cost with GPT-3.5:** 5 instructions + 5 responses ≈ 3,000 tokens total ≈ $0.005. Full 2,900 examples ≈ $5–15.
