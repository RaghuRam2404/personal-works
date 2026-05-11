# Week 25 TakeAway — Dataset Construction: Formats and Planning

**One-liner:** ChatML format, loss on assistant turn only, 3-tier dataset (Spider+BIRD → hand-written → self-instruct), SQL validity via sqlglot.

---

## Dataset Format Reference

```python
# ChatML format — what Qwen2.5-Coder expects
example = {
    "messages": [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
        {"role": "assistant", "content": sql_query_only}  # NO explanation
    ]
}

# SYSTEM_PROMPT for SQL assistant:
SYSTEM_PROMPT = (
    "You are an expert PostgreSQL database engineer. "
    "Given a database schema and a natural language question, "
    "write a correct and efficient PostgreSQL SQL query. "
    "Output only the SQL query with no explanation."
)
```

---

## Dataset Plan (v1 = 5,000 examples)

| Tier | Source | Count | Quality |
|---|---|---|---|
| 1 | Spider + BIRD (filtered, converted) | 2,000 | High |
| 2 | Hand-written PostgreSQL/TimescaleDB | 100 | Highest |
| 3 | Self-Instruct synthetic | 2,900 | Medium-High |

---

## Key Code Pattern — SQL Quality Filter

```python
import sqlglot

SQLITE_ONLY = ['GROUP_CONCAT', 'AUTOINCREMENT', 'PRAGMA ', 'ROWID']
SQL_KEYWORDS = {'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'WITH'}

def sql_quality_filter(sql, question):
    if len(question) < 10: return False
    tokens = sql.split()
    if not (5 <= len(tokens) <= 400): return False
    sql_up = sql.upper()
    if any(p in sql_up for p in SQLITE_ONLY): return False
    if not any(kw in sql_up for kw in SQL_KEYWORDS): return False
    try:
        sqlglot.parse(sql, dialect="postgres")
        return True
    except Exception:
        return False
```

---

## SQLite → PostgreSQL Conversion Table

| SQLite | PostgreSQL |
|---|---|
| `GROUP_CONCAT(col, ',')` | `STRING_AGG(col, ',')` |
| `AUTOINCREMENT` | `SERIAL` or `GENERATED ALWAYS AS IDENTITY` |
| `strftime('%Y', d)` | `EXTRACT(YEAR FROM d)` |
| `LIMIT 10, 5` | `LIMIT 10 OFFSET 5` |
| `INSERT OR REPLACE` | `INSERT ... ON CONFLICT DO UPDATE` |

---

## Decision Rules

- Loss on assistant turn only → use `DataCollatorForCompletionOnlyLM` in TRL
- Include schema in user message → the model cannot write correct SQL without it
- Strip all explanation from assistant turn → deployment SQL must be executable directly
- Up-weight hand-written examples → repeat 3–5× in training mix
- Validate with sqlglot before adding any example → zero tolerance for invalid SQL in training

---

## Red Flags

- Training loss is stuck high → check that assistant tokens are not masked out (loss on wrong tokens)
- Model outputs "Sure! Here is your query: ..." → explanation text leaked into training data
- Model generates invalid PostgreSQL syntax → SQLite-specific examples slipped through filter
- Duplicate SQL for different questions → deduplication step missing
