# Week 39 TakeAway — Execution-Based Evaluation

Execute generated SQL against real Postgres; compare result sets, not token sequences.

---

## Key Code Patterns

**PostgreSQL setup in Colab (one-time):**
```bash
apt-get install -y postgresql > /dev/null
service postgresql start
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
```

**Safe SQL execution with timeout:**
```python
import sqlparse

def is_safe_sql(sql):
    parsed = sqlparse.parse(sql.strip())
    if not parsed:
        return False
    return parsed[0].get_type() == "SELECT"

def execute_safe(cursor, sql, timeout_ms=5000):
    if not is_safe_sql(sql):
        return None, "Non-SELECT rejected"
    try:
        cursor.execute(f"SET statement_timeout = {timeout_ms}")
        cursor.execute(sql)
        return cursor.fetchall(), None
    except Exception as e:
        return None, str(e)
```

**Result set comparison:**
```python
def compare_results(expected, actual):
    if expected is None or actual is None:
        return False
    def norm(row):
        return tuple("NULL" if v is None else str(v) for v in row)
    return sorted(norm(r) for r in expected) == sorted(norm(r) for r in actual)
```

**Schema isolation per example:**
```python
cur.execute("DROP SCHEMA IF EXISTS eval_tmp CASCADE")
cur.execute("CREATE SCHEMA eval_tmp")
cur.execute("SET search_path TO eval_tmp")
cur.execute(schema_sql)   # create tables
```

---

## Decision Rules

- If exact match < execution correctness: your model is generating valid but differently-phrased SQL — this is good; report execution correctness as your headline metric.
- If execution correctness < execution success by more than 20 points: the model generates syntactically valid but semantically wrong SQL — analyze failure categories, add targeted training data.
- If execution success < 85%: the model has structural issues (wrong table/column references) — check whether the schema was properly included in the prompt during training.
- If a CTE query is rejected by `is_safe_sql`: update the check to scan for DML tokens rather than relying on top-level statement type.

---

## Numbers to Remember

- Expected exec success rate for a well-trained Week 38 model: 90–97%
- Expected exec correctness for Week 38 model: 60–80%
- Base model (no fine-tuning) exec correctness: 20–35%
- Statement timeout: 5000 ms (5 seconds) per query
- Test set size: 100 examples from `held_out_test.json`
- Harness runtime: 2–5 minutes for 100 examples on Colab with local Postgres

---

## Red Flags

- Exec correctness equals exact match exactly: your comparison function may have a bug; they should not be identical.
- Exec success > 99% but correctness < 50%: the model is generating trivially valid SQL (`SELECT 1`) for hard cases — check model output samples manually.
- DROP or CREATE appearing in model-generated SQL that passes `is_safe_sql`: your safety check has a flaw; the harness could destroy test schemas.
- `psycopg2.OperationalError: SSL connection required`: wrong connection string for Colab local Postgres — remove `sslmode` or set it to `disable`.
- Schema not found errors in model SQL: the model is not seeing the schema in its prompt — recheck the prompt template used for inference.
