# Week 39 — Domain-Tuning Sprint Week 3: Execution-Based Evaluation

## Learning Objectives

By the end of this week, you will be able to:

- Spin up a PostgreSQL database inside a Colab notebook using Docker
- Execute model-generated SQL against live Postgres and compare result sets
- Implement an execution-based evaluation harness from scratch (or adapting sql-eval)
- Compare your model against the base model and against GPT-4o/Claude on execution correctness
- Interpret execution correctness as a more reliable metric than exact match

---

## Concepts

### 1. Why Execution-Based Eval is Critical

Exact match is a proxy metric. Two SQL queries can return identical rows while being textually different:

```sql
-- Both return the same result for a given dataset:
SELECT name FROM employees WHERE department_id = 5
SELECT e.name FROM employees e WHERE e.department_id = 5
```

Exact match would count the second query as wrong if the expected is the first. Execution-based eval counts both as correct.

Conversely, a model can generate SQL that matches exactly character-for-character but fails to execute (rare syntax error), or generates SQL that executes but returns wrong rows (semantic error). Only execution-based eval catches the semantic error.

For a production text-to-SQL system, the metric that matters is: **"Does the generated SQL return the correct rows on the actual database?"**

### 2. Architecture of the Eval Harness

```
For each test example (schema, question, expected_sql):
  1. Spin up a fresh test database with the schema
  2. Insert reference data (or use deterministic data generation)
  3. Execute expected_sql → get expected_result_set
  4. Execute model_generated_sql → get actual_result_set
  5. Compare result sets:
     - exact match: sorted(expected) == sorted(actual)
     - partial match: overlap / union of rows
  6. Record: pass/fail, error message if execution fails
```

The full harness runs ~100 examples in 2–5 minutes on Colab with a local Postgres instance.

### 3. Setting Up PostgreSQL in Colab

Colab has Docker support. The easiest approach:

**Option A — Direct PostgreSQL installation (no Docker):**
```bash
# In Colab:
!apt-get install -y postgresql > /dev/null
!service postgresql start
!sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';"
```

**Option B — Docker:**
```bash
!docker pull postgres:15
!docker run -d -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:15
import time; time.sleep(5)  # wait for Postgres to start
```

Connect from Python:
```python
import psycopg2

conn = psycopg2.connect(
    host="localhost", port=5432,
    user="postgres", password="postgres",
    database="postgres"
)
conn.autocommit = True
```

### 4. Test Data Generation Strategy

For SQL evaluation, you need actual data in the tables so that queries return non-empty result sets. Two approaches:

**Approach A — Deterministic data:** For each test schema, generate synthetic rows using a fixed seed. The expected SQL is executed on this data to get the reference output. Then the model SQL is executed on the same data.

```python
import random

def generate_test_data(cursor, create_table_sql, n_rows=100, seed=42):
    """Parse CREATE TABLE, insert random but deterministic rows."""
    random.seed(seed)
    # Execute the CREATE TABLE
    cursor.execute(create_table_sql)
    # Parse column names and types, insert n_rows of random data
    # (implementation depends on schema parser)
```

**Approach B — Use reference SQL to define correctness:** If you can't easily generate matching data, use a weaker evaluation: check that the model SQL is syntactically valid and returns the same number of columns as the expected SQL when run on the same data.

For Week 39, Approach A is preferred. The `sql-eval` repo from Defog AI implements this pattern — use it as a starting point.

### 5. Executing SQL Safely

Generated SQL can be dangerous (DROP TABLE, DELETE, etc.). Sanitize before execution:

```python
import sqlparse

def is_safe_sql(sql):
    """Only allow SELECT queries."""
    parsed = sqlparse.parse(sql.strip())
    if not parsed:
        return False
    stmt_type = parsed[0].get_type()
    return stmt_type == "SELECT"

def execute_sql_safe(cursor, sql, timeout_s=5):
    """Execute with timeout and safety check."""
    if not is_safe_sql(sql):
        return None, "Non-SELECT query rejected"
    try:
        cursor.execute(f"SET statement_timeout = {timeout_s * 1000}")
        cursor.execute(sql)
        return cursor.fetchall(), None
    except Exception as e:
        return None, str(e)
```

### 6. Comparing Result Sets

SQL result set comparison has subtleties:
- **Column order:** Usually ignore. Use frozensets of frozensets.
- **Row order:** Ignore unless the query has `ORDER BY`. Sort both sets.
- **NULL handling:** NULL != NULL in SQL, but for eval, NULL should equal NULL (match null presence).
- **Type coercion:** `1` (int) and `'1'` (string) may be different types in Python; normalize.

```python
def compare_results(expected, actual, order_matters=False):
    """Compare two result sets."""
    if expected is None or actual is None:
        return False
    
    def normalize_row(row):
        return tuple(str(v) if v is not None else "NULL" for v in row)
    
    expected_normalized = [normalize_row(r) for r in expected]
    actual_normalized = [normalize_row(r) for r in actual]
    
    if order_matters:
        return expected_normalized == actual_normalized
    else:
        return sorted(expected_normalized) == sorted(actual_normalized)
```

### 7. Interpreting Results

After running the eval harness on 100 examples:

- **Execution success rate:** % of model SQL that executes without error (syntax correctness)
- **Execution correctness:** % where result sets match (semantic correctness)
- **Error taxonomy:** Categorize execution failures (table not found, column not found, syntax error, wrong result)

Expected results for a well-trained Week 38 model:
- Execution success rate: 90–97% (most generated SQL is syntactically valid)
- Execution correctness: 60–80% (higher than exact match)
- Your model vs. base model: 60–80% vs. 20–35% execution correctness

---

## Connections

**Builds on:** Week 38 model is what you are evaluating. Week 32's held-out test set provides the test cases.

**Needed for:** Phase 5 (GRPO training uses execution correctness as the verifiable reward signal — this harness is Phase 5's reward function prototype).

---

## Common Misconceptions / Pitfalls

- **"Exact match and execution correctness are proportional."** Not always. Some models generate very different but correct SQL. Your exact match may be 55% but execution correctness 75%.
- **"I can skip setting up Postgres and just check syntax."** Syntax checking does not catch semantic errors (wrong table join, wrong aggregation). Execution is necessary for a meaningful eval.
- **"Execution errors mean the model failed."** Not always — if the model generates a valid but different SQL that happens to return wrong rows, that's a semantic error. If the model generates syntactically invalid SQL, that's a structural failure. These require different interventions.
- **"The eval harness is a one-time script."** No — this becomes your permanent evaluation infrastructure. Clean it up, parameterize it, and keep it in your repo. It is the foundation for Phase 5's reward signal.

---

## Time Allocation (6–8 hrs)

| Activity | Time |
|---|---|
| Study Defog AI's sql-eval repo (read README + key files) | 1h |
| Set up PostgreSQL in Colab (Option A or B) | 1h |
| Implement the evaluation harness | 2h |
| Run eval on 100 examples: base model, Week 33 model, Week 38 model | 1.5h |
| Write `week39_eval_report.md` with all results | 1h |
| Clean up and commit harness to GitHub | 30m |
