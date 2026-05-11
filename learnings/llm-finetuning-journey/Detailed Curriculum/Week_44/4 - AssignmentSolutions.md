# Week 44 Assignment Solutions

## Task 1 — Execution Harness Key Points

The critical parts: timeout enforcement and clean error handling.

```python
import psycopg2
from psycopg2 import sql as psql

def execute_sql(query: str, dsn: str, timeout_ms: int = 5000) -> dict:
    """Returns dict with success, rows, error, row_count."""
    stripped = query.strip()
    upper = stripped.upper()
    
    # Safety gate — only allow read-only queries
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return {"success": False, "rows": None, 
                "error": "Only SELECT/WITH allowed", "row_count": 0}
    
    try:
        conn = psycopg2.connect(dsn)
        try:
            with conn.cursor() as cur:
                # Set per-statement timeout
                cur.execute(f"SET statement_timeout = {timeout_ms}")
                cur.execute(stripped)
                rows = cur.fetchall()
                return {"success": True, "rows": rows, 
                        "error": None, "row_count": len(rows)}
        finally:
            conn.close()
    except psycopg2.errors.QueryCanceled:
        return {"success": False, "rows": None, 
                "error": "Query timed out", "row_count": 0}
    except Exception as e:
        return {"success": False, "rows": None, 
                "error": type(e).__name__ + ": " + str(e)[:200], "row_count": 0}
```

**Common gotchas:**
- `SET statement_timeout` is per-session, not per-query — set it before executing the query, not after
- `fetchall()` on a query that returned 0 rows returns `[]`, not None — `row_count = 0` is NOT the same as `success = False`
- Forgetting to close the connection leads to connection pool exhaustion when processing thousands of prompts — always use context managers or explicit `.close()`
- `psycopg2.errors.SyntaxError` is a subclass of `Exception` — your generic except should catch it, but log the specific type for analysis

---

## Task 3 — Labeling Statistics (Expected)

For a dataset with prompts from Spider/WikiSQL adapted to your schema:
- Total attempted: 3000 prompts × 2 models = 3000 pairs
- Expected discard rate: 40–60% (both wrong or both right)
- Clean pairs: typically 1200–2000 from 3000 prompts
- SFT model (v1) wins: 60–70% (your fine-tuned model should be better on domain SQL)
- Base model (Qwen) wins: 15–25% (base model occasionally generates better SQL due to its broader SQL knowledge)
- Both succeed (discarded): 10–20%
- Both fail (discarded): 20–30%

If v1 is winning less than 50% of the time, revisit your Phase 4 training — something may have gone wrong.

---

## Task 4 — Push to Hub

```python
from datasets import Dataset
import json

# Load jsonl
pairs = [json.loads(l) for l in open("week-44-prefdata/preferences.jsonl")]
ds = Dataset.from_list(pairs)

# Train/val split
split = ds.train_test_split(test_size=0.1, seed=42)

# Push
split.push_to_hub("<your-handle>/postgres-sql-preferences-v1", 
                   private=False)
```

**Expected output:** Dataset card shows: train: ~1800 rows, validation: ~200 rows. When you load it: `load_dataset("<your-handle>/postgres-sql-preferences-v1")`.

---

## How to Verify You Did It Right

1. All `chosen` examples return `success=True` from `execute_sql()` — verify with a 10-sample check.
2. None of the `chosen` examples contain DML keywords (INSERT, UPDATE, DELETE).
3. `len(dataset["train"])` ≥ 1800.
4. Dataset card clearly states: "Labels assigned by execution-based comparison on Postgres 15."
5. Run a quick sanity check: sample 5 pairs, read the prompt, read chosen and rejected. Does the chosen SQL look better for the prompt?
