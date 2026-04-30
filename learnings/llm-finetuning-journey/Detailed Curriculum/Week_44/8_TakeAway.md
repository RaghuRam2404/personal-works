# Week 44 TakeAway — SQL Preference Dataset

**One-liner:** Execution-based labeling uses Postgres as the judge — no humans, no subjectivity, ground truth for free.

---

## Execution Harness Pattern

```python
def execute_sql(query: str, dsn: str, timeout_ms=5000) -> dict:
    if not query.strip().upper().startswith(("SELECT", "WITH")):
        return {"success": False, "rows": None, "error": "DML blocked", "row_count": 0}
    try:
        with psycopg2.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {timeout_ms}")
                cur.execute(query.strip())
                rows = cur.fetchall()
                return {"success": True, "rows": rows, "error": None, 
                        "row_count": len(rows)}
    except Exception as e:
        return {"success": False, "rows": None, "error": str(e)[:200], "row_count": 0}
```

---

## Labeling Decision Table

| SQL_A executes | SQL_B executes | Reference match | Action |
|---|---|---|---|
| Yes | No | — | A=chosen, B=rejected |
| No | Yes | — | B=chosen, A=rejected |
| Yes | Yes | A matches ref | A=chosen, B=rejected |
| Yes | Yes | B matches ref | B=chosen, A=rejected |
| Both match | Both match | — | Discard (no signal) |
| Neither | Neither | — | Discard |

---

## Dataset Format

```python
{
    "prompt": "Show all orders from last month",
    "chosen": "SELECT * FROM orders WHERE created_at >= NOW() - INTERVAL '1 month'",
    "rejected": "SELECT order_id FROM orders WHERE date > '2024-01-01'"  # wrong/broken
}
```

---

## Decision Rules

- If discard rate > 70%: prompt distribution mismatches schema — fix prompts first
- If v1 wins < 50%: SFT training had problems — revisit Phase 4 before continuing
- Do NOT include pairs where chosen does not actually execute — this poisons DPO
- Quality > quantity: 2000 clean pairs >> 10,000 noisy pairs
- For ambiguous pairs (both execute, different rows): use reference SQL to resolve

---

## Numbers to Remember

- Target dataset size: ≥ 2000 clean preference pairs
- Expected discard rate: 40–60% (normal)
- Statement timeout: 5000ms per query
- HF Hub schema: 3 required columns — `prompt`, `chosen`, `rejected`
- SQL safety gate: only allow SELECT or WITH statements

---

## Red Flags

- chosen SQL returns empty rows (row_count = 0): not actually good SQL — add row_count > 0 requirement
- Both models fail on >50% of prompts: schema mismatch — prompts reference non-existent tables
- Dataset has no JOIN examples: prompt set is too simple — add complex query prompts
- Pushing to HF Hub fails: `huggingface-cli login` may be needed, or token scopes may be insufficient
