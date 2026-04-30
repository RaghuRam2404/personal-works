# Week 54 TakeAway — Synthetic Data Generation

**One-liner:** Ground every teacher prompt in the target schema DDL; validate execution before saving; resume from checkpoints.

---

## Generation Prompt Template (Genie-style)

```
You are an expert PostgreSQL/TimescaleDB engineer.

Schema:
{schema_ddl}

Generate {n} text-to-SQL pairs using {target_skill} at {difficulty} difficulty.
Each pair must be executable against the schema above.
Return JSON: [{"question": "...", "sql": "..."}, ...]
Do not use these recent examples: {recent_examples}
```

## Async Pipeline Skeleton

```python
sem = asyncio.Semaphore(10)  # max concurrent calls

async def generate_safe(prompt):
    async with sem:
        for wait in [1, 2, 4, 8]:
            try:
                return await client.chat.completions.create(...)
            except RateLimitError:
                await asyncio.sleep(wait)
        return None

# Save immediately, append mode
with open("v3_raw.jsonl", "a") as f:
    json.dump(example, f)
    f.write("\n")
```

## Execution Validation

```python
def validate_sql(sql, ddl, conn):
    cur = conn.cursor()
    try:
        cur.execute("BEGIN")
        cur.execute(ddl)
        cur.execute(f"EXPLAIN {sql}")
        conn.rollback()
        return "pass"
    except Exception as e:
        conn.rollback()
        return f"fail:{str(e)[:60]}"
```

---

## Decision Rules

- If execution rate < 55% for a skill → add 3+ few-shot examples to prompt
- If parse rate < 85% → simplify JSON output format in prompt
- If duplicate rate > 20% → add recent-examples diversity instruction
- Use GPT-4o for Expert + TimescaleDB; GPT-4o-mini for Easy + Medium standard SQL
- Always generate 5–10 examples per API call (not 1) to amortize input token cost

---

## Numbers to Remember

- Rate limit safe concurrency: 10 async calls max without semaphore tuning
- Token cost rule: ~1,500 input + 2,500 output per batch of 5 examples
- Realistic usable rate from raw generation: 50–65% of total generated
- TimescaleDB target: ≥ 3,000 verified examples in v3

---

## Red Flags

- Parse rate < 70%: prompts are too complex; simplify output format
- TimescaleDB execution rate < 40%: extension not loaded in test DB, or no schema grounding
- All examples from ≤ 3 schemas: pipeline not rotating schemas per call
- Pipeline not resumable: you will lose data on any crash
