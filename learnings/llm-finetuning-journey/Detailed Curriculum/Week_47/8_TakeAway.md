# Week 47 TakeAway — SQL Reward Function Design

**One-liner:** Design rewards with multiple levels and explicit anti-hack guards; test on 100 completions before connecting to GRPO.

---

## Reward Hierarchy

| Level | Reward | Condition |
|---|---|---|
| Syntax error / no SQL | 0.0 | Parse failure or extract_sql returns None |
| Anti-hack blocked | 0.0 | info_schema, pg_catalog, no FROM, too many rows |
| Runtime error | 0.1 | Executes as valid SQL but fails on DB (wrong table/col) |
| Executes + rows | 0.2 | Runs without error, returns ≥ 1 row |
| Correct row count | 0.5 | Row count == expected count |
| Exact match | 1.0 | sorted(rows) == sorted(expected) |

---

## SQL Extraction Pattern

```python
def extract_sql(text: str) -> str | None:
    # Try ```sql fence first
    m = re.search(r'```(?:sql|SQL)?\s*\n(.*?)\n```', text, re.DOTALL)
    if m and m.group(1).strip().upper().startswith(("SELECT", "WITH")):
        return m.group(1).strip()
    # Try bare SQL
    m = re.search(r'(?:^|\n)((?:SELECT|WITH)\b.*?)(?:\n\n|\Z)', text, 
                  re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None
```

---

## Anti-Hack Guards (mandatory)

```python
BLOCKED = {"INFORMATION_SCHEMA", "PG_CATALOG", "PG_STAT"}
if any(b in sql.upper() for b in BLOCKED): return 0.0
if result["row_count"] > max(5 * len(expected), 10): return 0.1
if "FROM" not in sql.upper(): return 0.0  # catches SELECT NULL, SELECT 1
```

---

## Reward Function Signature for TRL GRPO

```python
def sql_reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    # completions and prompts are flattened: [prompt1_comp1, prompt1_comp2, ..., prompt2_comp1, ...]
    # Return list of same length as completions
    return [_score_sql(extract_sql(c), kwargs.get("expected"), db_dsn) for c in completions]
```

---

## Diagnostic Goals (100 prompts × 4 completions)

| Metric | Target |
|---|---|
| % reward 0.0 | 20–40% |
| % reward 0.1 | 10–25% |
| % reward 0.2+ | ≥ 30% |
| % reward 1.0 | ≥ 10% |
| % zero-gradient prompts | < 50% |

---

## Red Flags

- 0.0 rate > 70%: model not generating SQL — check output format first
- 0.0 rate drops to 0 after 100 steps: model found hack — audit reward function
- 1.0 rate jumps to 80%+ suddenly: test set contamination or reward function bug
- All rewards identical in a group: shaped reward levels too coarse — add more levels
- Reasoning bonus > 10% of max reward: will cause verbosity hacking
